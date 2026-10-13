"""LLM controller interface and a deterministic rule-based mock.

The paper's controller is an LLM that writes the evidence plan, picks tools, updates
the hypothesis and synthesises the answer (Sec. 3.2 - 3.4). Here `LLMClient` is the
interface, and `MockController` fills the four roles with fixed rules so the demo runs
offline and gives the same output every time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .text import content_tokens, polarity, words
from .tools import ALL_TOOLS, MODALITIES, MODALITY_FOR_TOOL, TOOL_FOR_MODALITY, Evidence, Region
from .workspace import Workspace

STOP = "STOP"
YES_NO_STARTS = {"is", "are", "was", "were", "does", "do", "did", "can", "has", "have"}
TEMPORAL_CUES = {"before", "after", "first", "earlier", "later", "then", "until"}
# Question cues (stemmed) that raise the priority of one modality.
CUES = {
    "asr": {"say", "said", "announce", "mention", "tell", "told", "speak", "spoken", "narrator"},
    "aud": {"sound", "hear", "heard", "noise", "music", "audio", "beep", "applause"},
    "ocr": {"sign", "text", "written", "label", "display", "banner", "price", "date", "caption"},
    "vis": {"see", "seen", "show", "shown", "wear", "wearing", "look", "color", "holding", "visible"},
}
BASE_PRIORITY = {"vis": 0.6, "asr": 0.6, "ocr": 0.5, "aud": 0.1}
ANSWER_MODALITY_ORDER = ("asr", "ocr", "vis", "aud")


@dataclass
class Plan:
    """Evidence plan P = (m, R_temp, h0, need_counter) from paper Eq. (1)."""

    priority: dict[str, float]  # tool name -> priority weight (the vector m)
    region: Region  # temporal region of interest; None means the whole video
    h0: str | None
    need_counter: bool

    def needed_modalities(self) -> list[str]:
        """Retrieval modalities with priority >= 0.5, highest priority first."""
        mods = [m for m in MODALITIES if self.priority[TOOL_FOR_MODALITY[m]] >= 0.5]
        return sorted(mods, key=lambda m: -self.priority[TOOL_FOR_MODALITY[m]])

    def to_dict(self) -> dict:
        return {"priority": {k: round(v, 3) for k, v in self.priority.items()},
                "needed_modalities": self.needed_modalities(),
                "region": list(self.region) if self.region else None,
                "h0": self.h0, "need_counter": self.need_counter}


@dataclass
class ToolCall:
    """A tool name (or STOP) with its arguments."""

    tool: str
    args: dict = field(default_factory=dict)


@dataclass
class CallRecord:
    """What the controller may look at about an earlier call."""

    call: ToolCall
    stage: str
    n_found: int


@dataclass
class Hypothesis:
    answer: str | None
    support: list[int] = field(default_factory=list)  # workspace node ids behind the answer


@dataclass
class Answer:
    """Final output (y, E, c, r) of the framework."""

    text: str | None
    confidence: float
    abstain: bool
    reason: str
    evidence: list[Evidence] = field(default_factory=list)


class LLMClient(Protocol):
    """The four roles the controller plays in Algorithm 1."""

    def plan(self, question: str, summary: list[tuple[float, float, str]]) -> Plan: ...

    def act(self, question: str, plan: Plan, ws: Workspace, history: list[CallRecord],
            mode: str) -> ToolCall: ...

    def verify(self, question: str, ws: Workspace, h: Hypothesis) -> Hypothesis: ...

    def synthesize(self, question: str, ws: Workspace, h: Hypothesis, tau_r: float) -> Answer: ...


def is_yes_no(question: str) -> bool:
    ws = words(question)
    return bool(ws) and ws[0] in YES_NO_STARTS


def stance(question: str, text: str) -> int:
    """+1 if the text agrees with the polar claim in the question, -1 if it opposes, 0 if unrelated."""
    pq, pt = polarity(question), polarity(text)
    if not pq:
        return 1
    return sum(pq[g] * pt[g] for g in pq.keys() & pt.keys())


class MockController:
    """Rule-based stand-in for the LLM controller."""

    # --- Plan -------------------------------------------------------------
    def plan(self, question: str, summary: list[tuple[float, float, str]]) -> Plan:
        q = set(content_tokens(question))
        priority = {}
        for mod, tool in TOOL_FOR_MODALITY.items():
            cued = bool(q & CUES[mod])
            priority[tool] = min(1.0, BASE_PRIORITY[mod] + (0.4 if cued else 0.0))
        yes_no = is_yes_no(question)
        priority["temporal_checker"] = 1.0 if set(words(question)) & TEMPORAL_CUES else 0.3
        priority["contradiction_verifier"] = 1.0 if yes_no else 0.5
        assert set(priority) == set(ALL_TOOLS)
        # Region of interest: the summary lines that share the most words with the question.
        scores = [len(q & set(content_tokens(text))) for _, _, text in summary]
        best = max(scores, default=0)
        region = None
        if best > 0:
            hit = [(s, e) for (s, e, _), sc in zip(summary, scores) if sc == best]
            region = (min(s for s, _ in hit), max(e for _, e in hit))
        return Plan(priority, region, "yes" if yes_no else None, need_counter=yes_no)

    # --- Act (tool selection) --------------------------------------------
    def act(self, question: str, plan: Plan, ws: Workspace, history: list[CallRecord], mode: str) -> ToolCall:
        ran = {(MODALITY_FOR_TOOL[r.call.tool], r.call.args.get("region") is None): r.n_found
               for r in history if r.call.tool in MODALITY_FOR_TOOL}
        done = {m for m, _ in ran}
        needed = plan.needed_modalities()
        if mode == "counter":
            return self._counter(question, plan, ws, ran, done, needed)
        for mod in needed:
            if mod not in done:
                return self._call(mod, question, plan.region)
        if not ws.unresolved_contradictions() and not any(r.call.tool == "temporal_checker" for r in history):
            for e in ws.edges:
                if e.kind == "support":
                    return ToolCall("temporal_checker", {"a": e.a, "b": e.b})
        return ToolCall(STOP)

    @staticmethod
    def _call(mod: str, query: str, region: Region) -> ToolCall:
        args: dict = {"region": list(region) if region else None}
        if mod in ("vis", "aud") or region is None:
            args["query"] = query
        return ToolCall(TOOL_FOR_MODALITY[mod], args)

    def _counter(self, question, plan, ws, ran, done, needed) -> ToolCall:
        """Pick a search for counter-evidence over the whole video, using the disputed texts as query."""
        edge = ws.unresolved_contradictions()[0]
        query = f"{question} {ws.nodes[edge.a].text} {ws.nodes[edge.b].text}"
        cands = [m for m in needed if m not in done]
        cands += [m for m in MODALITIES if m not in needed and m not in done]
        cands += [m for m in needed if ran.get((m, False)) == 0 and (m, True) not in ran]
        return self._call(cands[0], query, None) if cands else ToolCall(STOP)

    # --- Verify (hypothesis update) --------------------------------------
    def verify(self, question: str, ws: Workspace, h: Hypothesis) -> Hypothesis:
        gone = ws.refuted()
        live = [n for n in ws.nodes if n not in gone]
        if is_yes_no(question):
            pos = [n for n in live if stance(question, ws.nodes[n].text) > 0]
            neg = [n for n in live if stance(question, ws.nodes[n].text) < 0]
            mp, mn = ws.modality_count(set(pos)), ws.modality_count(set(neg))
            if mp > mn:
                return Hypothesis("yes", pos)
            if mn > mp:
                return Hypothesis("no", neg)
            return Hypothesis(None, [])
        q = set(content_tokens(question))
        comps = [c for c in ws.components() if any(q & set(content_tokens(ws.nodes[i].text)) for i in c)]
        if not comps:
            return Hypothesis(None, [])

        def rank(c: set[int]) -> int:
            return min(ANSWER_MODALITY_ORDER.index(ws.nodes[i].modality) for i in c)

        best = max(comps, key=lambda c: (ws.modality_count(c), -rank(c), -min(c)))
        pick = min(best, key=lambda i: (ANSWER_MODALITY_ORDER.index(ws.nodes[i].modality), i))
        return Hypothesis(ws.nodes[pick].text, sorted(best))

    # --- Synthesize (answer, confidence, abstention) ---------------------
    def synthesize(self, question: str, ws: Workspace, h: Hypothesis, tau_r: float) -> Answer:
        conflict = ws.unresolved_contradictions()
        if conflict:
            # Confidence from the best-supported side of a dispute, cut by a fixed penalty.
            n = max(ws.modality_count(ws.side(i)) for e in conflict for i in (e.a, e.b))
            conf = n / (n + 1) * 0.2
            reason = "contradictory evidence that counter-retrieval did not resolve"
        else:
            n = ws.modality_count(set(h.support)) if h.support else 0
            conf = n / (n + 1)
            reason = "" if h.support else "no evidence found"
        abstain = conf < tau_r
        if abstain:
            ids = sorted({i for e in conflict for i in (e.a, e.b)}) if conflict else []
            return Answer(None, conf, True, reason or "low confidence", [ws.nodes[i] for i in ids])
        return Answer(h.answer, conf, False, "", [ws.nodes[i] for i in h.support])
