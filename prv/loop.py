"""The Plan-Retrieve-Verify-Counter-Retrieve loop (paper Algorithm 1) and trajectory printing."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import tools
from .data import Question, Video, summarize
from .llm import STOP, Answer, CallRecord, Hypothesis, LLMClient, MockController, Plan, ToolCall
from .tools import Evidence
from .workspace import Workspace


@dataclass
class Step:
    """One tool call with the evidence it returned and the polarity edges it added."""

    index: int
    stage: str  # "retrieve" | "counter-retrieve"
    call: ToolCall
    found: list[Evidence] = field(default_factory=list)
    new_nodes: list[int] = field(default_factory=list)
    verifications: list[dict] = field(default_factory=list)  # contradiction_verifier outputs
    temporal: dict | None = None  # temporal_checker output
    verified: bool = False  # True when a hypothesis update followed this step
    hypothesis: str | None = None  # hypothesis after the verify update


@dataclass
class Trajectory:
    """Recorded tool-use trace tau plus the final workspace and answer."""

    video_id: str
    question: Question
    plan: Plan
    steps: list[Step]
    workspace: Workspace
    answer: Answer
    stop_reason: str

    def tool_sequence(self) -> list[tuple[str, str]]:
        return [(s.stage, s.call.tool) for s in self.steps]


def _execute(call: ToolCall, video: Video, ws: Workspace, step: Step) -> None:
    """Run one tool, store its evidence, and update polarity / temporal edges."""
    a = call.args
    region = tuple(a["region"]) if a.get("region") else None
    t = call.tool
    if t == "visual_retriever":
        found = tools.visual_retriever(video, a["query"], region)
    elif t == "audio_retriever":
        found = tools.audio_retriever(video, a["query"], region)
    elif t == "asr_parser":
        found = tools.asr_parser(video, region, a.get("query"))
    elif t == "ocr_extractor":
        found = tools.ocr_extractor(video, region, a.get("query"))
    elif t == "temporal_checker":
        ea, eb = ws.nodes[a["a"]], ws.nodes[a["b"]]
        res = tools.temporal_checker(ea, eb)
        ws.add_edge(a["a"], a["b"], "temporal", res.relation)
        step.temporal = {"a": ea.ref, "b": eb.ref, "relation": res.relation}
        return
    else:
        raise ValueError(f"unknown tool: {t}")
    step.found = found
    for ev in found:
        nid, is_new = ws.add(ev)
        if not is_new:
            continue
        step.new_nodes.append(nid)
        for other in list(ws.nodes):
            if other == nid:
                continue
            res = tools.contradiction_verifier(ws.nodes[other], ev)
            step.verifications.append({"a": ws.ref(other), "b": ev.ref, "label": res.label})
            if res.label != "neutral":
                ws.add_edge(other, nid, res.label)


def run_prv(video: Video, question: Question, llm: LLMClient | None = None,
            max_steps: int = 8, tau_r: float = 0.3, max_counter: int = 2) -> Trajectory:
    """Answer one question with the closed loop. Returns the full recorded trajectory."""
    llm = llm or MockController()
    plan = llm.plan(question.text, summarize(video))
    ws = Workspace()
    h = Hypothesis(plan.h0)
    steps: list[Step] = []
    history: list[CallRecord] = []
    stop_reason = "step budget reached"

    def do(call: ToolCall, stage: str) -> Step:
        step = Step(len(steps) + 1, stage, call)
        _execute(call, video, ws, step)
        steps.append(step)
        history.append(CallRecord(call, stage, len(step.found)))
        return step

    while len(steps) < max_steps:
        call = llm.act(question.text, plan, ws, history, "retrieve")
        if call.tool == STOP:
            stop_reason = "controller stop"
            break
        step = do(call, "retrieve")
        n_counter = 0
        while ws.unresolved_contradictions() and n_counter < max_counter and len(steps) < max_steps:
            call = llm.act(question.text, plan, ws, history, "counter")
            if call.tool == STOP:
                break
            step = do(call, "counter-retrieve")
            n_counter += 1
        h = llm.verify(question.text, ws, h)
        step.verified, step.hypothesis = True, h.answer
    answer = llm.synthesize(question.text, ws, h, tau_r)
    return Trajectory(video.id, question, plan, steps, ws, answer, stop_reason)


def format_trajectory(tr: Trajectory) -> str:
    """Readable multi-line trace of a trajectory."""
    ws, p = tr.workspace, tr.plan
    reg = f"{p.region[0]:g}-{p.region[1]:g}s" if p.region else "whole video"
    lines = [
        f"Q: {tr.question.text}   [video {tr.video_id}, synthetic]",
        f"[Plan] modalities={p.needed_modalities()} region={reg} h0={p.h0!r} need_counter={p.need_counter}",
    ]
    last_h = p.h0
    for s in tr.steps:
        a = s.call.args
        if "region" in a:
            r = a["region"]
            where = f"region={r[0]:g}-{r[1]:g}s" if r else "region=whole video"
        else:
            where = f"nodes={ws.ref(a['a'])}, {ws.ref(a['b'])}"
        lines.append(f"[Step {s.index} {s.stage}] {s.call.tool}({where}) -> {len(s.found)} evidence")
        for ev in s.found:
            lines.append(f"    + {ev.ref}  \"{ev.text}\"")
        for v in s.verifications:
            if v["label"] != "neutral":
                lines.append(f"    verify: {v['a']} <-{v['label']}-> {v['b']}")
        if s.temporal:
            lines.append(f"    temporal: {s.temporal['a']} is {s.temporal['relation']} {s.temporal['b']}")
        if s.verified and s.hypothesis != last_h:
            last_h = s.hypothesis
            lines.append(f"    hypothesis -> {s.hypothesis!r}")
    gone = ws.refuted()
    if gone:
        lines.append("[Verify] outvoted: " + ", ".join(ws.ref(n) for n in sorted(gone)))
    for e in ws.unresolved_contradictions():
        lines.append(f"[Verify] unresolved contradiction: {ws.ref(e.a)} vs {ws.ref(e.b)}")
    ans = tr.answer
    if ans.abstain:
        lines.append(f"[Answer] ABSTAIN (c={ans.confidence:.2f}): {ans.reason}")
    else:
        lines.append(f"[Answer] {ans.text!r} (c={ans.confidence:.2f})")
    lines.append("    evidence: " + ", ".join(e.ref for e in ans.evidence))
    return "\n".join(lines)
