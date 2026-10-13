"""Counterfactual tool-use trace supervision (paper Sec. 3.6), data side only.

Given a recorded trajectory, apply one of five perturbations to the video, re-run the
controller on the perturbed video, and emit a supervision record with the decision labels
for planning, tool choice, verification and the final answer / abstention.

The paper names the five types (audio-swap, visual-swap, temporal-shuffle, same-entity
distractor, pseudo-support) and trains the controller so it does not keep its original
trace on corrupted evidence. It does not spell out an implementation, so each type below
is this demo's own operationalisation on the toy JSON. No loss is computed here.
"""
from __future__ import annotations

import json
from typing import Callable

from .data import Question, Video
from .llm import LLMClient
from .loop import Trajectory, run_prv
from .text import content_tokens, flip_text

PERTURBATION_TYPES = ("audio_swap", "visual_swap", "temporal_shuffle", "same_entity_distractor", "pseudo_support")
FIELD_FOR_MODALITY = {"vis": "visual", "aud": "audio_events", "asr": "asr", "ocr": "ocr"}

Edit = dict
PerturbFn = Callable[[Video, Trajectory, Video], tuple[Video, list[Edit]]]


# --- Decision labels ---------------------------------------------------------
def decision_labels(tr: Trajectory) -> dict:
    """Decision labels of a trajectory: plan, tool choice, verification, final."""
    ws = tr.workspace
    pairs = sorted(({"a": v["a"], "b": v["b"], "label": v["label"]} for s in tr.steps for v in s.verifications),
                   key=lambda d: (d["a"], d["b"]))
    return {
        "plan": {k: v for k, v in tr.plan.to_dict().items() if k in ("needed_modalities", "region", "need_counter")},
        "tool_choice": [{"stage": s.stage, "tool": s.call.tool, "region": s.call.args.get("region"),
                         **({"nodes": [ws.ref(s.call.args["a"]), ws.ref(s.call.args["b"])]}
                            if s.call.tool == "temporal_checker" else {})}
                        for s in tr.steps],
        "verification": {
            "pairs": pairs,
            "temporal": [s.temporal for s in tr.steps if s.temporal],
            "outvoted": sorted(ws.ref(n) for n in ws.refuted()),
            "unresolved": [[ws.ref(e.a), ws.ref(e.b)] for e in ws.unresolved_contradictions()],
        },
        "final": {"abstain": tr.answer.abstain, "answer": tr.answer.text},
    }


# --- Helpers -------------------------------------------------------------------
def _seg_index(video: Video, start: float) -> int:
    return next(i for i, s in enumerate(video.segments) if s.start == start)


def _evidence_segments(video: Video, tr: Trajectory, modalities: tuple[str, ...]) -> list[int]:
    """Segment indices of trajectory evidence in the given modalities (all evidence if none match)."""
    nodes = list(tr.workspace.nodes.values())
    pick = [n for n in nodes if n.modality in modalities] or nodes
    return sorted({_seg_index(video, n.start) for n in pick})


def _set(video: Video, idx: int, name: str, value, edits: list[Edit]) -> None:
    seg = video.segments[idx]
    edits.append({"segment": idx, "field": name, "before": getattr(seg, name), "after": value})
    setattr(seg, name, value)


def _donor_segment(donor: Video, idx: int):
    return donor.segments[idx % len(donor.segments)]


def _free_slot(video: Video, anchor: int, name: str, allow_anchor: bool = True) -> int | None:
    """Nearest segment (ties: later one) whose field `name` is empty."""
    free = [i for i, s in enumerate(video.segments) if not getattr(s, name) and (allow_anchor or i != anchor)]
    return min(free, key=lambda i: (abs(i - anchor), -i)) if free else None


# --- The five perturbations ----------------------------------------------------
def audio_swap(video: Video, tr: Trajectory, donor: Video) -> tuple[Video, list[Edit]]:
    """Replace speech and audio events of the evidence segments with those of a donor video."""
    out, edits = video.clone(), []
    for i in _evidence_segments(video, tr, ("asr", "aud")):
        d = _donor_segment(donor, i)
        _set(out, i, "asr", d.asr, edits)
        _set(out, i, "audio_events", list(d.audio_events), edits)
    return out, edits


def visual_swap(video: Video, tr: Trajectory, donor: Video) -> tuple[Video, list[Edit]]:
    """Replace the visual caption of the evidence segments with the donor video's caption."""
    out, edits = video.clone(), []
    for i in _evidence_segments(video, tr, ("vis",)):
        _set(out, i, "visual", _donor_segment(donor, i).visual, edits)
    return out, edits


def temporal_shuffle(video: Video, tr: Trajectory, donor: Video) -> tuple[Video, list[Edit]]:
    """Reverse the order of segment contents; timestamps stay in place."""
    out = video.clone()
    n = len(video.segments)
    perm = list(range(n - 1, -1, -1))
    for i, j in enumerate(perm):
        src = video.segments[j]
        dst = out.segments[i]
        dst.visual, dst.asr, dst.ocr = src.visual, src.asr, src.ocr
        dst.audio_events = list(src.audio_events)
    return out, [{"kind": "reverse_order", "permutation": perm}]


def same_entity_distractor(video: Video, tr: Trajectory, donor: Video) -> tuple[Video, list[Edit]]:
    """Add a neutral line that names the same entity but states nothing about the asked fact."""
    out, edits = video.clone(), []
    first = min(tr.workspace.nodes.values(), key=lambda n: n.start)
    anchor = _seg_index(video, first.start)
    q = content_tokens(tr.question.text)
    ev = set(content_tokens(first.text))
    entity = " ".join([t for t in q if t in ev][:2]) or "the same entity"
    for name in ("ocr", "asr"):
        slot = _free_slot(out, anchor, name)
        if slot is not None:
            _set(out, slot, name, f"{entity} mentioned in an unrelated notice", edits)
            break
    return out, edits


def pseudo_support(video: Video, tr: Trajectory, donor: Video) -> tuple[Video, list[Edit]]:
    """Insert fabricated evidence that looks like support for the opposite answer."""
    out, edits = video.clone(), []
    ws = tr.workspace
    gone = ws.refuted()
    live = [n for n in sorted(ws.nodes) if n not in gone]
    node = ws.nodes[live[-1] if live else max(ws.nodes)]
    name = FIELD_FOR_MODALITY[node.modality] if node.modality in ("asr", "ocr") else "ocr"
    anchor = _seg_index(video, min(ws.nodes.values(), key=lambda n: n.start).start)
    slot = _free_slot(out, anchor, name)
    if slot is not None:
        _set(out, slot, name, flip_text(node.text), edits)
    return out, edits


PERTURBATIONS: dict[str, PerturbFn] = {
    "audio_swap": audio_swap,
    "visual_swap": visual_swap,
    "temporal_shuffle": temporal_shuffle,
    "same_entity_distractor": same_entity_distractor,
    "pseudo_support": pseudo_support,
}


# --- Supervision records -------------------------------------------------------
def make_record(video: Video, question: Question, tr: Trajectory, ptype: str, donor: Video,
                llm: LLMClient | None = None) -> dict:
    """Apply one perturbation, re-run the controller, and build the supervision record."""
    new_video, edits = PERTURBATIONS[ptype](video, tr, donor)
    tr2 = run_prv(new_video, question, llm)
    before, after = decision_labels(tr), decision_labels(tr2)
    changed = {k: before[k] != after[k] for k in before}
    return {
        "record_id": f"{video.id}:{question.id}:{ptype}",
        "synthetic": True,
        "question": question.text,
        "perturbation": {"type": ptype, "edits": edits},
        "original_labels": before,
        "target_labels": after,
        "decision_changed": changed,
        "original_trace_reproduced": not (changed["tool_choice"] or changed["final"]),
    }


def generate_records(video: Video, question: Question, tr: Trajectory, donor: Video,
                     llm: LLMClient | None = None) -> list[dict]:
    """One supervision record per perturbation type, in the paper's order."""
    return [make_record(video, question, tr, p, donor, llm) for p in PERTURBATION_TYPES]


def to_jsonl(records: list[dict]) -> str:
    """Serialise records, one JSON object per line (sorted keys, so output is byte-stable)."""
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
