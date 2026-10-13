"""End-to-end demo: `python -m prv.demo [--out supervision.jsonl]`.

Runs three questions on synthetic videos (verified answer, verified after counter-retrieval,
abstention), then builds the five counterfactual supervision records for one trajectory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .counterfactual import generate_records, to_jsonl
from .data import load_toy_data
from .loop import format_trajectory, run_prv

CASES = (
    ("Case 1: answer verified by two modalities", "q_oven"),
    ("Case 2: contradiction, counter-retrieval, then a verified answer", "q_line"),
    ("Case 3: contradiction that counter-retrieval cannot resolve, so the controller abstains", "q_stall"),
)
SUPERVISED_CASE = "q_line"


def _final(lab: dict) -> str:
    f = lab["final"]
    return "ABSTAIN" if f["abstain"] else repr(f["answer"])


def _tools(lab: dict) -> str:
    out = []
    for t in lab["tool_choice"]:
        name = t["tool"].replace("_retriever", "").replace("_parser", "").replace("_extractor", "")
        name = name.replace("temporal_checker", "temporal")
        reg = t["region"]
        where = "" if t["tool"] == "temporal_checker" else f"[{reg[0]:g}-{reg[1]:g}]" if reg else "[all]"
        out.append(("counter:" if t["stage"] == "counter-retrieve" else "") + name + where)
    return " > ".join(out)


def _plan(lab: dict) -> str:
    p = lab["plan"]
    reg = f"{p['region'][0]:g}-{p['region'][1]:g}s" if p["region"] else "whole video"
    return f"{reg}, modalities {'/'.join(p['needed_modalities'])}"


def _verif(lab: dict) -> str:
    v = lab["verification"]
    counts: dict[str, int] = {}
    for p in v["pairs"]:
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    rel = ",".join(t["relation"] for t in v["temporal"]) or "-"
    return (f"pairs {dict(sorted(counts.items()))}, outvoted {v['outvoted']}, "
            f"unresolved {v['unresolved']}, temporal {rel}")


def print_record(i: int, rec: dict) -> None:
    """Readable view of one supervision record (the JSONL line holds the full labels)."""
    p = rec["perturbation"]
    print(f"--- record {i}/5  {rec['record_id']}")
    for e in p["edits"]:
        if "permutation" in e:
            print(f"  edit: reverse segment order {e['permutation']}")
        else:
            print(f"  edit: segment {e['segment']} {e['field']}: {e['before']!r} -> {e['after']!r}")
    o, t, ch = rec["original_labels"], rec["target_labels"], rec["decision_changed"]
    for key, fn in (("plan", _plan), ("tool_choice", _tools), ("verification", _verif), ("final", _final)):
        mark = "CHANGED" if ch[key] else "same"
        print(f"  {key:<12} [{mark}]")
        print(f"      clean:     {fn(o)}")
        if ch[key]:
            print(f"      perturbed: {fn(t)}")
    print(f"  original trace reproduced on perturbed input: {rec['original_trace_reproduced']}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="PRV minimal demo (synthetic data, mock controller)")
    ap.add_argument("--out", type=Path, help="write the supervision records to this JSONL file")
    args = ap.parse_args(argv)

    videos, questions = load_toy_data()
    print("PRV demo: SYNTHETIC toy videos, rule-based mock controller. Not the paper's code; no paper numbers.")
    trajectories = {}
    for title, qid in CASES:
        q = questions[qid]
        tr = run_prv(videos[q.video_id], q)
        trajectories[qid] = tr
        print(f"\n=== {title}")
        print(format_trajectory(tr))

    q = questions[SUPERVISED_CASE]
    video = videos[q.video_id]
    donor = videos[sorted(k for k in videos if k != video.id)[0]]
    print(f"\n=== Counterfactual supervision records from the trajectory of '{q.text}'")
    print(f"(perturbations applied to {video.id}; donor video for swaps: {donor.id})")
    records = generate_records(video, q, trajectories[SUPERVISED_CASE], donor)
    for i, rec in enumerate(records, 1):
        print_record(i, rec)
    if args.out:
        args.out.write_text(to_jsonl(records), encoding="utf-8")
        print(f"\nwrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
