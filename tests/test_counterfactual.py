import json

import pytest

from prv.counterfactual import PERTURBATION_TYPES, PERTURBATIONS, generate_records, make_record, to_jsonl
from prv.loop import run_prv
from prv.text import opposite_polarity


@pytest.fixture()
def ctx(toy):
    videos, questions = toy
    q = questions["q_line"]
    video = videos[q.video_id]
    return video, q, run_prv(video, q), videos["v_cooking"]


def test_five_types_in_paper_order():
    assert PERTURBATION_TYPES == ("audio_swap", "visual_swap", "temporal_shuffle",
                                  "same_entity_distractor", "pseudo_support")
    assert set(PERTURBATIONS) == set(PERTURBATION_TYPES)


def test_audio_swap_replaces_speech_and_audio_only(ctx):
    video, _, tr, donor = ctx
    new, edits = PERTURBATIONS["audio_swap"](video, tr, donor)
    assert new.segments[2].asr == donor.segments[2].asr
    assert new.segments[2].audio_events == donor.segments[2].audio_events
    assert new.segments[2].visual == video.segments[2].visual
    assert {e["field"] for e in edits} == {"asr", "audio_events"}
    assert video.segments[2].asr.startswith("this production line")  # input is not mutated


def test_visual_swap_replaces_caption_only(ctx):
    video, _, tr, donor = ctx
    new, edits = PERTURBATIONS["visual_swap"](video, tr, donor)
    assert new.segments[2].visual == donor.segments[2].visual
    assert new.segments[2].asr == video.segments[2].asr
    assert [e["field"] for e in edits] == ["visual"]


def test_temporal_shuffle_keeps_timestamps_and_reverses_content(ctx):
    video, _, tr, donor = ctx
    new, edits = PERTURBATIONS["temporal_shuffle"](video, tr, donor)
    n = len(video.segments)
    for i in range(n):
        assert (new.segments[i].start, new.segments[i].end) == (video.segments[i].start, video.segments[i].end)
        assert new.segments[i].visual == video.segments[n - 1 - i].visual
        assert new.segments[i].ocr == video.segments[n - 1 - i].ocr
    assert edits[0]["permutation"] == list(range(n - 1, -1, -1))


def test_same_entity_distractor_is_ignored_as_neutral(ctx):
    video, q, tr, donor = ctx
    new, edits = PERTURBATIONS["same_entity_distractor"](video, tr, donor)
    assert len(edits) == 1 and edits[0]["before"] == ""
    assert "production line" in new.segments[edits[0]["segment"]].ocr
    rec = make_record(video, q, tr, "same_entity_distractor", donor)
    distractor_ref = f"ocr@{video.segments[edits[0]['segment']].start:g}-{video.segments[edits[0]['segment']].end:g}"
    pairs = [p for p in rec["target_labels"]["verification"]["pairs"] if distractor_ref in (p["a"], p["b"])]
    assert len(pairs) >= 3 and all(p["label"] == "neutral" for p in pairs)
    assert rec["decision_changed"]["final"] is False
    assert rec["original_trace_reproduced"] is True


def test_pseudo_support_inserts_opposite_claim_and_forces_abstention(ctx):
    video, q, tr, donor = ctx
    new, edits = PERTURBATIONS["pseudo_support"](video, tr, donor)
    fake = edits[0]["after"]
    assert opposite_polarity(fake, "Production line status RUNNING live feed")
    assert new.segments[edits[0]["segment"]].ocr == fake
    rec = make_record(video, q, tr, "pseudo_support", donor)
    assert rec["target_labels"]["final"]["abstain"] is True
    assert rec["target_labels"]["verification"]["unresolved"]
    assert rec["decision_changed"]["final"] and rec["original_trace_reproduced"] is False


def test_swaps_change_decisions_on_the_clean_trace(ctx):
    video, q, tr, donor = ctx
    a = make_record(video, q, tr, "audio_swap", donor)
    assert a["decision_changed"]["tool_choice"]  # the counter-retrieval step disappears
    assert not any(t["stage"] == "counter-retrieve" for t in a["target_labels"]["tool_choice"])
    v = make_record(video, q, tr, "visual_swap", donor)
    assert v["decision_changed"]["plan"] and v["target_labels"]["final"]["abstain"]
    t = make_record(video, q, tr, "temporal_shuffle", donor)
    assert t["target_labels"]["verification"]["temporal"][0]["relation"] == "after"
    assert video.segments[2].visual.startswith("the production line")  # clean video unchanged


def test_records_cover_all_types_and_roundtrip_jsonl(ctx):
    video, q, tr, donor = ctx
    recs = generate_records(video, q, tr, donor)
    assert [r["perturbation"]["type"] for r in recs] == list(PERTURBATION_TYPES)
    text = to_jsonl(recs)
    lines = text.splitlines()
    assert len(lines) == 5 and text == to_jsonl(generate_records(video, q, tr, donor))
    for line, rec in zip(lines, recs):
        back = json.loads(line)
        assert back == rec and back["synthetic"] is True
        assert set(back["target_labels"]) == {"plan", "tool_choice", "verification", "final"}
        assert set(back["decision_changed"]) == set(back["target_labels"])


@pytest.mark.parametrize("qid", ["q_oven", "q_line", "q_stall"])
@pytest.mark.parametrize("ptype", PERTURBATION_TYPES)
def test_every_type_runs_on_every_question(toy, qid, ptype):
    videos, questions = toy
    q = questions[qid]
    video = videos[q.video_id]
    donor = videos[sorted(k for k in videos if k != video.id)[0]]
    rec = make_record(video, q, run_prv(video, q), ptype, donor)
    json.dumps(rec)
    assert rec["perturbation"]["edits"]
