from prv.data import Question
from prv.llm import STOP, MockController, ToolCall
from prv.loop import format_trajectory, run_prv


def run(toy, qid, **kw):
    videos, questions = toy
    q = questions[qid]
    return run_prv(videos[q.video_id], q, **kw)


def test_plan_has_all_six_tools_region_and_counter_flag(toy):
    tr = run(toy, "q_line")
    assert set(tr.plan.priority) == {"visual_retriever", "audio_retriever", "asr_parser", "ocr_extractor",
                                     "temporal_checker", "contradiction_verifier"}
    assert tr.plan.region == (20, 30)
    assert tr.plan.need_counter and tr.plan.h0 == "yes"
    assert not run(toy, "q_oven").plan.need_counter


def test_verified_answer_uses_two_modalities(toy):
    tr = run(toy, "q_oven")
    assert not tr.answer.abstain
    assert "180" in tr.answer.text
    assert {e.modality for e in tr.answer.evidence} == {"asr", "ocr"}
    assert all(s.stage == "retrieve" for s in tr.steps)
    assert tr.steps[-1].call.tool == "temporal_checker"


def test_contradiction_triggers_counter_retrieval_and_is_resolved(toy):
    tr = run(toy, "q_line")
    stages = [s.stage for s in tr.steps]
    assert "counter-retrieve" in stages
    assert stages.index("counter-retrieve") == 2  # right after the contradicting ASR line is found
    counter = tr.steps[2]
    assert counter.call.args["region"] is None  # widened to the whole video
    assert tr.workspace.refuted() == {2}  # the ASR claim is outvoted by vision + OCR
    assert not tr.workspace.unresolved_contradictions()
    assert tr.answer.text == "yes" and not tr.answer.abstain


def test_abstains_on_unresolved_contradiction(toy):
    tr = run(toy, "q_stall")
    assert tr.answer.abstain and tr.answer.text is None
    assert "contradict" in tr.answer.reason
    assert tr.answer.confidence < 0.3
    assert len(tr.workspace.unresolved_contradictions()) == 1
    assert any(s.stage == "counter-retrieve" for s in tr.steps)


def test_abstains_when_no_evidence(toy):
    videos, _ = toy
    tr = run_prv(videos["v_cooking"], Question("q_x", "v_cooking", "Is the dinosaur running?"))
    assert tr.answer.abstain and tr.answer.reason == "no evidence found"
    assert tr.answer.confidence == 0


def test_step_budget_is_respected(toy):
    tr = run(toy, "q_oven", max_steps=2)
    assert len(tr.steps) == 2
    assert tr.stop_reason == "step budget reached"


def test_custom_llm_client_plugs_in(toy):
    class StopNow(MockController):
        def act(self, *args, **kwargs):
            return ToolCall(STOP)

    tr = run(toy, "q_oven", llm=StopNow())
    assert tr.steps == [] and tr.answer.abstain


def test_run_is_deterministic(toy):
    assert format_trajectory(run(toy, "q_line")) == format_trajectory(run(toy, "q_line"))
