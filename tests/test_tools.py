from prv import tools
from prv.tools import Evidence


def test_visual_retriever_matches_caption_and_respects_region(toy):
    videos, _ = toy
    v = videos["v_factory"]
    hits = tools.visual_retriever(v, "production line running")
    assert [e.ref for e in hits] == ["vis@20-30"]
    assert tools.visual_retriever(v, "production line running", region=(60, 80)) == []
    assert tools.visual_retriever(v, "dinosaur parade") == []


def test_audio_retriever_matches_event_label(toy):
    videos, _ = toy
    hits = tools.audio_retriever(videos["v_cooking"], "oven door sound")
    assert [(e.modality, e.text) for e in hits] == [("aud", "oven door")]
    assert tools.audio_retriever(videos["v_cooking"], "oven door sound", region=(0, 20)) == []


def test_asr_parser_region_and_query(toy):
    videos, _ = toy
    v = videos["v_cooking"]
    assert [e.text for e in tools.asr_parser(v, (30, 40))] == ["now set the oven to 180 degrees"]
    assert len(tools.asr_parser(v)) == 8
    assert [e.ref for e in tools.asr_parser(v, None, "shape loaves gently")] == ["asr@50-60"]
    assert tools.asr_parser(v, None, "dinosaur parade") == []


def test_ocr_extractor_skips_empty_fields(toy):
    videos, _ = toy
    v = videos["v_factory"]
    assert [e.text for e in tools.ocr_extractor(v, (50, 60))] == ["Production line status RUNNING live feed"]
    assert all(e.text for e in tools.ocr_extractor(v))
    assert tools.ocr_extractor(v, (10, 20)) == []


def test_temporal_checker_relations():
    a, b = Evidence("vis", 0, 10, "x"), Evidence("asr", 20, 30, "y")
    assert tools.temporal_checker(a, b).relation == "before"
    assert tools.temporal_checker(b, a).relation == "after"
    assert tools.temporal_checker(a, Evidence("ocr", 5, 15, "z")).relation == "overlap"
    assert tools.temporal_checker(a, b, "before").holds
    assert not tools.temporal_checker(a, b, "after").holds


def test_contradiction_verifier_labels():
    ev = lambda t: Evidence("asr", 0, 10, t)  # noqa: E731
    run = ev("the production line is running")
    assert tools.contradiction_verifier(run, ev("production line status RUNNING live")).label == "support"
    assert tools.contradiction_verifier(run, ev("this production line has been shut down")).label == "contradict"
    assert tools.contradiction_verifier(ev("oven set to 180 degrees"), ev("oven set to 220 degrees")).label == "contradict"
    assert tools.contradiction_verifier(run, ev("production line mentioned in an unrelated notice")).label == "neutral"
    assert tools.contradiction_verifier(run, ev("gift shop souvenirs")).label == "neutral"
