"""The six tools of the PRV tool set (paper Sec. 3.3), implemented over the toy JSON video.

Each tool is atomic and stateless and returns structured data. The retrievers use
lexical overlap instead of InternVideo2 / BEATs embeddings; ASR and OCR read the
stored fields instead of running Whisper / an OCR model; the contradiction verifier
is a rule over shared words and polar word pairs instead of a learned bilinear scorer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Video
from .text import content_tokens, numbers, opposite_polarity, same_polarity

MODALITIES = ("vis", "aud", "asr", "ocr")
TOOL_FOR_MODALITY = {
    "vis": "visual_retriever",
    "aud": "audio_retriever",
    "asr": "asr_parser",
    "ocr": "ocr_extractor",
}
MODALITY_FOR_TOOL = {v: k for k, v in TOOL_FOR_MODALITY.items()}
ALL_TOOLS = (*TOOL_FOR_MODALITY.values(), "temporal_checker", "contradiction_verifier")

Region = tuple[float, float] | None


@dataclass(frozen=True)
class Evidence:
    """One evidence span with modality attribution and a textual summary."""

    modality: str
    start: float
    end: float
    text: str
    score: float = 1.0

    @property
    def ref(self) -> str:
        return f"{self.modality}@{self.start:g}-{self.end:g}"

    def to_dict(self) -> dict:
        return {"ref": self.ref, "modality": self.modality, "span": [self.start, self.end],
                "text": self.text, "score": round(self.score, 3)}


@dataclass(frozen=True)
class PolarityResult:
    """Output of the contradiction verifier for a pair of evidence spans."""

    label: str  # "support" | "contradict" | "neutral"
    shared: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalResult:
    """Output of the temporal checker."""

    relation: str  # "before" | "after" | "overlap" (of a relative to b)
    holds: bool = True
    requested: str | None = None


def _in_region(start: float, end: float, region: Region) -> bool:
    return region is None or (start < region[1] and end > region[0])


def _rank(query: str, items: list[tuple[float, float, str]], modality: str, k: int) -> list[Evidence]:
    q = set(content_tokens(query))
    need = min(2, len(q))
    scored = []
    for start, end, text in items:
        if not text:
            continue
        shared = q & set(content_tokens(text))
        if len(shared) >= need and shared:
            scored.append(Evidence(modality, start, end, text, len(shared) / len(q)))
    scored.sort(key=lambda e: (-e.score, e.start))
    return scored[:k]


def visual_retriever(video: Video, query: str, region: Region = None, k: int = 3) -> list[Evidence]:
    """T_vis: top-k visual moments whose caption matches the query."""
    items = [(s.start, s.end, s.visual) for s in video.segments if _in_region(s.start, s.end, region)]
    return _rank(query, items, "vis", k)


def audio_retriever(video: Video, query: str, region: Region = None, k: int = 3) -> list[Evidence]:
    """T_aud: top-k audio events whose label matches the query."""
    items = [(s.start, s.end, ev) for s in video.segments if _in_region(s.start, s.end, region)
             for ev in s.audio_events]
    return _rank(query, items, "aud", k)


def asr_parser(video: Video, region: Region = None, query: str | None = None) -> list[Evidence]:
    """T_asr: speech transcript lines inside the region; with a query, only lines that match it."""
    items = [(s.start, s.end, s.asr) for s in video.segments if _in_region(s.start, s.end, region)]
    return _rank(query, items, "asr", len(items)) if query else _all(items, "asr")


def ocr_extractor(video: Video, region: Region = None, query: str | None = None) -> list[Evidence]:
    """T_ocr: on-screen text inside the region; with a query, only lines that match it."""
    items = [(s.start, s.end, s.ocr) for s in video.segments if _in_region(s.start, s.end, region)]
    return _rank(query, items, "ocr", len(items)) if query else _all(items, "ocr")


def _all(items: list[tuple[float, float, str]], modality: str) -> list[Evidence]:
    return [Evidence(modality, s, e, t) for s, e, t in items if t]


def temporal_checker(a: Evidence, b: Evidence, relation: str | None = None) -> TemporalResult:
    """T_temp: before/after/overlap relation of span a to span b; optionally test a requested relation."""
    if a.end <= b.start:
        actual = "before"
    elif b.end <= a.start:
        actual = "after"
    else:
        actual = "overlap"
    return TemporalResult(actual, relation is None or relation == actual, relation)


def contradiction_verifier(a: Evidence, b: Evidence) -> PolarityResult:
    """T_ver: classify a pair of spans as support, contradict or neutral.

    Rule: the spans must share at least two non-numeric content words (same topic).
    They contradict when they take opposite sides of a polar word pair or give
    different numbers. They support each other when they share three words, or two
    words plus the same polar side or the same number. Anything else is neutral.
    """
    ta, tb = set(content_tokens(a.text)), set(content_tokens(b.text))
    shared = ta & tb
    topic = {t for t in shared if not t.isdigit()}
    na, nb = numbers(a.text), numbers(b.text)
    key = tuple(sorted(shared))
    if len(topic) >= 2:
        if opposite_polarity(a.text, b.text) or (na and nb and na.isdisjoint(nb)):
            return PolarityResult("contradict", key)
        if len(shared) >= 3 or same_polarity(a.text, b.text) or (na & nb):
            return PolarityResult("support", key)
    return PolarityResult("neutral", key)
