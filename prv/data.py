"""Synthetic long-video representation: a video is a list of timed JSON segments."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from importlib import resources


@dataclass
class Segment:
    """One time slice with the four per-modality fields the tools read."""

    start: float
    end: float
    visual: str = ""
    asr: str = ""
    audio_events: list[str] = field(default_factory=list)
    ocr: str = ""


@dataclass
class Video:
    """A synthetic video: contiguous segments plus an id."""

    id: str
    title: str
    segments: list[Segment]
    synthetic: bool = True

    @property
    def duration(self) -> float:
        return self.segments[-1].end

    def clone(self) -> "Video":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "synthetic": self.synthetic,
                "segments": [vars(s).copy() for s in self.segments]}


@dataclass
class Question:
    id: str
    video_id: str
    text: str


def load_toy_data() -> tuple[dict[str, Video], dict[str, Question]]:
    """Load the bundled synthetic videos and questions."""
    raw = json.loads(resources.files("prv").joinpath("data/toy_videos.json").read_text(encoding="utf-8"))
    videos = {v["id"]: Video(v["id"], v["title"], [Segment(**s) for s in v["segments"]], v.get("synthetic", True))
              for v in raw["videos"]}
    questions = {q["id"]: Question(**q) for q in raw["questions"]}
    return videos, questions


def summarize(video: Video) -> list[tuple[float, float, str]]:
    """Coarse video summary: one caption plus audio tags per segment (paper Sec. 3.2)."""
    out = []
    for s in video.segments:
        tags = f" [audio: {', '.join(s.audio_events)}]" if s.audio_events else ""
        out.append((s.start, s.end, s.visual + tags))
    return out
