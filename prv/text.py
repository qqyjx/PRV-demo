"""Tiny text utilities shared by the tools and the mock controller.

Everything here is lexical and deterministic. It stands in for the learned
encoders and the learned polarity scorer used by the real system.
"""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    "a an the is are was were be been being do does did what which who whom when where how why "
    "of in on at to for from by with and or not no during this that these those it its has have "
    "had there their they he she his her you your we our as if then than so such can will would "
    "should could may might into over after before about up down since just also very more most "
    "any some all each other only own same too now".split()
)

# Each pair is (positive words, negative words). Two texts that use opposite
# sides of the same pair express opposite states of the same thing.
POLAR_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"running", "operating", "active", "moving"}), frozenset({"shut", "stopped", "idle", "halted"})),
    (frozenset({"open"}), frozenset({"closed", "shuttered"})),
    (frozenset({"crowded", "busy"}), frozenset({"empty", "deserted"})),
)
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    """Lowercased alphanumeric words."""
    return _WORD.findall(text.lower())


def stem(word: str) -> str:
    """Strip a plural 's' (very light normalisation)."""
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def content_tokens(text: str) -> list[str]:
    """Stemmed words without stopwords, in order of first appearance."""
    seen: list[str] = []
    for w in words(text):
        if w in STOPWORDS:
            continue
        s = stem(w)
        if s not in seen:
            seen.append(s)
    return seen


def numbers(text: str) -> set[str]:
    """Numeric words in the text."""
    return {w for w in words(text) if w.isdigit()}


def polarity(text: str) -> dict[int, int]:
    """Map each polar-pair index to +1 or -1 when the text uses exactly one side."""
    ws = set(words(text))
    out: dict[int, int] = {}
    for i, (pos, neg) in enumerate(POLAR_PAIRS):
        has_pos, has_neg = bool(ws & pos), bool(ws & neg)
        if has_pos != has_neg:
            out[i] = 1 if has_pos else -1
    return out


def opposite_polarity(a: str, b: str) -> bool:
    """True when the two texts take opposite sides of some polar pair."""
    pa, pb = polarity(a), polarity(b)
    return any(pa[i] == -pb[i] for i in pa.keys() & pb.keys())


def same_polarity(a: str, b: str) -> bool:
    """True when the two texts take the same side of some polar pair."""
    pa, pb = polarity(a), polarity(b)
    return any(pa[i] == pb[i] for i in pa.keys() & pb.keys())


def flip_text(text: str) -> str:
    """Return a copy of the text that states the opposite (polar word swap or number change)."""
    out, changed = [], False
    for tok in re.findall(r"\s+|[A-Za-z0-9]+|[^A-Za-z0-9\s]+", text):
        low = tok.lower()
        repl = None
        for pos, neg in POLAR_PAIRS:
            if low in pos and not changed:
                repl = min(neg)
            elif low in neg and not changed:
                repl = min(pos)
            if repl:
                break
        if repl:
            tok = repl.upper() if tok.isupper() else repl
            changed = True
        out.append(tok)
    if changed:
        return "".join(out)
    return re.sub(r"\d+", lambda m: str(int(m.group()) + 40), text, count=1)
