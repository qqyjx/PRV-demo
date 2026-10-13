"""Support-contradiction evidence workspace (paper Sec. 3.5).

A directed graph of evidence nodes with typed edges: support, contradict, temporal.
The workspace is the controller's external memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tools import Evidence


@dataclass
class Edge:
    a: int
    b: int
    kind: str  # "support" | "contradict" | "temporal"
    detail: str = ""  # temporal relation of a to b


@dataclass
class Workspace:
    nodes: dict[int, Evidence] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add(self, ev: Evidence) -> tuple[int, bool]:
        """Add an evidence span. Returns (node id, is_new); identical spans are stored once."""
        for nid, old in self.nodes.items():
            if (old.modality, old.start, old.end, old.text) == (ev.modality, ev.start, ev.end, ev.text):
                return nid, False
        nid = len(self.nodes) + 1
        self.nodes[nid] = ev
        return nid, True

    def add_edge(self, a: int, b: int, kind: str, detail: str = "") -> None:
        """Add an edge once. Support and contradict are mutually exclusive for a pair."""
        polarity_kinds = {"support", "contradict"}
        for e in self.edges:
            if {e.a, e.b} == {a, b} and (e.kind == kind or {e.kind, kind} <= polarity_kinds):
                return
        self.edges.append(Edge(a, b, kind, detail))

    def _linked(self, kind: str) -> list[Edge]:
        return [e for e in self.edges if e.kind == kind]

    def side(self, nid: int) -> set[int]:
        """A node plus its direct supporters."""
        out = {nid}
        for e in self._linked("support"):
            if e.a == nid:
                out.add(e.b)
            elif e.b == nid:
                out.add(e.a)
        return out

    def modality_count(self, ids: set[int]) -> int:
        return len({self.nodes[i].modality for i in ids})

    def refuted(self) -> set[int]:
        """Nodes outvoted in a contradiction: the other side has strictly more modalities."""
        out: set[int] = set()
        for e in self._linked("contradict"):
            sa, sb = self.side(e.a), self.side(e.b)
            ma, mb = self.modality_count(sa), self.modality_count(sb)
            if ma > mb:
                out.add(e.b)
            elif mb > ma:
                out.add(e.a)
        return out

    def unresolved_contradictions(self) -> list[Edge]:
        """Contradiction edges whose two endpoints are both still standing."""
        gone = self.refuted()
        return [e for e in self._linked("contradict") if e.a not in gone and e.b not in gone]

    def components(self) -> list[set[int]]:
        """Connected groups of standing nodes linked by support edges."""
        gone = self.refuted()
        live = [n for n in self.nodes if n not in gone]
        seen: set[int] = set()
        out = []
        for n in live:
            if n in seen:
                continue
            comp, stack = set(), [n]
            while stack:
                x = stack.pop()
                if x in comp:
                    continue
                comp.add(x)
                for e in self._linked("support"):
                    if x in (e.a, e.b):
                        y = e.b if e.a == x else e.a
                        if y not in gone:
                            stack.append(y)
            seen |= comp
            out.append(comp)
        return out

    def ref(self, nid: int) -> str:
        return self.nodes[nid].ref
