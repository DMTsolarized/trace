from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

from ase import Atoms

from utils.reactive_site import ReactiveSite


@dataclass(frozen=True)
class Candidate:
    atoms: Atoms
    spin: int = 0
    reactive_center: ReactiveSite | None = None


class CandidatePool:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def filter(self, fn: Callable[[Candidate], bool]) -> "CandidatePool":
        return CandidatePool([c for c in self.candidates if fn(c)])

    def map(self, fn: Callable[[Candidate], Candidate]) -> "CandidatePool":
        return CandidatePool([fn(c) for c in self.candidates])

    @property
    def atoms(self) -> list[Atoms]:
        return [candidate.atoms for candidate in self.candidates]

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self.candidates)
