from __future__ import annotations

from dataclasses import dataclass

from typing import Callable, TYPE_CHECKING, Optional, Any
import os
import shutil

from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.optimize.optimize import Optimizer


@dataclass
class PipelineContext:
    reactant_atoms: dict[str, Atoms]
    reactants: dict[str, "Reactant"]
    bond_formation: "BondFormationSettings | None"
    calculator_factories: dict[
        str, Callable[[Optional[dict[str, Any]]], Optional[BaseCalculator]]
    ]
    optimizer_factory: type[Optimizer]
    _pool: "CandidatePool"
    workdir: str
    filter_registry: dict[str, Callable[..., bool]]

    @property
    def pool(self) -> "CandidatePool":
        return self._pool

    @pool.setter
    def pool(self, value: "CandidatePool") -> None:
        self._pool = value

    def mount(self) -> None:
        if os.path.exists(self.workdir):
            shutil.rmtree(self.workdir)
        os.makedirs(self.workdir, exist_ok=True)

    def stage_dir(self, stage_id: str) -> str:
        path = os.path.join(self.workdir, stage_id)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        return path

    def stage_step_dir(self, stage_id: str, step_id: str) -> str:
        path = os.path.join(self.stage_dir(stage_id), step_id)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        return path

    def split_pair_token(self, token: str) -> tuple[str, int]:
        if ":" not in token:
            raise ValueError(f"Move pair value '{token}' must be in 'id:idx' form.")
        reactant_id, idx_raw = token.split(":", 1)
        try:
            atom_idx = int(idx_raw)
        except ValueError as exc:
            raise ValueError(
                f"Move pair index '{idx_raw}' must be an integer."
            ) from exc
        return reactant_id, atom_idx

    def resolve_offsets(self, a_id: str, b_id: str) -> tuple[int, int, int]:
        if self.bond_formation is None:
            raise ValueError("bond_formation settings are required for move stages.")
        if a_id == b_id:
            for first_id, second_id in self.bond_formation.atom_pairs:
                if a_id not in (first_id, second_id):
                    continue
                first_atoms = self.reactant_atoms.get(first_id)
                second_atoms = self.reactant_atoms.get(second_id)
                if first_atoms is None or second_atoms is None:
                    raise ValueError("Reactant atoms missing for move stage offsets.")
                first_len = len(first_atoms)
                offset = 0 if a_id == first_id else first_len
                return offset, offset, offset
            raise ValueError(f"Move pair references unknown reactant id: {a_id}.")
        for first_id, second_id in self.bond_formation.atom_pairs:
            if {first_id, second_id} != {a_id, b_id}:
                continue
            first_atoms = self.reactant_atoms.get(first_id)
            second_atoms = self.reactant_atoms.get(second_id)
            if first_atoms is None or second_atoms is None:
                raise ValueError("Reactant atoms missing for move stage offsets.")
            first_len = len(first_atoms)
            if a_id == first_id:
                a_offset = 0
                b_offset = first_len
            else:
                a_offset = first_len
                b_offset = 0
            move_start_idx = b_offset if b_offset != 0 else a_offset
            return a_offset, b_offset, move_start_idx
        raise ValueError(f"Move pair references unknown reactant ids: {a_id}, {b_id}.")


if TYPE_CHECKING:
    from runner.candidate_pool import CandidatePool
    from runner.reaction_parser import BondFormationSettings, Reactant
