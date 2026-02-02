from __future__ import annotations

import numpy as np
import os
from ase.io import write

from runner.reaction_parser import MoveSettings, MoveTarget, MoveOffset
from runner.candidate_pool import Candidate, CandidatePool
from runner.stage_runners.base_stage_runner import StageRunner
from utils.generate_sample_distances import (
    get_covalent_bond_length,
    set_pair_distance_split,
    walk,
)
from utils.harmonic_bond import HarmonicBond
from utils.reverse_hookean import ReverseHookean
from ase import Atoms


class MoveStageRunner(StageRunner):
    def run(self) -> None:
        calculator_factory = self.context.calculator_factories.get(self.stage.id)
        if calculator_factory is None or not self.stage.moves:
            return
        self.context.pool = self._apply_moves(
            self.context.pool, calculator_factory, self.stage.moves
        )

    def _apply_moves(
        self,
        pool: CandidatePool,
        calculator_factory,
        moves: list[MoveSettings],
    ) -> CandidatePool:
        step_dir = self.context.stage_step_dir(self.stage.id, "step")
        new_candidates: list[Candidate] = []
        for candidate_idx, candidate in enumerate(list(pool)):
            current_atoms = candidate.atoms
            move_specs = _prepare_move_specs(self.context, moves, current_atoms)
            max_len = max(len(spec.path) for spec in move_specs)
            _normalize_paths(move_specs, max_len)
            for step_idx in range(max_len):
                worker = current_atoms.copy()
                constraints: list = []
                pair_candidates: list[tuple[float, _MoveSpec, float]] = []
                for spec in move_specs:
                    if spec.kind != "pair_pull":
                        continue
                    d = spec.path[step_idx]
                    dist = float(
                        np.linalg.norm(
                            worker[spec.i].position - worker[spec.j].position
                        )
                    )
                    pair_candidates.append((abs(d - dist), spec, d))
                if pair_candidates:
                    _, chosen_spec, chosen_d = min(
                        pair_candidates, key=lambda item: item[0]
                    )
                    set_pair_distance_split(
                        worker,
                        chosen_spec.i,
                        chosen_spec.j,
                        chosen_d,
                        chosen_spec.move_start_idx,
                    )
                for spec in move_specs:
                    d = spec.path[step_idx]
                    if spec.kind == "pair_pull":
                        constraints.append(HarmonicBond(spec.i, spec.j, d, spec.k))
                    else:
                        _apply_tether(worker, spec.i, spec.j, d, spec.k, constraints)
                worker.set_constraint(constraints)
                calculator = calculator_factory(None)
                if calculator is None:
                    return pool
                worker.calc = calculator
                opt = self.context.optimizer_factory(worker)
                try:
                    opt.run(fmax=self.stage.fmax, steps=self.stage.steps)
                except Exception:
                    continue
                worker.set_constraint()
                filename = f"cand_{candidate_idx}_step_{step_idx}.xyz"
                write_path = os.path.join(step_dir, filename)
                write(write_path, worker)
                current_atoms = worker
            new_candidates.append(
                Candidate(
                    atoms=current_atoms,
                    spin=candidate.spin,
                    reactive_center=candidate.reactive_center,
                )
            )
            # TODO: add proper way to replace candidates (filters maybe or similar), for now we just replace with last point (but we might not want to)
        return CandidatePool(new_candidates)


def _resolve_pair_pull(
    target: MoveTarget, atoms: Atoms, i: int, j: int
) -> tuple[float, float, float]:
    current = float(np.linalg.norm(atoms[i].position - atoms[j].position))
    bond_target = get_covalent_bond_length((atoms[i].number, atoms[j].number))
    start = current if target.start == "auto" else float(target.start)
    end = bond_target if target.end == "auto" else float(target.end)
    step = 0.1 if target.step == "auto" else float(target.step)
    return start, end, step


def _resolve_tether_pull(
    offset: MoveOffset, atoms: Atoms, i: int, j: int
) -> tuple[float, float, float]:
    current = float(np.linalg.norm(atoms[i].position - atoms[j].position))
    start = current if offset.start == "auto" else float(offset.start)
    end = float(offset.end)
    step = float(offset.step)
    return start, end, step


class _MoveSpec:
    def __init__(
        self,
        i: int,
        j: int,
        move_start_idx: int,
        k: int,
        path: list[float],
        kind: str,
    ) -> None:
        self.i = i
        self.j = j
        self.move_start_idx = move_start_idx
        self.k = k
        self.path = path
        self.kind = kind


def _prepare_move_specs(
    context, moves: list[MoveSettings], atoms: Atoms
) -> list[_MoveSpec]:
    specs: list[_MoveSpec] = []
    for move in moves:
        a_id, a_idx = context.split_pair_token(move.pair.a)
        b_id, b_idx = context.split_pair_token(move.pair.b)
        a_offset, b_offset, move_start_idx = context.resolve_offsets(a_id, b_id)
        i = a_offset + a_idx
        j = b_offset + b_idx
        if move.kind == "pair_pull":
            if not move.target:
                raise ValueError("Pair Pull requires target property")
            start, target, step = _resolve_pair_pull(move.target, atoms, i, j)
        else:
            if not move.offset:
                raise ValueError("Thethering requires atom offset")
            start, target, step = _resolve_tether_pull(move.offset, atoms, i, j)
        path = list(walk(start, target, step))
        specs.append(
            _MoveSpec(
                i=i,
                j=j,
                move_start_idx=move_start_idx,
                k=move.k,
                path=path,
                kind=move.kind,
            )
        )
    return specs


def _normalize_paths(specs: list[_MoveSpec], target_len: int) -> None:
    for spec in specs:
        if not spec.path:
            spec.path = [0.0] * target_len
            continue
        if len(spec.path) < target_len:
            spec.path.extend([spec.path[-1]] * (target_len - len(spec.path)))


def _apply_tether(
    atoms: Atoms,
    i: int,
    j: int,
    offset: float,
    k: int,
    constraints: list,
) -> None:
    if abs(offset) < 1e-12:
        return
    constraints.append(ReverseHookean(i, j, abs(offset), k))
