#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np

from ase.io import read, write
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate
from ase.neighborlist import neighbor_list, natural_cutoffs

from scipy.optimize import linear_sum_assignment
from utils.dtxb_calculator import DXTBCalculator


# ============================================================
# 1) Hungarian atom MAPPING between reactant & product
# ============================================================


def build_atom_mapping(R_pos, P_pos, R_symbols, P_symbols):
    R_pos = np.asarray(R_pos)
    P_pos = np.asarray(P_pos)
    n = len(R_pos)

    big = 1e6
    C = np.zeros((n, n))

    # Precompute neighbor signatures to discourage swapping atoms with different environments
    def neighbor_signature(pos, symbols):
        atoms_tmp_pos = pos
        # build quick neighbor signatures using distances and covalent cutoffs
        from ase import Atoms

        atoms_tmp = Atoms(symbols=symbols, positions=atoms_tmp_pos)
        cutoffs = natural_cutoffs(atoms_tmp, mult=1.1)
        i_list, j_list = neighbor_list("ij", atoms_tmp, cutoff=cutoffs)
        sig = [list() for _ in range(n)]
        for i, j in zip(i_list, j_list):
            sig[i].append(symbols[j])
        return [tuple(sorted(s)) for s in sig]

    R_sig = neighbor_signature(R_pos, R_symbols)
    P_sig = neighbor_signature(P_pos, P_symbols)

    for i in range(n):
        for j in range(n):
            if R_symbols[i] != P_symbols[j]:
                C[i, j] = big
            elif R_sig[i] != P_sig[j]:
                C[i, j] = big / 10.0  # discourage mapping to different environments
            else:
                C[i, j] = np.linalg.norm(R_pos[i] - P_pos[j])

            # small bias to keep mapping near the diagonal (prefer original ordering)
            C[i, j] += 0.05 * abs(i - j)

    _, col_ind = linear_sum_assignment(C)
    return col_ind  # mapping: R[i] corresponds to P[col_ind[i]]


def reorder_positions(P_pos, mapping):
    P_pos = np.asarray(P_pos)
    return P_pos[mapping]


def kabsch_align(reference, mobile):
    """Center both sets, compute optimal rotation, and return aligned mobile."""
    reference = np.asarray(reference)
    mobile = np.asarray(mobile)
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    ref_c = reference - ref_center
    mob_c = mobile - mob_center
    H = mob_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    aligned = mob_c @ R + ref_center
    return aligned


# ============================================================
# 2) BOND inference (ASE NeighborList)
# ============================================================


def infer_bonds(atoms, mult=1.1):
    """
    Build a bond set using ASE neighbor_list with covalent radii cutoffs.
    mult scales the covalent radii (e.g., 1.1 = 10% longer than covalent sum).
    """
    cutoffs = natural_cutoffs(atoms, mult=mult)
    i_list, j_list = neighbor_list("ij", atoms, cutoff=cutoffs)
    bonds = set()
    for i, j in zip(i_list, j_list):
        if i == j:
            continue
        bonds.add((min(i, j), max(i, j)))
    return bonds


# ============================================================
# 3) Automatic reactive atom detection
# ============================================================


def detect_reactive_atoms(
    reactant_atoms,
    product_atoms,
    rmsd_thresh=0.35,
    bond_mult=1.1,
    use_rmsd: bool = True,
):
    """
    Uses BOTH:
      - RMSD per atom
      - Bond-change detection via ASE NeighborList
    Union gives reactive atom set.
    """
    R_pos = reactant_atoms.get_positions()
    P_pos = product_atoms.get_positions()

    rmsd_atoms = set()
    if use_rmsd:
        diff = np.linalg.norm(R_pos - P_pos, axis=1)
        rmsd_atoms = set(np.where(diff > rmsd_thresh)[0])

    # --- Bond changes ---
    bonds_R = infer_bonds(reactant_atoms, mult=bond_mult)
    bonds_P = infer_bonds(product_atoms, mult=bond_mult)
    changed = bonds_R.symmetric_difference(bonds_P)
    bond_atoms = set()
    for i, j in changed:
        bond_atoms.add(i)
        bond_atoms.add(j)

    return sorted(rmsd_atoms.union(bond_atoms))


# ============================================================
# 4) Localized displacement vector (only reactive atoms move)
# ============================================================


def build_local_displacement_vector(R_pos, P_pos, reactive_atoms, step):
    """
    Displacement only on reactive atoms.
    Others get zero displacement.
    """
    disp = np.zeros_like(R_pos)

    local_diff = P_pos - R_pos
    disp[reactive_atoms] = local_diff[reactive_atoms]

    norm = np.linalg.norm(disp)
    if norm < 1e-8:
        raise ValueError("Reactive displacement vector has zero norm.")

    disp = disp / norm * step
    return disp


# ============================================================
# 5) MAIN Dimer Search Workflow
# ============================================================


def run_dimer_search(
    reactant_path: Path | str = "dontdelete/test_out.xyz",
    product_path: Path | str = "ase_ox.xyz",
    method: str = "GFN1",
    displacement_step: float = 0.5,
    fmax: float = 0.08,
    max_steps: int | None = None,
    traj_path: Path | str = "dimer.traj",
    logfile: Path | str | None = "dimer.log",
    output_xyz: Path | str = "saddle_point.xyz",
) -> None:

    # --- Load structures ---
    R = read(reactant_path)
    P = read(product_path)

    if len(R) != len(P):
        raise ValueError("Atom count mismatch between reactant and product")

    R_symbols = R.get_chemical_symbols()
    P_symbols = P.get_chemical_symbols()
    R_pos = R.get_positions()

    # ============================================================
    # Step 1 — Atom mapping via Hungarian algorithm
    # ============================================================

    mapping = build_atom_mapping(R_pos, P.get_positions(), R_symbols, P_symbols)
    # reorder product atoms to match reactant indexing
    P_reordered = P[mapping]
    P_mapped_pos = P_reordered.get_positions()

    # ============================================================
    # Step 2 — Kabsch alignment (mapped product → reactant)
    # ============================================================

    P_aligned = kabsch_align(R_pos, P_mapped_pos)
    P_reordered.set_positions(P_aligned)

    # Also compute an alignment without permutation (fallback if mapped detects nothing)
    P_no_perm = P.copy()
    P_no_perm_aligned = kabsch_align(R_pos, P_no_perm.get_positions())
    P_no_perm.set_positions(P_no_perm_aligned)

    # ============================================================
    # Step 3 — Detect reactive atoms
    # ============================================================

    reactive_mapped = detect_reactive_atoms(R, P_reordered)
    reactive_fallback = detect_reactive_atoms(R, P_no_perm, use_rmsd=False)

    if reactive_mapped and (
        not reactive_fallback or len(reactive_mapped) <= len(reactive_fallback)
    ):
        reactive_atoms = reactive_mapped
    elif reactive_fallback:
        reactive_atoms = reactive_fallback
    else:
        raise RuntimeError("No reactive atoms detected.")

    print(f"[INFO] Reactive atoms detected: {reactive_atoms}")
    # For testing: force reactive mask to first atom plus last 12 atoms
    n = len(R)
    reactive_atoms = [0] + list(range(max(0, n - 12), n))
    # ============================================================
    # Step 4 — Build localized displacement vector
    # ============================================================

    disp_vec = build_local_displacement_vector(
        R_pos,
        P_aligned,
        reactive_atoms,
        displacement_step,
    )

    # ============================================================
    # Step 5 — Setup calculator
    # ============================================================

    R.calc = DXTBCalculator(method=method)

    # ============================================================
    # Step 6 — Perform Dimer Saddle Search
    # ============================================================

    with DimerControl(
        initial_eigenmode_method="displacement",
        displacement_method="vector",  # NEW ASE versions
        f_rot_max=0.05,
        f_rot_min=0.001,
        logfile=str(logfile) if logfile else None,
    ) as control:

        d_atoms = MinModeAtoms(R, control)

        # apply *localized* initial displacement vector
        mask = [i in set(reactive_atoms) for i in range(len(R))]
        d_atoms.displace(displacement_vector=disp_vec, mask=mask)

        with MinModeTranslate(
            d_atoms,
            trajectory=str(traj_path),
            logfile=str(logfile) if logfile else None,
        ) as opt:

            if max_steps is None:
                opt.run(fmax=fmax)
            else:
                opt.run(fmax=fmax, steps=max_steps)

    write(output_xyz, d_atoms)
    print(f"[✓] Saddle point written to {output_xyz}")


# ============================================================
# 6) CLI ENTRY POINT (unchanged)
# ============================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ASE Dimer saddle point search with dxtb."
    )

    parser.add_argument("--reactant", default="dontdelete/test_out.xyz")
    parser.add_argument("--product", default="ase_ox.xyz")
    parser.add_argument("--method", default="GFN1")
    parser.add_argument("--displacement", type=float, default=0.5)
    parser.add_argument("--fmax", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--traj", default="dimer.traj")
    parser.add_argument("--log", default="dimer.log")
    parser.add_argument("--output", default="saddle_point.xyz")

    args = parser.parse_args()

    run_dimer_search(
        reactant_path=args.reactant,
        product_path=args.product,
        method=args.method,
        displacement_step=args.displacement,
        fmax=args.fmax,
        max_steps=args.steps,
        traj_path=args.traj,
        logfile=args.log or None,
        output_xyz=args.output,
    )


if __name__ == "__main__":
    main()
