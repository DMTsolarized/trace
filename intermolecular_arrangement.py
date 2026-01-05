"""
Intramolecular arrangement sampler (single structure, two sites).

Takes one molecule, splits it into two logical fragments by index lists, and
samples approach geometries between the two sites using the same clash and
alignment scoring as the precomplex builder utilities.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from ase.io import iread, write

from utils.precomplex_builder import (
    ReactiveSite,
    PrecomplexBuilder,
    BondSigmaStarApproach,
    PiFaceApproach,
    MixedApproach,
    EnergyScorer,
)


def parse_int_list(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_distances(raw: str) -> np.ndarray:
    """Comma-separated list or start:stop:count linspace."""
    if ":" in raw:
        start, stop, count = (float(x) if i < 2 else int(x) for i, x in enumerate(raw.split(":")))
        return np.linspace(start, stop, count)
    return np.fromstring(raw, sep=",", dtype=float)


def rel_index(abs_idx: int, subset: List[int], label: str) -> int:
    if abs_idx not in subset:
        raise ValueError(f"{label}={abs_idx} not contained in specified fragment indices {subset}")
    return subset.index(abs_idx)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Intramolecular arrangement builder (one structure, two reactive sites)."
    )
    p.add_argument("--structure", required=True, help="Path to structure (ASE-readable).")
    p.add_argument(
        "--fragA",
        required=True,
        help="Comma-separated atom indices defining fragment A (will be moved).",
    )
    p.add_argument(
        "--fragB",
        required=True,
        help="Comma-separated atom indices defining fragment B (anchor).",
    )
    p.add_argument("--siteA", type=int, required=True, help="Reactive atom index (absolute) in fragA.")
    p.add_argument("--siteB", type=int, required=True, help="Reactive atom index (absolute) in fragB.")
    p.add_argument("--partnerB", type=int, default=None, help="Optional partner atom (absolute) on fragB for sigma*.")
    p.add_argument(
        "--planeB",
        type=str,
        default="",
        help="Comma-separated absolute atom indices defining a plane on fragB (pi-face).",
    )
    p.add_argument(
        "--distances",
        type=str,
        default="2.8:3.4:5",
        help="Distances to sample (comma list or start:stop:count).",
    )
    p.add_argument("--top", type=int, default=5, help="How many arrangements to write.")
    p.add_argument("--outdir", type=str, default="intramol_arrange", help="Output directory.")
    p.add_argument(
        "--energy",
        action="store_true",
        help="If set, annotate energies via xTB (can be slower).",
    )
    args = p.parse_args()

    full = list(iread(args.structure))[-1]
    fragA_indices = parse_int_list(args.fragA)
    fragB_indices = parse_int_list(args.fragB)
    plane_indices_abs = parse_int_list(args.planeB)

    if not fragA_indices or not fragB_indices:
        raise ValueError("Both fragA and fragB index lists must be provided.")
    siteA_rel = rel_index(args.siteA, fragA_indices, "siteA")
    siteB_rel = rel_index(args.siteB, fragB_indices, "siteB")
    partnerB_rel = None
    if args.partnerB is not None:
        partnerB_rel = rel_index(args.partnerB, fragB_indices, "partnerB")
    plane_indices_rel = [rel_index(i, fragB_indices, "planeB") for i in plane_indices_abs] if plane_indices_abs else []

    fragA = full[fragA_indices]
    fragB = full[fragB_indices]

    siteA = ReactiveSite(atom_idx=siteA_rel, site_type="fragA_site")
    siteB = ReactiveSite(
        atom_idx=siteB_rel,
        partner_idx=partnerB_rel,
        plane_indices=plane_indices_rel,
        site_type="fragB_site",
    )

    distances = parse_distances(args.distances)
    builder = PrecomplexBuilder(
        fragA,
        fragB,
        siteA,
        siteB,
        distances=distances,
    )

    sigma = BondSigmaStarApproach() if partnerB_rel is not None else None
    pi_face = PiFaceApproach() if plane_indices_rel else None
    if sigma:
        builder.add_approach_generator(sigma)
    if pi_face:
        builder.add_approach_generator(pi_face)
    if sigma and pi_face:
        builder.add_approach_generator(
            MixedApproach(sigma, pi_face, weights=np.linspace(0.0, 0.9, 10))
        )

    if args.energy:
        builder.set_energy_scorer(EnergyScorer(method="GFN1", spin=1))

    results = builder.build(max_keep=max(args.top, 30), do_energy=args.energy, max_energy_eval=args.top)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for i, cand in enumerate(results[: args.top]):
        fragA_copy = fragA.copy()
        fragA_copy.set_positions(cand["at_pos"])
        merged = fragA_copy + fragB
        out_path = outdir / f"arrangement_{i}.xyz"
        write(out_path.as_posix(), merged)

        summary.append(
            {
                "file": out_path.as_posix(),
                "clash_score": cand["clash_score"],
                "alignment_score": cand["alignment_score"],
                "distance": cand["distance"],
                "approach": cand["approach"].tolist(),
                "inside_bite": cand["inside_bite"],
                **({"energy": cand["energy"]} if "energy" in cand else {}),
            }
        )

    with (outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(summary)} arrangements to {outdir}")


if __name__ == "__main__":
    main()
