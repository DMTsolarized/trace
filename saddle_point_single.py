from __future__ import annotations

from pathlib import Path

from ase.io import read, write
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

from utils.dtxb_calculator import DXTBCalculator
from saddle_point import (
    build_atom_mapping,
    detect_reactive_atoms,
    kabsch_align,
)


def run_dimer_search(
    start_path: Path | str = "dontdelete/test_out.xyz",
    product_path: Path | str | None = None,
    method: str = "GFN1",
    gauss_std: float = 0.1,
    fmax: float = 0.05,
    max_steps: int | None = None,
    traj_path: Path | str = "dimer.traj",
    logfile: Path | str | None = "dimer.log",
    output_xyz: Path | str = "saddle_point.xyz",
) -> None:
    """Single-ended Dimer search: random Gaussian displacement; optional product mask for reactive atoms."""
    start = read(start_path)
    start.calc = DXTBCalculator(method=method)

    with DimerControl(
        initial_eigenmode_method="gauss",
        displacement_method="gauss",
        gauss_std=gauss_std,
        logfile=str(logfile) if logfile is not None else None,
    ) as control:
        d_atoms = MinModeAtoms(start, control)
        d_atoms.displace()

        with MinModeTranslate(
            d_atoms,
            trajectory=str(traj_path),
            logfile=str(logfile) if logfile else None,
        ) as dim_rlx:
            if max_steps is None:
                dim_rlx.run(fmax=fmax)
            else:
                dim_rlx.run(fmax=fmax, steps=max_steps)

    write(output_xyz, d_atoms)
    print(f"Saddle point structure written to {output_xyz}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ASE Dimer saddle point search with dxtb."
    )
    parser.add_argument(
        "--start",
        default="dontdelete/test_out.xyz",
        help="Starting geometry (default: dontdelete/test_out.xyz)",
    )
    parser.add_argument(
        "--method",
        default="GFN1",
        help="xTB method for DXTB calculator (default: GFN1)",
    )
    parser.add_argument(
        "--product",
        default="ase_ox.xyz",
        help="Optional product geometry to guess reactive atoms and mask displacement",
    )
    parser.add_argument(
        "--gauss-std",
        type=float,
        default=0.1,
        help="Standard deviation (Å) for Gaussian initial displacement (default: 0.1)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion for MinModeTranslate (default: 0.05 eV/Å)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Maximum optimizer steps (default: ASE default)",
    )
    parser.add_argument(
        "--traj",
        default="dimer.traj",
        help="Trajectory file to write optimization path (default: dimer.traj)",
    )
    parser.add_argument(
        "--log",
        default="dimer.log",
        help="Log file for dimer optimizer (default: dimer.log; pass '' to silence)",
    )
    parser.add_argument(
        "--output",
        default="saddle_point.xyz",
        help="Output XYZ file for saddle-point geometry (default: saddle_point.xyz)",
    )

    args = parser.parse_args()
    run_dimer_search(
        start_path=args.start,
        method=args.method,
        product_path=args.product,
        gauss_std=args.gauss_std,
        fmax=args.fmax,
        max_steps=args.steps,
        traj_path=args.traj,
        logfile=args.log or None,
        output_xyz=args.output,
    )
