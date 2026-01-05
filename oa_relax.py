from __future__ import annotations

from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS
import torch

from utils.dtxb_calculator import DXTBCalculator


def relax_oa_guess(
    infile: Path | str,
    outfile: Path | str | None = None,
    method: str = "GFN1",
    fmax: float = 0.02,
) -> None:
    """Relax an oxidative-addition guess with GFN1 to the requested fmax."""
    infile = Path(infile)
    if outfile is None:
        outfile = infile.with_stem(infile.stem + "_relaxed")
    outfile = Path(outfile)

    atoms = read(infile)
    atoms.calc = DXTBCalculator(method=method)
    opt = BFGS(atoms, trajectory=str(Path(outfile).with_suffix(".traj")))
    opt.run(fmax=fmax)

    write(outfile, atoms)
    print(f"Wrote relaxed structure to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Relax an oxidative-addition guess with GFN1"
    )
    parser.add_argument(
        "infile",
        help="Input XYZ file to relax",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output file (default: append _relaxed before extension)",
        default=None,
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.02,
        help="Force convergence criterion (default 0.02 eV/Å)",
    )
    parser.add_argument(
        "--method",
        default="GFN1",
        help="xTB method (default GFN1)",
    )

    args = parser.parse_args()
    relax_oa_guess(args.infile, args.outfile, method=args.method, fmax=args.fmax)
