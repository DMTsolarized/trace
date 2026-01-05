from __future__ import annotations

import numpy as np
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS
from ase.vibrations import Vibrations

from utils.dtxb_calculator import DXTBCalculator


def compute_lowest_mode(
    atoms,
    hessian_out: Path | str,
    delta: float = 0.01,
    vib_prefix: str = "vib",
):
    """
    Compute vibrational frequencies and normal modes using ASE Vibrations.
    Saves all frequencies to a text file and returns the lowest-frequency mode.
    """

    vib = Vibrations(atoms, name=vib_prefix, delta=delta)
    vib.run()

    # ASE vibrational frequencies in cm^-1
    freqs = vib.get_frequencies()

    # Modes: each is (natoms, 3)
    modes = [vib.get_mode(i) for i in range(len(freqs))]

    # Sort frequencies and modes together
    freq_mode_pairs = sorted(zip(freqs, modes), key=lambda x: x[0])
    lowest_freq, lowest_mode = freq_mode_pairs[0]

    # Save frequencies to file
    with open(hessian_out, "w") as f:
        f.write("VIBRATIONAL FREQUENCIES (cm^-1)\n")
        f.write("--------------------------------\n\n")
        for i, (fr, _) in enumerate(freq_mode_pairs):
            f.write(f"Mode {i:2d}: {fr:10.4f} cm^-1\n")

        f.write("\nLowest mode:\n")
        f.write(f"{lowest_freq:.4f} cm^-1\n")

    vib.clean()  # remove temporary vibration files

    return freqs, lowest_freq, lowest_mode


def relax_along_mode(
    atoms,
    mode: np.ndarray,
    step: float,
    method: str,
    fmax: float,
    traj_path: Path | str,
) -> "ase.Atoms":
    displaced = atoms.copy()
    displaced.set_positions(displaced.get_positions() + step * mode)
    displaced.calc = DXTBCalculator(method=method)
    opt = BFGS(displaced, trajectory=str(traj_path))
    opt.run(fmax=fmax)
    return displaced


def run_irc(
    saddle_path: Path | str = "saddle_point.xyz",
    method: str = "GFN1",
    delta: float = 0.01,
    step: float = 0.1,
    fmax: float = 0.05,
    hessian_out: Path | str = "hessian.npy",
    vib_prefix: str = "vib",
    forward_xyz: Path | str = "irc_forward.xyz",
    reverse_xyz: Path | str = "irc_reverse.xyz",
    forward_traj: Path | str = "irc_forward.traj",
    reverse_traj: Path | str = "irc_reverse.traj",
) -> None:
    saddle = read(saddle_path)
    saddle.calc = DXTBCalculator(method=method)

    _freqs, _lowest_freq, mode = compute_lowest_mode(
        saddle, delta=delta, vib_prefix=vib_prefix, hessian_out=hessian_out
    )

    forward = relax_along_mode(saddle, mode, step, method, fmax, traj_path=forward_traj)
    reverse = relax_along_mode(
        saddle, -mode, step, method, fmax, traj_path=reverse_traj
    )

    write(forward_xyz, forward)
    write(reverse_xyz, reverse)
    print(f"Forward IRC product written to {forward_xyz}")
    print(f"Reverse IRC product written to {reverse_xyz}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Hessian and simple IRC (downhill optimizations) from a saddle point using DXTB."
    )
    parser.add_argument(
        "--saddle",
        default="saddle_point.xyz",
        help="Input saddle-point geometry (default: saddle_point.xyz)",
    )
    parser.add_argument(
        "--method",
        default="GFN1",
        help="xTB method for DXTB calculator (default: GFN1)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="Finite difference step for Hessian (default: 0.01 Å)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Initial displacement along mode for IRC (default: 0.1 Å)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion for IRC optimizations (default: 0.05 eV/Å)",
    )
    parser.add_argument(
        "--hessian-out",
        default="hessian.txt",
        help="Path to save Hessian matrix (default: hessian.txt)",
    )
    parser.add_argument(
        "--vib-prefix",
        default="vib",
        help="Prefix for ASE Vibrations files (default: vib)",
    )
    parser.add_argument(
        "--forward-xyz",
        default="irc_forward.xyz",
        help="Relaxed forward IRC structure (default: irc_forward.xyz)",
    )
    parser.add_argument(
        "--reverse-xyz",
        default="irc_reverse.xyz",
        help="Relaxed reverse IRC structure (default: irc_reverse.xyz)",
    )
    parser.add_argument(
        "--forward-traj",
        default="irc_forward.traj",
        help="Trajectory file for forward IRC optimization (default: irc_forward.traj)",
    )
    parser.add_argument(
        "--reverse-traj",
        default="irc_reverse.traj",
        help="Trajectory file for reverse IRC optimization (default: irc_reverse.traj)",
    )

    args = parser.parse_args()
    run_irc(
        saddle_path=args.saddle,
        method=args.method,
        delta=args.delta,
        step=args.step,
        fmax=args.fmax,
        hessian_out=args.hessian_out,
        vib_prefix=args.vib_prefix,
        forward_xyz=args.forward_xyz,
        reverse_xyz=args.reverse_xyz,
        forward_traj=args.forward_traj,
        reverse_traj=args.reverse_traj,
    )
