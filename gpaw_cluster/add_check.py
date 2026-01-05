# benzene_on_pt.py
# Requires: ase, gpaw (compiled with libvdwxc for optPBE-vdW if possible)
# export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

from ase import Atoms
from ase.build import bulk, surface, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS

from gpaw import GPAW, PW
import numpy as np
from ase.calculators.eam import EAM

# -------------------------
# Parameters
# -------------------------
a = 3.92  # Pt lattice constant (Å)
layers = 3  # slab depth (bottom 1–2 layers fixed)
supercell = (3, 3, 1)
vacuum = 18.0
kpts_relax = (2, 2, 1)  # k-points for pre-relaxation
kpts_final = (1, 1, 1)
pw_cut_relax = 400
pw_cut_final = 200
xc_functional = "optPBE-vdW"
relax_tol_pre = 0.05
relax_tol_final = 0.02

# -------------------------
# Build Pt(111) slab
# -------------------------
pt_bulk = bulk("Pt", "fcc", a=a)
slab = surface(pt_bulk, (1, 1, 1), layers)
slab: Atoms = slab * supercell
slab.center(vacuum=vacuum, axis=2)
# Fix bottom layer
zs = slab.get_positions()[:, 2]
bottom_z = np.min(zs)
mask = [z == bottom_z for z in zs]
const = FixAtoms(mask=mask)
slab.set_constraint(FixAtoms(mask=mask))
# -------------------------
# Place benzene molecule
# -------------------------
benz: Atoms = molecule("C6H6")
# orient benzene: flat (normal along z), scale/translate to sit ~2.5 Å above surface
benz.rotate("x", 90, rotate_cell=False)  # make benzene plane parallel to surface
# initial height above top-most Pt atoms
top_z = max(slab.get_positions()[:, 2])
benz_center = benz.get_center_of_mass()
benz.translate([0, 0, top_z + 3.0 - benz_center[2]])  # start ~3 Å above surface

# -------------------------
# Calculator (GPAW) setup
# -------------------------
# Use plane-wave mode (PW) here. If you have a GPAW real-space/GPU build prefer mode='fd' or GPU settings.
# calc_kwargs = dict(
#     mode=PW(pw_cut),
#     kpts=kpts,
#     setups={"Pt": "10"},
#     xc="PBE",
#     txt="gpaw_out.txt",
# )

# Create GPAW calculator
slab.calc = EAM(potential="Pt.eam.alloy")

optimizer = BFGS(slab, trajectory="pre_optimize_slab.traj")
optimizer.run(fmax=0.02)

print("done")
slab_and_benz: Atoms = slab + benz

calc_kwargs = dict(
    mode=PW(pw_cut_final),
    kpts=kpts_final,
    setups={"Pt": "10"},
    xc="optPBE-vdW",
    txt="gpaw_out.txt",
)

gpaw_calc = GPAW(**calc_kwargs)
slab_and_benz.calc = gpaw_calc

optimizer = BFGS(slab_and_benz, trajectory="post_opt.traj")
optimizer.run(fmax=0.02)

# -------------------------
# Relaxation: only unconstrained atoms move
# -------------------------
