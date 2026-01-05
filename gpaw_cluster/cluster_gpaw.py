"""
Benzene adsorption on Pt(111) using GPAW
Optimized for 28-core CPU with MPI parallelization
Run with: mpirun -np 28 python benzene_pt_optimized.py
"""

from ase import Atoms
from ase.build import bulk, surface, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from gpaw import GPAW, PW, gpu, cgpaw
import numpy as np
import sys
from ase.io import write


# -------------------------
# Parameters
# -------------------------
a = 3.92  # Pt lattice constant (Å)
layers = 3  # slab depth
supercell = (6, 6, 1)
vacuum = 18.0
kpts = (4, 4, 1)  # k-points for surface
pw_cutoff = 500  # eV - adjust based on convergence tests
xc_functional = "optPBE-vdW"
relax_tol = 0.05  # eV/Å


print(gpu.cupy_is_fake)
# -------------------------
# Build Pt(111) slab
# -------------------------
print("Building Pt(111) slab...")
pt_bulk = bulk("Pt", "fcc", a=a)
slab = surface(pt_bulk, (1, 1, 1), layers)
slab = slab * supercell
slab.center(vacuum=vacuum, axis=2)

# Fix bottom layer
zs = slab.get_positions()[:, 2]
bottom_z = np.min(zs)
mask = [z == bottom_z for z in zs]
constraint = FixAtoms(mask=mask)
slab.set_constraint(constraint)

print(f"Slab: {len(slab)} Pt atoms, {sum(mask)} fixed")

# -------------------------
# Place benzene molecule
# -------------------------
benz = molecule("C6H6")


box = 20.0
benz_box = Atoms(
    positions=benz.positions,
    numbers=benz.numbers,
    cell=[[box, 0, 0], [0, box, 0], [0, 0, box]],
    pbc=(False, False, False),
)
benz_box.center()
# Position benzene ~3 Å above surface
# Combine slab and benzene
print(f"Calculating energy of Benzene in vacuum")
# calc_final_benz = GPAW(
#     filename="benz_calc.gpaw",
#     mode=PW(pw_cutoff),
#     xc=xc_functional,
#     kpts=(1, 1, 1),
#     setups={"Pt": "10"},
#     occupations={"name": "fermi-dirac", "width": 0.05},
#     convergence={"energy": 0.0005},
#     mixer={"beta": 0.05, "nmaxold": 5, "weight": 50},
#     symmetry={"point_group": False, "time_reversal": True},
#     txt="final_relax.txt",
#     parallel={"band": 12},
#     spinpol=True,
#     poissonsolver={"dipolelayer": "z"},
# )

# benz_box.calc = calc_final_benz


# opt_benz = BFGS(benz_box, trajectory="benz.traj")
# opt_benz.run(fmax=0.02)
# benz_box.calc.write("benz_calc.gpaw", mode="all")

print("Adding benzene molecule to slab")
cell = slab.get_cell()  # cell vectors as rows
# cartesian center in xy = 0.5*(a + b)
center_xy_cart = 0.5 * (cell[0] + cell[1])
top_z = np.max(slab.get_positions()[:, 2])
initial_height = 2.8  # Å above top Pt layer
desired_z = top_z + initial_height

benz_com = benz.get_center_of_mass()
translation = np.array(
    [
        center_xy_cart[0] - benz_com[0],
        center_xy_cart[1] - benz_com[1],
        desired_z - benz_com[2],
    ]
)
benz.translate(translation)

system = slab + benz
system.set_constraint(constraint)

print(f"Total system: {len(system)} atoms ({len(slab)} Pt + {len(benz)} C/H)")

write("slab_benz.xyz", system)
sys.exit()
# -------------------------
# Step 1: Quick pre-relaxation with lower accuracy
# -------------------------
print("\n" + "=" * 60)
print("Step 1: Pre-relaxation (lower cutoff, loose convergence)")
print("=" * 60)

calc_pre = GPAW(
    mode=PW(300),
    xc=xc_functional,
    kpts=kpts,
    setups={"Pt": "10"},  # 10 valence electrons for Pt
    occupations={"name": "fermi-dirac", "width": 0.2},
    convergence={"energy": 0.001},  # Loose convergence
    mixer={"beta": 0.05, "nmaxold": 5, "weight": 50},
    symmetry={"point_group": False, "time_reversal": True},
    txt="pre_relax.txt",
    parallel={"band": 12},
    spinpol=True,
    poissonsolver={"dipolelayer": "z"},
)

system.calc = calc_pre

print("Running pre-relaxation with BFGS...")
opt_pre = BFGS(system, trajectory="pre_relax.traj", logfile="pre_relax.log")
opt_pre.run(fmax=0.1)  # Loose force convergence

print(f"Pre-relaxation complete!")
print(f"Final energy: {system.get_potential_energy():.4f} eV")

# -------------------------
# Step 2: Final relaxation with full accuracy
# -------------------------
print("\n" + "=" * 60)
print("Step 2: Final relaxation (full cutoff, tight convergence)")
print("=" * 60)

calc_final = GPAW(
    mode=PW(pw_cutoff),
    xc=xc_functional,
    kpts=kpts,
    setups={"Pt": "10"},
    occupations={"name": "fermi-dirac", "width": 0.05},
    convergence={"energy": 0.0005},
    mixer={"beta": 0.05, "nmaxold": 5, "weight": 50},
    symmetry={"point_group": False, "time_reversal": True},
    txt="final_relax.txt",
    parallel={"band": 12},
    spinpol=True,
    poissonsolver={"dipolelayer": "z"},
)

system.calc = calc_final

print("Running final relaxation with BFGS...")
opt_final = BFGS(system, trajectory="final_relax.traj", logfile="final_relax.log")
opt_final.run(fmax=relax_tol)

# -------------------------
# Results
# -------------------------
print("\n" + "=" * 60)
print("CALCULATION COMPLETE")
print("=" * 60)

final_energy = system.get_potential_energy()
forces = system.get_forces()
max_force = np.max(np.abs(forces))

print(f"\nFinal energy: {final_energy:.6f} eV")
print(f"Max force: {max_force:.4f} eV/Å")

# Save final structure

write("benzene_pt_final.xyz", system)
write("benzene_pt_final.traj", system)

print("\nOutput files:")
print("  - pre_relax.txt, pre_relax.traj")
print("  - final_relax.txt, final_relax.traj")
print("  - benzene_pt_final.xyz")
print("  - benzene_pt_final.traj")

print("\nTo visualize: ase gui final_relax.traj")
print("=" * 60)
