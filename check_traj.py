from ase import Atoms
from ase.io import read, write
from ase.build import bulk, surface, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from gpaw import GPAW, PW
import numpy as np
from ase.calculators.eam import EAM
from ase.calculators.lj import LennardJones
from utils.dtxb_calculator import DXTBCalculator


atoms_pre = read("dimer.traj", index=":")
lsat_frame_2 = atoms_pre[-1]

write("dimer_last.xyz", lsat_frame_2)
