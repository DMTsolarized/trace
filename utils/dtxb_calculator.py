import torch
import dxtb

from ase.calculators.calculator import Calculator, all_changes

# Units / conversion constants
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
DEBYE_PER_EA = 4.80320427  # 1 e·Å = 4.80320427 Debye
# e·Bohr -> Debye factor = BOHR_TO_ANGSTROM * DEBYE_PER_EA
EBOHR_TO_DEBYE = BOHR_TO_ANGSTROM * DEBYE_PER_EA


class DXTBCalculator(Calculator):
    """
    ASE Calculator wrapper for dxtb (GFN1 / GFN2).
    Lazily instantiates a dxtb calculator with the atomic numbers from the ASE Atoms object.

    Supported ASE properties: "energy", "forces", "charges", "dipole".
    """

    implemented_properties = ["energy", "forces", "charges", "dipole"]

    def __init__(
        self,
        method: str = "GFN1",
        device: str = "cpu",
        dtype: torch.dtype = torch.double,
        opts: dict | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        method : "GFN1" or "GFN2"
            Which dxtb parametrization to use.
        device : "cpu" or "cuda" or torch.device
            Torch device for tensors.
        dtype : torch.dtype
            Torch dtype for positions (float).
        opts : dict | None
            Options forwarded to the dxtb calculator (e.g., {"maxiter": 100}).
        kwargs : other args forwarded to ase.Calculator (label, ignore_changes, etc.)
        """
        super().__init__(**kwargs)
        self.method = method.upper()
        if self.method not in ("GFN1", "GFN2"):
            raise ValueError("Unsupported method: must be 'GFN1' or 'GFN2'")

        # device/dtype for dxtb tensors
        self.device = torch.device(device)
        self.dtype = dtype
        self.opts = opts

        # Backend cache: created on first calculate() when 'numbers' are known.
        self._calc = None
        self._calc_numbers_key = (
            None  # tuple of ints for the numbers used by the cached calculator
        )

    def _make_calc_for_numbers(self, numbers_tensor: torch.Tensor):
        """
        Instantiate (or reuse) a dxtb calculator for the given atomic numbers.
        numbers_tensor: torch.Tensor on any device with integer atomic numbers.
        """
        # key for caching: immutable tuple of ints
        nums_cpu = tuple(int(x) for x in numbers_tensor.cpu().tolist())

        if self._calc is not None and self._calc_numbers_key == nums_cpu:
            return self._calc

        # instantiate appropriate dxtb calculator
        dd = {"device": self.device, "dtype": self.dtype}

        if self.method == "GFN1":
            calc = dxtb.calculators.GFN1Calculator(numbers_tensor, opts=self.opts, **dd)
        else:  # GFN2
            calc = dxtb.calculators.GFN2Calculator(numbers_tensor, opts=self.opts, **dd)

        self._calc = calc
        self._calc_numbers_key = nums_cpu
        return self._calc

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        ASE-required calculate method.
        Requests contain property names in `properties`.
        """
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            raise ValueError("DXTBCalculator.calculate called with atoms=None")

        # GFN1/GFN2 are molecular; raise for periodic systems
        if getattr(atoms, "pbc", None) is not None and any(atoms.pbc):
            raise NotImplementedError(
                "GFN1/GFN2 (dxtb) do not support periodic boundary conditions in this wrapper."
            )

        # Prepare atomic numbers tensor (integer type)
        numbers = torch.tensor(
            list(atoms.numbers), device=self.device, dtype=torch.long
        )

        # Ensure backend calculator for these numbers exists (cached where possible)
        calc = self._make_calc_for_numbers(numbers)

        # Reset the backend to avoid autodiff caching issues if needed
        try:
            calc.reset()
        except Exception:
            # reset may not be strictly necessary, ignore if unavailable
            pass

        # Positions -> torch tensor in chosen dtype/device
        pos = torch.tensor(
            atoms.get_positions(),
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
        )
        # Convert to Bohr as dxtb expects positions in Bohr
        pos_bohr = pos / BOHR_TO_ANGSTROM

        # Results dict to be returned by ASE
        results = {}

        # Energy (compute if requested or if forces requested — many optimizers expect energy)
        if "energy" in properties or "forces" in properties:
            # dxtb returns energy in Hartree
            # API: calc.get_energy(position_tensor)
            energy_hartree = calc.get_energy(pos_bohr)
            energy_ev = float(energy_hartree.detach().cpu().item() * HARTREE_TO_EV)
            results["energy"] = energy_ev

        # Forces (Hartree / Bohr -> eV / Å)
        if "forces" in properties:
            forces_hb = calc.get_forces(
                pos_bohr
            )  # returns tensor (nat,3) in Hartree/Bohr
            forces_np = forces_hb.detach().cpu().numpy() * (
                HARTREE_TO_EV / BOHR_TO_ANGSTROM
            )
            results["forces"] = forces_np

        # Charges (electrons)
        if "charges" in properties:
            charges_t = calc.get_charges(pos_bohr)  # returns tensor (nat,)
            results["charges"] = charges_t.detach().cpu().numpy()

        # Dipole moment: use get_dipole_moment (returns e·Bohr) -> convert to Debye
        if "dipole" in properties:
            # prefer get_dipole_moment if available
            if hasattr(calc, "get_dipole_moment"):
                dip_bohr = calc.get_dipole_moment(pos_bohr)  # shape (3,)
            elif hasattr(calc, "get_dipole"):
                dip_bohr = calc.get_dipole(pos_bohr)
            else:
                raise RuntimeError("dxtb backend does not provide a dipole method")

            dip_debye = dip_bohr.detach().cpu().numpy() * EBOHR_TO_DEBYE
            results["dipole"] = dip_debye

        # Put computed results into ASE expected container
        self.results.update(results)
