from __future__ import annotations

import io
import importlib
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Literal

from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.io import read as ase_read
import yaml
from ase.optimize.optimize import Optimizer


def _default_optimizer() -> type[Optimizer]:
    return getattr(importlib.import_module("ase.optimize"), "LBFGS")


class ValidationError(ValueError):
    """Raised when the YAML structure is missing required fields or is invalid."""


class ReactantType(str, Enum):
    ORGANIC = "organic"
    ORGANOMETALLIC = "organometallic"
    XYZ = "xyz"
    METAL = "metal"


REACTANT_TYPES_DEFAULT = {item.value for item in ReactantType}


class CalculatorType(str, Enum):
    DXTB = "dxtb"
    ORCA = "orca"


CALCULATOR_TYPES_DEFAULT = {item.value for item in CalculatorType}


def _default_calculator_factory() -> (
    Callable[[Optional[dict[str, Any]]], BaseCalculator]
):
    from utils.dtxb_calculator import DXTBCalculator

    def _factory(overrides: Optional[dict[str, Any]] = None) -> BaseCalculator:
        merged_options: dict[str, Any] = {}
        if overrides:
            merged_options.update(overrides)
        return DXTBCalculator(**merged_options)

    return _factory


@dataclass(frozen=True)
class Reaction:
    name: str
    type: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetalCenter:
    element: str
    oxidation_state: int
    spin: int
    coordination: int
    geometry: str


@dataclass(frozen=True)
class Ligand:
    name: str
    count: int
    smicat: Optional[int]


@dataclass(frozen=True)
class ReactiveSite:
    atom_idx: int
    partner_idx: Optional[int] = None
    plane_indices: list[int] = field(default_factory=list)
    site_type: str = "generic"


@dataclass(frozen=True)
class Reactant:
    id: str
    type: str
    smiles: Optional[str] = None
    xyz_path: Optional[str] = None
    atoms: Optional[Atoms] = None
    metal: Optional[MetalCenter] = None
    ligands: list[Ligand] = field(default_factory=list)
    reactive_centers: list[ReactiveSite] = field(default_factory=list)


@dataclass(frozen=True)
class BondFormationSettings:
    atom_pairs: list[tuple[str, str]] = field(default_factory=list)
    distance_scale: float = 1.35
    distance_range: tuple[float, float] = (0.9, 1.1)
    distances: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class SamplingSettings:
    rotations: bool = False
    rotation_angles: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizationSettings:
    optimizer: type[Optimizer] = field(default_factory=_default_optimizer)
    optimizer_name: str = "LBFGS"
    optimizer_module: Optional[str] = None


@dataclass(frozen=True)
class CalculatorSettings:
    name: str
    steps: int = 10
    fmax: float = 0.5
    options: dict[str, Any] = field(default_factory=dict)
    calculator_factory: Callable[[Optional[dict[str, Any]]], BaseCalculator] = field(
        default_factory=_default_calculator_factory
    )
    move: Literal[False] | dict[str, float | str] = False


@dataclass(frozen=True)
class ReactionSettings:
    sample_size: int
    bond_formation: Optional[BondFormationSettings] = None
    sampling: Optional[SamplingSettings] = None
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)


@dataclass(frozen=True)
class ReactionDefinition:
    reaction: Reaction
    reactants: list[Reactant]
    workdir: str
    settings: ReactionSettings = field(default_factory=ReactionSettings)
    pipeline: list[CalculatorSettings] = field(default_factory=list)


def load_reaction_from_file(
    path: str,
    *,
    reaction_types: Optional[Iterable[str]] = None,
    reactant_types: Optional[Iterable[str]] = None,
) -> ReactionDefinition:
    """Load and validate a reaction definition from a YAML file."""

    with open(path, "r", encoding="utf-8") as handle:
        base_dir = os.path.dirname(os.path.abspath(path))
        return load_reaction_from_str(
            handle.read(),
            reaction_types=reaction_types,
            reactant_types=reactant_types,
            base_dir=base_dir,
        )


def load_reaction_from_str(
    yaml_str: str,
    *,
    reaction_types: Optional[Iterable[str]] = None,
    reactant_types: Optional[Iterable[str]] = None,
    base_dir: Optional[str] = None,
) -> ReactionDefinition:
    """Load and validate a reaction definition from a YAML string."""

    data = yaml.safe_load(yaml_str)
    return parse_reaction_dict(
        data,
        reaction_types=reaction_types,
        reactant_types=reactant_types,
        base_dir=base_dir,
    )


def parse_reaction_dict(
    data: object,
    *,
    reaction_types: Optional[Iterable[str]] = None,
    reactant_types: Optional[Iterable[str]] = None,
    base_dir: Optional[str] = None,
) -> ReactionDefinition:
    if not isinstance(data, dict):
        raise ValidationError("Top-level YAML must be a mapping.")

    reaction_types_set = set(reaction_types) if reaction_types is not None else None
    reactant_types_set = set(reactant_types or REACTANT_TYPES_DEFAULT)

    reaction_data = _require_mapping(data, "reaction")
    reaction = _parse_reaction(reaction_data, reaction_types_set)

    workdir = _require_str(data, "workdir", "workdir")

    reactants_data = _require_list(data, "reactants")
    reactants = _parse_reactants(reactants_data, reactant_types_set, base_dir)

    settings_data = data.get("settings")
    settings = _parse_settings(settings_data)
    pipeline = _parse_pipeline(data.get("pipeline"))

    return ReactionDefinition(
        reaction=reaction,
        reactants=reactants,
        workdir=workdir,
        settings=settings,
        pipeline=pipeline,
    )


def _parse_reaction(
    reaction_data: dict, reaction_types: Optional[set[str]]
) -> Reaction:
    name = _require_str(reaction_data, "name", "reaction.name")
    reaction_type = _require_str(reaction_data, "type", "reaction.type")
    _validate_no_spaces(reaction_type, "reaction.type")
    if reaction_types is not None:
        _validate_enum(reaction_type, reaction_types, "reaction.type")
    tags = _optional_str_list(reaction_data.get("tags"), "reaction.tags")
    return Reaction(name=name, type=reaction_type, tags=tags)


def _parse_reactants(
    reactants_data: list,
    reactant_types: set[str],
    base_dir: Optional[str],
) -> list[Reactant]:
    reactants: list[Reactant] = []
    for idx, reactant_data in enumerate(reactants_data):
        path = f"reactants[{idx}]"
        if not isinstance(reactant_data, dict):
            raise ValidationError(f"{path} must be a mapping.")

        reactant_id = _require_str(reactant_data, "id", f"{path}.id")
        reactant_type = _require_str(reactant_data, "type", f"{path}.type")
        _validate_enum(reactant_type, reactant_types, f"{path}.type")

        smiles = _optional_str(reactant_data.get("smiles"), f"{path}.smiles")
        xyz_path = _optional_str(reactant_data.get("xyz"), f"{path}.xyz")
        metal = _parse_metal(reactant_data.get("metal"), f"{path}.metal")
        if reactant_type == ReactantType.ORGANOMETALLIC.value and metal is None:
            raise ValidationError(f"{path}.metal is required for organometallic.")
        ligands = _parse_ligands(reactant_data.get("ligands"), f"{path}.ligands")
        atoms = (
            _parse_xyz_atoms(xyz_path, base_dir, f"{path}.xyz") if xyz_path else None
        )
        if atoms is None and reactant_type in {
            ReactantType.ORGANOMETALLIC.value,
        }:
            if metal is None:
                raise ValidationError(f"{path}.metal is required to build a structure.")
            generated_xyz = _generate_molsimplify_xyz(
                metal,
                ligands,
                f"{path}.metal",
            )
            atoms = _parse_xyz_atoms_from_string(generated_xyz, f"{path}.xyz")
        reactive_centers = _parse_reactive_centers(
            reactant_data.get("reactive_centers"),
            f"{path}.reactive_centers",
        )
        if reactant_type == ReactantType.ORGANOMETALLIC.value and not reactive_centers:
            if atoms is None:
                raise ValidationError(
                    f"{path}.xyz or molSimplify generation is required."
                )
            if metal is None:
                raise ValidationError(f"{path}.metal is required for reactive center.")
            reactive_centers = [
                _default_metal_reactive_center(atoms, metal.element, path)
            ]

        reactants.append(
            Reactant(
                id=reactant_id,
                type=reactant_type,
                smiles=smiles,
                xyz_path=xyz_path,
                atoms=atoms,
                metal=metal,
                ligands=ligands,
                reactive_centers=reactive_centers,
            )
        )

    return reactants


def _parse_metal(metal_data: object, path: str) -> Optional[MetalCenter]:
    if metal_data is None:
        return None
    if not isinstance(metal_data, dict):
        raise ValidationError(f"{path} must be a mapping.")

    element = _require_str(metal_data, "element", f"{path}.element")
    oxidation_state = _require_int(
        metal_data, "oxidation_state", f"{path}.oxidation_state"
    )
    spin = _require_int(metal_data, "spin", f"{path}.spin")
    coordination = _require_int(metal_data, "coordination", f"{path}.coordination")
    geometry = _require_str(metal_data, "geometry", f"{path}.geometry")

    return MetalCenter(
        element=element,
        oxidation_state=oxidation_state,
        spin=spin,
        coordination=coordination,
        geometry=geometry,
    )


def _parse_ligands(ligands_data: object, path: str) -> list[Ligand]:
    if ligands_data is None:
        return []
    if not isinstance(ligands_data, list):
        raise ValidationError(f"{path} must be a list.")

    ligands: list[Ligand] = []

    for idx, ligand_data in enumerate(ligands_data):
        ligand_path = f"{path}[{idx}]"
        if not isinstance(ligand_data, dict):
            raise ValidationError(f"{ligand_path} must be a mapping.")
        name = _require_str(ligand_data, "name", f"{ligand_path}.name")
        count = _require_int(ligand_data, "count", f"{ligand_path}.count")
        smicat = _optional_int(ligand_data.get("smicat"), f"{ligand_path}.smicat")
        ligands.append(Ligand(name=name, count=count, smicat=smicat))

    return ligands


def _parse_reactive_centers(centers_data: object, path: str) -> list[ReactiveSite]:
    if centers_data is None:
        return []
    if not isinstance(centers_data, list):
        raise ValidationError(f"{path} must be a list.")

    centers: list[ReactiveSite] = []
    for idx, center_data in enumerate(centers_data):
        center_path = f"{path}[{idx}]"
        if not isinstance(center_data, dict):
            raise ValidationError(f"{center_path} must be a mapping.")
        atom_idx = (
            _require_int(
                center_data,
                "atom_idx",
                f"{center_path}.atom_idx",
            )
            if "atom_idx" in center_data
            else _require_int(
                center_data,
                "atom",
                f"{center_path}.atom",
            )
        )
        partner_idx = _optional_int(
            center_data.get("partner_idx"),
            f"{center_path}.partner_idx",
        )
        plane_indices = _optional_int_list(
            center_data.get("plane_indices"),
            f"{center_path}.plane_indices",
        )
        site_type = (
            _optional_str(
                center_data.get("site_type"),
                f"{center_path}.site_type",
            )
            or "generic"
        )
        centers.append(
            ReactiveSite(
                atom_idx=atom_idx,
                partner_idx=partner_idx,
                plane_indices=plane_indices,
                site_type=site_type,
            )
        )

    return centers


def _parse_settings(settings_data: object) -> ReactionSettings:
    if settings_data is None:
        raise ValidationError("Missing required field: settings.sample_size")
    if not isinstance(settings_data, dict):
        raise ValidationError("settings must be a mapping.")

    sample_size = _require_int(settings_data, "sample_size", "settings.sample_size")

    bond_formation = None
    if "bond_formation" in settings_data:
        bond_formation = _parse_bond_formation(settings_data.get("bond_formation"))

    sampling = None
    if "sampling" in settings_data:
        sampling = _parse_sampling(settings_data.get("sampling"))

    optimization = None
    if "optimization" in settings_data:
        optimization = _parse_optimization(settings_data.get("optimization"))
    else:
        optimization = _parse_optimization({})

    return ReactionSettings(
        sample_size=sample_size,
        bond_formation=bond_formation,
        sampling=sampling,
        optimization=optimization,
    )


def _parse_pipeline(data: object) -> list[CalculatorSettings]:
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValidationError("pipeline must be a list.")

    calculators: list[CalculatorSettings] = []
    for idx, entry in enumerate(data):
        entry_path = f"pipeline[{idx}]"
        if not isinstance(entry, dict):
            raise ValidationError(f"{entry_path} must be a mapping.")
        name = _require_str(entry, "name", f"{entry_path}.name")
        _validate_no_spaces(name, f"{entry_path}.name")
        name_norm = name.strip().lower()
        _validate_enum(name_norm, CALCULATOR_TYPES_DEFAULT, f"{entry_path}.name")
        steps = _optional_int(entry.get("steps"), f"{entry_path}.steps") or 10
        fmax = _optional_float(entry.get("fmax"), f"{entry_path}.fmax") or 0.5
        move = _parse_calculator_move(entry.get("move"), f"{entry_path}.move")
        options_raw = entry.get("options", {})
        if options_raw is None:
            options_raw = {}
        if not isinstance(options_raw, dict):
            raise ValidationError(f"{entry_path}.options must be a mapping.")
        calculator_factory = _build_calculator_factory(
            name_norm,
            options_raw,
            entry_path,
        )
        calculators.append(
            CalculatorSettings(
                name=name_norm,
                steps=steps,
                fmax=fmax,
                options=options_raw,
                calculator_factory=calculator_factory,
                move=move,
            )
        )
    return calculators


def _build_calculator(
    name: str,
    options: dict[str, Any],
    path: str,
) -> BaseCalculator:
    if name == CalculatorType.DXTB.value:
        from utils.dtxb_calculator import DXTBCalculator

        return DXTBCalculator(**options)
    if name == CalculatorType.ORCA.value:
        try:
            from ase.calculators.orca import ORCA
        except Exception as exc:  # pragma: no cover - depends on environment
            raise ValidationError(
                f"Failed to import ORCA calculator for {path}: {exc}"
            ) from exc
        return ORCA(**options)
    raise ValidationError(f"Unsupported calculator '{name}' for {path}.")


def _build_calculator_factory(
    name: str,
    options: dict[str, Any],
    path: str,
) -> Callable[[Optional[dict[str, Any]]], BaseCalculator]:
    def _factory(overrides: Optional[dict[str, Any]] = None) -> BaseCalculator:
        merged_options = dict(options)
        if overrides:
            merged_options.update(overrides)
        return _build_calculator(name, merged_options, path)

    return _factory


def _parse_calculator_move(
    value: object,
    path: str,
) -> Literal[False] | dict[str, float | str]:
    if value is None or value is False:
        return False
    if value is True:
        raise ValidationError(f"{path} must be false or a mapping.")
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must be false or a mapping.")
    start = _optional_float_or_auto(value.get("start"), f"{path}.start")
    target = _optional_float_or_auto(value.get("target"), f"{path}.target")
    step = _optional_float_or_auto(value.get("step"), f"{path}.step")
    return {"start": start, "target": target, "step": step}


def _optional_float_or_auto(value: object, path: str) -> float | str:
    if value is None:
        raise ValidationError(f"{path} is required when move is a mapping.")
    if isinstance(value, str):
        if value != "auto":
            raise ValidationError(f"{path} must be a number or 'auto'.")
        return value
    return _ensure_float(value, path)


def _parse_bond_formation(data: object) -> Optional[BondFormationSettings]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValidationError("settings.bond_formation must be a mapping.")

    atom_pairs_raw = data.get("atom_pairs")
    atom_pairs: list[tuple[str, str]] = []
    if atom_pairs_raw is not None:
        if not isinstance(atom_pairs_raw, list):
            raise ValidationError("settings.bond_formation.atom_pairs must be a list.")
        for idx, pair in enumerate(atom_pairs_raw):
            pair_path = f"settings.bond_formation.atom_pairs[{idx}]"
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValidationError(f"{pair_path} must be a 2-item list.")
            left = _ensure_str(pair[0], f"{pair_path}[0]")
            right = _ensure_str(pair[1], f"{pair_path}[1]")
            atom_pairs.append((left, right))

    distance_scale = (
        _optional_float(
            data.get("distance_scale"), "settings.bond_formation.distance_scale"
        )
        or 1.35
    )

    distance_range_raw = data.get("distance_range")
    distance_range = (0.9, 1.1)
    if distance_range_raw is not None:
        if not isinstance(distance_range_raw, list) or len(distance_range_raw) != 2:
            raise ValidationError(
                "settings.bond_formation.distance_range must be a 2-item list."
            )
        distance_range = (
            _ensure_float(
                distance_range_raw[0],
                "settings.bond_formation.distance_range[0]",
            ),
            _ensure_float(
                distance_range_raw[1],
                "settings.bond_formation.distance_range[1]",
            ),
        )

    distances_raw = data.get("distances")
    distances: list[float] = []
    if distances_raw is not None:
        if not isinstance(distances_raw, list):
            raise ValidationError("settings.bond_formation.distances must be a list.")
        distances = [
            _ensure_float(item, f"settings.bond_formation.distances[{idx}]")
            for idx, item in enumerate(distances_raw)
        ]

    return BondFormationSettings(
        atom_pairs=atom_pairs,
        distance_scale=distance_scale,
        distance_range=distance_range,
        distances=distances,
    )


def _parse_sampling(data: object) -> Optional[SamplingSettings]:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValidationError("settings.sampling must be a mapping.")

    rotations = _optional_bool(data.get("rotations"), "settings.sampling.rotations")
    rotation_angles_raw = data.get("rotation_angles")
    rotation_angles: list[float] = []
    if rotation_angles_raw is not None:
        if not isinstance(rotation_angles_raw, list):
            raise ValidationError("settings.sampling.rotation_angles must be a list.")
        for idx, angle in enumerate(rotation_angles_raw):
            rotation_angles.append(
                _ensure_float(angle, f"settings.sampling.rotation_angles[{idx}]")
            )

    return SamplingSettings(rotations=rotations, rotation_angles=rotation_angles)


def _parse_optimization(data: object) -> OptimizationSettings:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValidationError("settings.optimization must be a mapping.")

    optimizer_name = "LBFGS"
    optimizer_module = None
    optimizer_data = data.get("optimizer")
    if optimizer_data is not None:
        if isinstance(optimizer_data, str):
            optimizer_name = _ensure_str(
                optimizer_data,
                "settings.optimization.optimizer",
            )
        elif isinstance(optimizer_data, dict):
            optimizer_name = _require_str(
                optimizer_data,
                "name",
                "settings.optimization.optimizer.name",
            )
            optimizer_module = _optional_str(
                optimizer_data.get("module"),
                "settings.optimization.optimizer.module",
            )
        else:
            raise ValidationError(
                "settings.optimization.optimizer must be a string or mapping."
            )
    optimizer = _load_optimizer(
        optimizer_name,
        optimizer_module,
        "settings.optimization.optimizer",
    )

    return OptimizationSettings(
        optimizer_name=optimizer_name,
        optimizer_module=optimizer_module,
        optimizer=optimizer,
    )


def _load_optimizer(
    name: str,
    module: Optional[str],
    path: str,
) -> type[Optimizer]:
    module_name = module or "ase.optimize"
    try:
        optimizer_module = importlib.import_module(module_name)
    except Exception as exc:
        raise ValidationError(
            f"Failed to import optimizer module '{module_name}' for {path}: {exc}"
        ) from exc
    try:
        return getattr(optimizer_module, name)
    except AttributeError as exc:
        raise ValidationError(
            f"Optimizer '{name}' not found in module '{module_name}' for {path}."
        ) from exc


def _parse_xyz_atoms(
    xyz_path: str,
    base_dir: Optional[str],
    path: str,
) -> Atoms:
    resolved_path = _resolve_path(xyz_path, base_dir)
    if not os.path.exists(resolved_path):
        raise ValidationError(f"{path} file not found: {resolved_path}")
    try:
        atoms = ase_read(resolved_path, index=0)
    except Exception as exc:  # pragma: no cover - ASE error formats vary
        raise ValidationError(f"Failed to read {path}: {exc}") from exc
    if isinstance(atoms, list):
        raise ValidationError(f"{path} must contain a single structure.")
    return atoms


def _resolve_path(value: str, base_dir: Optional[str]) -> str:
    if base_dir and not os.path.isabs(value):
        return os.path.join(base_dir, value)
    return value


def _parse_xyz_atoms_from_string(xyz: str, path: str) -> Atoms:
    try:
        atoms = ase_read(io.StringIO(xyz), format="xyz")
    except Exception as exc:  # pragma: no cover - ASE error formats vary
        raise ValidationError(f"Failed to read {path}: {exc}") from exc
    if isinstance(atoms, list):
        raise ValidationError(f"{path} must contain a single structure.")
    return atoms


def _generate_molsimplify_xyz(
    metal: MetalCenter,
    ligands: list[Ligand],
    path: str,
) -> str:
    try:
        from molSimplify.Scripts.generator import startgen_pythonic
    except Exception as exc:  # pragma: no cover - depends on environment
        raise ValidationError(
            f"molSimplify is required to generate {path}: {exc}"
        ) from exc

    cmd_dict = {"-rprompt": "True"}
    cmd_dict["-core"] = metal.element.lower()
    cmd_dict["-geometry"] = metal.geometry.lower()
    cmd_dict["-coord"] = str(metal.coordination)
    if ligands:
        cmd_dict["-lig"] = ",".join(ligand.name for ligand in ligands)
        cmd_dict["-ligocc"] = ",".join(str(ligand.count) for ligand in ligands)
        smicats = [
            str(ligand.smicat) for ligand in ligands if ligand.smicat is not None
        ]
        if smicats:
            cmd_dict["-smicat"] = f"[{','.join(smicats)}]"
    cmd_dict["-spin"] = str(metal.spin)
    cmd_dict["-oxstate"] = str(metal.oxidation_state)

    complex_result = startgen_pythonic(
        input_dict=cmd_dict,
        write=False,
        flag=False,
    )
    if complex_result is None:
        raise ValidationError(f"molSimplify failed to generate {path}.")
    _, emsg, result = complex_result
    if emsg:
        raise ValidationError(f"molSimplify error for {path}: {emsg}")
    if result is None or result.mol is None:
        raise ValidationError(f"molSimplify returned no molecule for {path}.")
    try:
        return result.mol.writexyz(filename="buffer.xyz", writestring=True)
    except Exception as exc:  # pragma: no cover - molSimplify specifics
        raise ValidationError(f"Failed to write xyz for {path}: {exc}") from exc


def _default_metal_reactive_center(
    atoms: Atoms,
    element: str,
    path: str,
) -> ReactiveSite:
    symbols = atoms.get_chemical_symbols()
    try:
        atom_idx = symbols.index(element)
    except ValueError as exc:
        raise ValidationError(
            f"{path}: metal element '{element}' not found in generated atoms."
        ) from exc
    return ReactiveSite(atom_idx=atom_idx, site_type="metal")


def _require_mapping(data: dict, key: str) -> dict:
    if key not in data:
        raise ValidationError(f"Missing required field: {key}")
    value = data[key]
    if not isinstance(value, dict):
        raise ValidationError(f"{key} must be a mapping.")
    return value


def _require_list(data: dict, key: str) -> list:
    if key not in data:
        raise ValidationError(f"Missing required field: {key}")
    value = data[key]
    if not isinstance(value, list):
        raise ValidationError(f"{key} must be a list.")
    return value


def _require_str(data: dict, key: str, path: str) -> str:
    if key not in data:
        raise ValidationError(f"Missing required field: {path}")
    return _ensure_str(data[key], path)


def _require_int(data: dict, key: str, path: str) -> int:
    if key not in data:
        raise ValidationError(f"Missing required field: {path}")
    return _ensure_int(data[key], path)


def _optional_str(value: object, path: str) -> Optional[str]:
    if value is None:
        return None
    return _ensure_str(value, path)


def _optional_int(value: object, path: str) -> Optional[int]:
    if value is None:
        return None
    return _ensure_int(value, path)


def _optional_float(value: object, path: str) -> Optional[float]:
    if value is None:
        return None
    return _ensure_float(value, path)


def _optional_bool(value: object, path: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValidationError(f"{path} must be a boolean.")
    return value


def _optional_str_list(value: object, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be a list.")
    return [_ensure_str(item, f"{path}[{idx}]") for idx, item in enumerate(value)]


def _optional_int_list(value: object, path: str) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationError(f"{path} must be a list.")
    return [_ensure_int(item, f"{path}[{idx}]") for idx, item in enumerate(value)]


def _ensure_str(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{path} must be a string.")
    return value


def _ensure_int(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{path} must be an integer.")
    return value


def _ensure_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{path} must be a number.")
    return float(value)


def _validate_enum(value: str, allowed: set[str], path: str) -> None:
    if value not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValidationError(
            f"Invalid value for {path}: '{value}'. Allowed: {allowed_list}"
        )


def _validate_no_spaces(value: str, path: str) -> None:
    if any(ch.isspace() for ch in value):
        raise ValidationError(f"{path} must not contain spaces.")


# Example usage:
# from reaction_parser import load_reaction_from_file
#
# definition = load_reaction_from_file("reaction.yaml")
# print(definition.reaction.name)
# for reactant in definition.reactants:
#     print(reactant.id, reactant.type)
