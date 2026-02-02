from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

from ase.atoms import Atoms
from ase.data import covalent_radii

from utils.dtxb_calculator import DXTBCalculator
from utils.bite_angle import is_approach_inside_bite
from utils.reactive_site import ReactiveSite


# ==========================
# Basic vector utilities
# because we dont trust git :)
# ==========================


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v.copy()


def rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = unit(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array(
        [
            [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
        ]
    )
    return R


def rotate_points(
    pts: np.ndarray,
    axis_point: np.ndarray,
    axis_vec: np.ndarray,
    angle_deg: float,
    indices: Optional[List[int]] = None,
) -> None:
    """Rotate pts in-place. If indices provided, only rotate those rows."""
    if indices is None:
        idxs = range(len(pts))
    else:
        idxs = indices
    R = rotation_matrix(axis_vec, np.radians(angle_deg))
    for i in idxs:
        pts[i] = axis_point + R @ (pts[i] - axis_point)


# ==========================
# Approach generators
# ==========================


class ApproachGenerator:
    def generate(self, atoms: Atoms, site: ReactiveSite) -> List[np.ndarray]:
        raise NotImplementedError


class BondSigmaStarApproach(ApproachGenerator):
    """
    Approach along the sigma* direction of a bond: opposite of (A -> B) vector.
    For a site defined by (atom_idx = A, partner_idx = B), we take:

        v_bond = pos[B] - pos[A]
        sigma* direction ~ -unit(v_bond)
    """

    def generate(self, atoms: Atoms, site: ReactiveSite) -> List[np.ndarray]:
        if site.partner_idx is None:
            raise ValueError("BondSigmaStarApproach requires site.partner_idx")
        pos = atoms.get_positions()
        v = pos[site.partner_idx] - pos[site.atom_idx]  # A -> B
        return [unit(-v)]  # opposite direction (toward sigma*)


class PiFaceApproach(ApproachGenerator):
    """
    Approach normal to a local plane defined by site.plane_indices (e.g. aromatic ring).

    We generate ± normal vectors.
    """

    def generate(self, atoms: Atoms, site: ReactiveSite) -> List[np.ndarray]:
        if not site.plane_indices:
            raise ValueError("PiFaceApproach requires site.plane_indices to be set")
        pos = atoms.get_positions()
        pts = pos[site.plane_indices]
        centroid = np.mean(pts, axis=0)
        coords = pts - centroid
        # PCA: last singular vector = normal to best-fit plane
        _, _, vt = np.linalg.svd(coords)
        normal = vt[-1]
        # orient relative to central atom for consistency
        if np.dot(normal, pos[site.atom_idx] - centroid) < 0:
            normal = -normal
        return [unit(normal), unit(-normal)]


class MixedApproach(ApproachGenerator):
    """
    Linear mixtures between two other generators (e.g. sigma* and pi-face).

    weights in [0,1] interpolates:  (1-w)*v_a + w*v_b
    """

    def __init__(
        self,
        gen_a: ApproachGenerator,
        gen_b: ApproachGenerator,
        weights: np.ndarray,
    ):
        self.gen_a = gen_a
        self.gen_b = gen_b
        self.weights = np.asarray(weights, dtype=float)

    def generate(self, atoms: Atoms, site: ReactiveSite) -> List[np.ndarray]:
        va = self.gen_a.generate(atoms, site)
        vb = self.gen_b.generate(atoms, site)
        if not va or not vb:
            return []
        a = va[0]
        b = vb[0]
        out: List[np.ndarray] = []
        for w in self.weights:
            raw = (1.0 - w) * unit(a) + w * unit(b)
            nm = np.linalg.norm(raw)
            if nm > 1e-8:
                out.append(unit(raw))
        return out


class RandomApproach(ApproachGenerator):
    """
    Generic random approach directions – covers unknown mechanisms and fallback.
    """

    def __init__(self, n: int = 16, seed: Optional[int] = None):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def generate(self, atoms: Atoms, site: ReactiveSite) -> List[np.ndarray]:
        vs = self.rng.normal(size=(self.n, 3))
        return [unit(v) for v in vs]


# ==========================
# Scorers
# ==========================


class ClashScorer:
    """Vectorized clash scoring using covalent radii (fully general)."""

    def __init__(self, scale_threshold: float = 0.8):
        self.scale = scale_threshold

    def score(
        self,
        frag_pos: np.ndarray,
        frag_numbers: np.ndarray,
        sub_pos: np.ndarray,
        sub_numbers: np.ndarray,
    ) -> Tuple[float, float, int]:
        """
        Returns: (total_score, max_overlap, n_overlaps)
        score is normalized overlap penalty; lower is better.
        """
        D = np.linalg.norm(
            frag_pos[:, None, :] - sub_pos[None, :, :],
            axis=-1,
        )
        r_frag = covalent_radii[frag_numbers]
        r_sub = covalent_radii[sub_numbers]
        Rsum = r_frag[:, None] + r_sub[None, :]

        threshold = self.scale * Rsum
        mask = D < threshold
        if not mask.any():
            return 0.0, 0.0, 0

        norm_pen = (threshold - D) / threshold
        norm_pen[~mask] = 0.0

        score = float(np.sum(norm_pen))
        max_overlap = float(np.max(np.where(mask, threshold - D, 0.0)))
        n_overlaps = int(np.sum(mask))
        return score, max_overlap, n_overlaps


class EnergyScorer:
    """
    Optional single-point energy annotation using DXTBCalculator.
    Not used for ranking by default.
    """

    def __init__(self, method: str = "GFN1", spin: Optional[int] = None):
        self.method = method
        self.spin = spin

    def single_point(self, atoms: Atoms) -> float:
        atoms = atoms.copy()
        calc = DXTBCalculator(method=self.method, spin=self.spin)
        atoms.calc = calc
        try:
            e = atoms.get_potential_energy()
        except Exception as exc:
            print("DXTB single-point failed:", exc)
            e = float("inf")
        return float(e)


# ==========================
# Graph helpers for local rotations
# ==========================


def infer_bonds_simple(
    pos: np.ndarray,
    numbers: np.ndarray,
    scale: float = 1.2,
) -> List[Tuple[int, int]]:
    """
    Crude distance-based bond inference to define a connectivity graph.
    Good enough for selecting local subgraphs to rotate.
    """
    n = len(pos)
    bonds: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ri = covalent_radii[numbers[i]]
            rj = covalent_radii[numbers[j]]
            cutoff = scale * (ri + rj)
            if np.linalg.norm(pos[i] - pos[j]) < cutoff:
                bonds.append((i, j))
    return bonds


def bfs_subgraph(
    bonds: List[Tuple[int, int]],
    start_idx: int,
    max_depth: int = 3,
) -> List[int]:
    """Return atoms within graph distance <= max_depth from start_idx."""
    neighbors = {i: [] for bp in bonds for i in bp}
    for i, j in bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)

    visited = {start_idx}
    frontier = [(start_idx, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for neigh in neighbors.get(current, []):
            if neigh not in visited:
                visited.add(neigh)
                frontier.append((neigh, depth + 1))
    return sorted(visited)


def alignment_score(
    at_pos: np.ndarray,
    siteA_idx: int,
    fragB_pos: np.ndarray,
    siteB_idx: int,
    approach_vec: np.ndarray,
) -> float:
    """
    Geometry-only alignment measure:
    We want direction B_site -> A_site to match approach_vec.

    Score ~0 is perfect, larger is worse (max 2).
    """
    pa = at_pos[siteA_idx]
    pb = fragB_pos[siteB_idx]
    actual = unit(pa - pb)
    ideal = unit(approach_vec)
    dot = float(np.clip(np.dot(actual, ideal), -1.0, 1.0))
    return 1.0 - dot


# ==========================
# Core sampler (3-body aware)
# ==========================


class PrecomplexSampler:
    """
    Samples approach vectors and distances, aligns fragment A (metal, etc.) to fragment B
    (substrate), prunes by clash + alignment, and optionally annotates energies.

    Fully geometry-based, aware of 3-body reactive core via ReactiveSite.partner_idx
    and ReactiveSite.plane_indices (but not reaction-type specific).
    """

    def __init__(
        self,
        fragA: Atoms,
        fragB: Atoms,
        siteA: ReactiveSite,
        siteB: ReactiveSite,
        approach_gens: List[ApproachGenerator],
        distances: np.ndarray,
        clash_scorer: Optional[ClashScorer] = None,
        energy_scorer: Optional[EnergyScorer] = None,
    ):
        self.fragA0 = fragA.copy()
        self.fragB0 = fragB.copy()
        self.siteA = siteA
        self.siteB = siteB
        self.approach_gens = approach_gens
        self.distances = distances
        self.clash_scorer = clash_scorer or ClashScorer()
        self.energy_scorer = energy_scorer

    def translate_to_target(
        self, posA: np.ndarray, idxA: int, target_point: np.ndarray
    ) -> None:
        trans = target_point - posA[idxA]
        posA += trans

    def align_vector(
        self,
        posA: np.ndarray,
        idxA: int,
        target_vec: np.ndarray,
        ref_point: np.ndarray,
    ) -> None:
        """
        Rotate posA about idxA so that vec(idxA->ref_point) aligns with target_vec.
        """
        cur_vec = ref_point - posA[idxA]
        n_cur = np.linalg.norm(cur_vec)
        if n_cur < 1e-8:
            return
        cur_n = cur_vec / n_cur
        tgt_n = unit(target_vec)

        cross = np.cross(cur_n, tgt_n)
        cross_norm = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(cur_n, tgt_n), -1.0, 1.0))

        if cross_norm < 1e-8:
            # parallel or antiparallel
            if dot < 0.0:
                axis = np.cross(cur_n, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(axis) < 1e-6:
                    axis = np.cross(cur_n, np.array([0.0, 1.0, 0.0]))
                rotate_points(posA, posA[idxA], axis, 180.0)
            return

        axis = cross / cross_norm
        angle_deg = np.degrees(np.arccos(dot))
        rotate_points(posA, posA[idxA], axis, angle_deg)

    def evaluate_candidate(self, at_pos: np.ndarray) -> Tuple[float, float, int]:
        fragA_nums = self.fragA0.get_atomic_numbers()
        fragB_nums = self.fragB0.get_atomic_numbers()
        return self.clash_scorer.score(
            at_pos,
            fragA_nums,
            self.fragB0.get_positions(),
            fragB_nums,
        )

    def sample(
        self,
        max_coarse_keep: int = 50,
        do_energy: bool = True,
        max_energy_eval: int = 8,
    ):
        """
        Returns a list of candidate dicts:
          - 'at_pos': positions for fragA
          - 'approach': approach vector
          - 'distance': float
          - 'clash_score', 'max_overlap', 'n_overlaps'
          - 'alignment_score'
          - 'inside_bite'
          - optional 'energy'
        """
        fragB_pos = self.fragB0.get_positions()
        fragA_pos0 = self.fragA0.get_positions().copy()
        fragA_nums = self.fragA0.get_atomic_numbers()

        candidates = []

        # Collect approach vectors from all generators
        approaches: List[np.ndarray] = []
        for gen in self.approach_gens:
            approaches.extend(gen.generate(self.fragB0, self.siteB))

        def unique_vectors(
            vecs: List[np.ndarray], tol: float = 1e-3
        ) -> List[np.ndarray]:
            out: List[np.ndarray] = []
            for v in vecs:
                if not any(np.linalg.norm(v - u) < tol for u in out):
                    out.append(v)
            return out

        approaches = unique_vectors(approaches)

        for approach in approaches:
            inside, _ = is_approach_inside_bite(
                self.fragA0, self.siteA.atom_idx, approach
            )

            for d in self.distances:
                at_pos = fragA_pos0.copy()

                # place siteA at siteB + approach * d
                target = fragB_pos[self.siteB.atom_idx] + approach * d
                self.translate_to_target(at_pos, self.siteA.atom_idx, target)

                # align A_site -> B_site with approach direction
                self.align_vector(
                    at_pos,
                    self.siteA.atom_idx,
                    approach,
                    fragB_pos[self.siteB.atom_idx],
                )

                clash_score, max_ov, n_ov = self.evaluate_candidate(at_pos)
                align_s = alignment_score(
                    at_pos,
                    self.siteA.atom_idx,
                    fragB_pos,
                    self.siteB.atom_idx,
                    approach,
                )

                candidates.append(
                    {
                        "at_pos": at_pos.copy(),
                        "approach": approach.copy(),
                        "distance": float(d),
                        "clash_score": float(clash_score),
                        "max_overlap": float(max_ov),
                        "n_overlaps": int(n_ov),
                        "alignment_score": float(align_s),
                        "inside_bite": bool(inside),
                    }
                )

        # Coarse pruning by clash + alignment
        candidates.sort(key=lambda c: (c["clash_score"], c["alignment_score"]))
        coarse = candidates[:max_coarse_keep]

        # Local refinement: rotate a local subgraph around siteA to relieve clashes
        refined = []
        bonds = infer_bonds_simple(fragA_pos0, fragA_nums, scale=1.2)
        local_cluster = bfs_subgraph(bonds, self.siteA.atom_idx, max_depth=3)

        for cand in coarse:
            at_pos = cand["at_pos"].copy()
            best_score = cand["clash_score"]
            best_pos = at_pos.copy()

            if cand["clash_score"] > 1e-4:
                axis_origin = at_pos[self.siteA.atom_idx]
                axis_vec = cand["approach"]
                if np.linalg.norm(axis_vec) > 1e-8:
                    for angle in np.arange(-60.0, 61.0, 15.0):
                        tmp = at_pos.copy()
                        rotate_points(
                            tmp, axis_origin, axis_vec, angle, indices=local_cluster
                        )
                        score, _, _ = self.evaluate_candidate(tmp)
                        if score < best_score:
                            best_score = score
                            best_pos = tmp.copy()

            final_align = alignment_score(
                best_pos,
                self.siteA.atom_idx,
                fragB_pos,
                self.siteB.atom_idx,
                cand["approach"],
            )

            cand["clash_score"] = float(best_score)
            cand["alignment_score"] = float(final_align)
            cand["at_pos"] = best_pos
            refined.append(cand)

        refined.sort(key=lambda c: (c["clash_score"], c["alignment_score"]))
        final = refined[:max_coarse_keep]

        # Optional energy annotation (not used for ranking)
        if self.energy_scorer and do_energy:
            n_eval = min(max_energy_eval, len(final))
            for i in range(n_eval):
                c = final[i]
                fragA_copy = self.fragA0.copy()
                fragA_copy.set_positions(c["at_pos"])
                merged = fragA_copy + self.fragB0
                e = self.energy_scorer.single_point(merged)
                c["energy"] = e

        return final


# ==========================
# High-level Builder (3-body aware)
# ==========================


class PrecomplexBuilder:
    """
    General, 3-body–aware precomplex builder.

    - siteA: typically the metal reactive atom
    - siteB: central atom of a reactive bond (with partner_idx for the other atom)
    - plane_indices on siteB: ring/π-system definition if available

    If no approach_generators are added manually, we automatically create:
      - BondSigmaStarApproach (if partner_idx is set)
      - PiFaceApproach (if plane_indices is non-empty)
      - MixedApproach between them (if both exist)
      - RandomApproach as a fallback
    """

    def __init__(
        self,
        fragA: Atoms,
        fragB: Atoms,
        siteA: ReactiveSite,
        siteB: ReactiveSite,
        distances: Optional[np.ndarray] = None,
    ):
        self.fragA = fragA.copy()
        self.fragB = fragB.copy()
        self.siteA = siteA
        self.siteB = siteB

        if distances is not None:
            self.distances = np.asarray(distances, dtype=float)
        else:
            # universal near-contact region for TS-like precomplexes
            self.distances = np.linspace(2.7, 3.1, 5)

        self.approach_generators: List[ApproachGenerator] = []
        self.clash_scorer = ClashScorer()
        self.energy_scorer: Optional[EnergyScorer] = None

    def add_approach_generator(self, gen: ApproachGenerator) -> None:
        self.approach_generators.append(gen)

    def set_distances(self, distances: np.ndarray) -> None:
        self.distances = np.asarray(distances, dtype=float)

    def set_energy_scorer(self, scorer: EnergyScorer) -> None:
        self.energy_scorer = scorer

    def _ensure_default_generators(self) -> None:
        """If user did not supply any generators, create sensible defaults."""
        if self.approach_generators:
            return

        gens: List[ApproachGenerator] = []

        # sigma* if bond partner known
        sigma_gen: Optional[BondSigmaStarApproach] = None
        pi_gen: Optional[PiFaceApproach] = None

        if self.siteB.partner_idx is not None:
            sigma_gen = BondSigmaStarApproach()

            # slight trick: we wrap it with a small class that forwards siteB.partner_idx
            class _SigmaWrapper(BondSigmaStarApproach):
                def generate(
                    self, atoms: Atoms, site: ReactiveSite
                ) -> List[np.ndarray]:
                    if site.partner_idx is None:
                        raise ValueError(
                            "ReactiveSite.partner_idx must be set for sigma*"
                        )
                    pos = atoms.get_positions()
                    v = pos[site.partner_idx] - pos[site.atom_idx]
                    return [unit(-v)]

            sigma_gen = _SigmaWrapper()
            gens.append(sigma_gen)

        # pi-face if ring / plane atoms known
        if self.siteB.plane_indices:
            pi_gen = PiFaceApproach()
            gens.append(pi_gen)

        # mixed if both sigma* and pi-face exist
        if sigma_gen is not None and pi_gen is not None:
            mixed = MixedApproach(
                sigma_gen,
                pi_gen,
                weights=np.linspace(0.1, 0.9, 9),
            )
            gens.append(mixed)

        # always include random fallback
        gens.append(RandomApproach(n=16, seed=42))

        self.approach_generators = gens

    def build(
        self,
        max_keep: int = 50,
        do_energy: bool = True,
        max_energy_eval: int = 8,
    ):
        self._ensure_default_generators()

        sampler = PrecomplexSampler(
            fragA=self.fragA,
            fragB=self.fragB,
            siteA=self.siteA,
            siteB=self.siteB,
            approach_gens=self.approach_generators,
            distances=self.distances,
            clash_scorer=self.clash_scorer,
            energy_scorer=self.energy_scorer,
        )
        return sampler.sample(
            max_coarse_keep=max_keep,
            do_energy=do_energy,
            max_energy_eval=max_energy_eval,
        )
