import numpy as np
from ase.atoms import Atoms


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def signed_angle_between(u, v, axis):

    u_n = unit(u)
    v_n = unit(v)
    cross = np.cross(u_n, v_n)
    sin = np.dot(cross, axis)
    cos = np.dot(u_n, v_n)
    return np.arctan2(sin, cos)


def is_approach_inside_bite(
    pre_complex: Atoms, pd_idx, approach, tolerance_deg=1.0, verbose=False
):
    """
    THINK ABOUT REMOVING THIS THING
    Return (inside:bool, info:dict)

    - pre_complex: ASE Atoms containing Pd and P donors
    - pd_idx: index of Pd within pre_complex
    - approach: 3-vector (doesn't have to be normalized)
    - tolerance_deg: small slack to avoid numerical edge-cases
    - verbose: print diagnostic info

    info contains:
      - 'bite_angle_rad', 'bite_angle_deg', 'approach_angle_rad', 'approach_angle_deg',
        'v1_index', 'v2_index', 'plane_normal'
    """
    pos = pre_complex.get_positions()
    sym = pre_complex.get_chemical_symbols()

    # find P donors and their distances (choose two nearest P if >2)
    p_indices = [i for i, s in enumerate(sym) if s == "P"]
    if len(p_indices) < 2:
        # not enough donors: treat as outside (no bite)
        return False, {"reason": "fewer_than_two_P"}

    # choose two closest P atoms to Pd (typical donors)
    dists = [(i, np.linalg.norm(pos[i] - pos[pd_idx])) for i in p_indices]
    dists.sort(key=lambda x: x[1])
    p1_idx, p2_idx = dists[0][0], dists[1][0]

    v1 = unit(pos[p1_idx] - pos[pd_idx])
    v2 = unit(pos[p2_idx] - pos[pd_idx])

    # plane normal (right-hand orientation from v1->v2)
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        # degenerate (P's collinear) — can't define bite reliably
        return False, {"reason": "collinear_Ps"}

    n = n / n_norm

    # project approach into Pd coordination plane (remove normal component)
    approach = np.array(approach, dtype=float)
    approach_proj = approach - np.dot(approach, n) * n
    ap_norm = np.linalg.norm(approach_proj)
    if ap_norm < 1e-8:
        # approach is essentially perpendicular to the plane (out-of-plane) -> treat as outside
        return False, {"reason": "approach_perpendicular_to_plane", "plane_normal": n}

    approach_proj = approach_proj / ap_norm

    # compute bite angle (0..pi)
    bite_angle = abs(signed_angle_between(v1, v2, n))
    # compute approach angle from v1 measured CCW about n, normalize to [0, 2pi)
    a = signed_angle_between(v1, approach_proj, n)
    if a < 0:
        a += 2 * np.pi

    # normalize bite to [0, 2pi)
    bite = bite_angle
    if bite < 0:
        bite += 2 * np.pi

    # inside if angle a is between 0 and bite (+ small tolerance)
    tol = np.radians(tolerance_deg)
    inside = (0 - tol) <= a <= (bite + tol)

    info = {
        "v1_index": p1_idx,
        "v2_index": p2_idx,
        "plane_normal": n,
        "bite_angle_rad": bite_angle,
        "bite_angle_deg": np.degrees(bite_angle),
        "approach_angle_rad": a,
        "approach_angle_deg": np.degrees(a),
        "approach_proj": approach_proj,
    }

    if verbose:
        print("P indices:", p1_idx, p2_idx)
        print("bite (deg):", info["bite_angle_deg"])
        print("approach angle (deg from v1):", info["approach_angle_deg"])
        print("inside:", inside)

    return inside, info
