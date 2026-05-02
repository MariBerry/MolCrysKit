"""
Packing-shell geometry analysis.

The routines here infer and score local shells from distance-ranked points and
convex-hull geometry. This is distinct from bond-graph chemical environment
analysis, which lives in :mod:`molcrys_kit.analysis.chemical_env`.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from ..structures.polyhedra import ideal_polyhedra_for_cn

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - optional dependency
    ConvexHull = None


DEFAULT_CENTROID_OFFSET_FRAC = 0.15


def _array(points: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.array(list(points), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def hull_encloses_center(
    coords: np.ndarray,
    center: np.ndarray,
    *,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
    face_tol: float = 1e-3,
) -> bool:
    """Return true when ``center`` is centred inside the hull of ``coords``.

    A packing-shell polyhedron is meaningful only when the central point sits
    roughly at the middle of the neighbour cage. The convex hull checks
    topological enclosure; the centroid-offset check rejects partial shells
    where the center is pressed against one side of the cage.
    """
    coords = np.asarray(coords, dtype=float)
    center = np.asarray(center, dtype=float)
    if len(coords) < 4 or ConvexHull is None:
        return False
    try:
        hull = ConvexHull(coords)
    except Exception:
        return False
    plane_vals = hull.equations[:, :3] @ center + hull.equations[:, 3]
    if np.any(plane_vals > face_tol):
        return False
    centroid = coords.mean(axis=0)
    radii = np.linalg.norm(coords - centroid, axis=1)
    mean_radius = float(np.mean(radii)) if len(radii) else 0.0
    if mean_radius < 1e-6:
        return True
    offset = float(np.linalg.norm(center - centroid))
    return offset <= centroid_offset_frac * mean_radius


def detect_coordination_number(
    distances: Sequence[float],
    fallback_max: int = None,
    *,
    coords: Sequence[Sequence[float]] = None,
    center: Sequence[float] = None,
    enforce_enclosure: bool = True,
    centroid_offset_frac: float = DEFAULT_CENTROID_OFFSET_FRAC,
) -> Dict[str, Any]:
    """Choose a shell coordination number from ordered neighbour distances.

    The base heuristic is the largest gap in the sorted distance list. When
    ``coords`` and ``center`` are provided, the selected shell must also
    enclose the center by convex hull and centrality checks. If the gap shell
    fails, the search expands monotonically until the shell wraps the center
    or the candidate pool is exhausted.
    """
    sorted_distances = np.sort(np.array(distances, dtype=float))
    n = len(sorted_distances)
    if n == 0:
        return {
            "coordination_number": 0,
            "gap_index": None,
            "gap_value": None,
            "enclosed": False,
            "enclosure_expanded": False,
            "primary_gap_cn": 0,
            "sorted_distances": [],
            "gaps": [],
        }
    if n == 1:
        return {
            "coordination_number": 1,
            "gap_index": 0,
            "gap_value": 0.0,
            "enclosed": False,
            "enclosure_expanded": False,
            "primary_gap_cn": 1,
            "sorted_distances": sorted_distances.tolist(),
            "gaps": [],
        }

    gaps = np.diff(sorted_distances)
    primary_cn = int(np.argmax(gaps) + 1)
    cn = primary_cn
    enclosed = False
    expanded = False

    coords_arr = np.asarray(coords, dtype=float) if coords is not None else None
    center_arr = np.asarray(center, dtype=float) if center is not None else None
    if (
        enforce_enclosure
        and coords_arr is not None
        and center_arr is not None
        and len(coords_arr) >= 4
    ):
        if hull_encloses_center(
            coords_arr[:primary_cn],
            center_arr,
            centroid_offset_frac=centroid_offset_frac,
        ):
            enclosed = True
        else:
            for candidate_cn in range(primary_cn + 1, len(coords_arr) + 1):
                if candidate_cn < 4:
                    continue
                if hull_encloses_center(
                    coords_arr[:candidate_cn],
                    center_arr,
                    centroid_offset_frac=centroid_offset_frac,
                ):
                    cn = candidate_cn
                    enclosed = True
                    expanded = True
                    break

    if fallback_max is not None:
        cn = min(cn, int(fallback_max))
    cn = max(1, cn)
    gap_index = min(cn - 1, len(gaps) - 1) if len(gaps) > 0 else None
    gap_value = float(gaps[gap_index]) if gap_index is not None else None
    return {
        "coordination_number": cn,
        "gap_index": gap_index,
        "gap_value": gap_value,
        "sorted_distances": sorted_distances.tolist(),
        "gaps": gaps.tolist(),
        "primary_gap_cn": primary_cn,
        "enclosed": enclosed,
        "enclosure_expanded": expanded,
    }


def compute_angular_signature(
    shell_coords: Iterable[Iterable[float]],
    center: Iterable[float] = None,
) -> Dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) == 0:
        return {"angles": [], "sorted_angles": [], "count": 0}
    center_vec = np.zeros(3, dtype=float) if center is None else np.array(center, dtype=float)
    vectors = coords - center_vec
    norms = np.linalg.norm(vectors, axis=1)
    angles = []
    for i, j in itertools.combinations(range(len(vectors)), 2):
        if norms[i] < 1e-8 or norms[j] < 1e-8:
            continue
        cosang = np.clip(np.dot(vectors[i], vectors[j]) / (norms[i] * norms[j]), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosang))))
    angles.sort()
    return {"angles": angles, "sorted_angles": angles, "count": len(angles)}


def angular_rmsd_vs_ideals(
    shell_coords: Iterable[Iterable[float]],
    center: Iterable[float] = None,
) -> Dict[str, Any]:
    coords = _array(shell_coords)
    cn = int(len(coords))
    signature = compute_angular_signature(coords, center=center)
    actual = np.array(signature["sorted_angles"], dtype=float)
    results = []
    for name, ideal in ideal_polyhedra_for_cn(cn).items():
        ideal_signature = np.array(compute_angular_signature(ideal)["sorted_angles"], dtype=float)
        size = min(len(actual), len(ideal_signature))
        if size == 0:
            rmsd = float("inf")
        else:
            diff = actual[:size] - ideal_signature[:size]
            rmsd = float(np.sqrt(np.mean(diff * diff)))
        results.append({"name": name, "angular_rmsd": rmsd})
    results.sort(key=lambda item: item["angular_rmsd"])
    return {
        "coordination_number": cn,
        "results": results,
        "best_match": results[0] if results else None,
    }


def planarity_analysis(
    shell_coords: Iterable[Iterable[float]],
    group_size: int = 5,
) -> Dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < group_size:
        return {"best_rms": None, "best_indices": [], "group_size": group_size}
    best_rms = float("inf")
    best_indices = None
    combo_iter = itertools.combinations(range(len(coords)), group_size)
    batch_size = 4096
    while True:
        batch = list(itertools.islice(combo_iter, batch_size))
        if not batch:
            break
        idx = np.array(batch, dtype=int)
        subsets = coords[idx]
        centered = subsets - subsets.mean(axis=1, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normals = vh[:, -1, :]
        distances = np.einsum("bgi,bi->bg", centered, normals)
        rms_values = np.sqrt(np.mean(distances * distances, axis=1))
        batch_pos = int(np.argmin(rms_values))
        rms = float(rms_values[batch_pos])
        if rms < best_rms:
            best_rms = rms
            best_indices = tuple(int(x) for x in idx[batch_pos])
    return {
        "best_rms": best_rms if best_indices is not None else None,
        "best_indices": list(best_indices or []),
        "group_size": group_size,
    }


def detect_prism_vs_antiprism(shell_coords: Iterable[Iterable[float]]) -> Dict[str, Any]:
    coords = _array(shell_coords)
    if len(coords) < 10:
        return {"classification": None, "twist_deg": None}
    z_sorted = np.argsort(coords[:, 2])
    bottom = coords[z_sorted[:5]]
    top = coords[z_sorted[-5:]]
    top_angles = np.sort(np.degrees(np.arctan2(top[:, 1], top[:, 0])) % 360.0)
    bottom_angles = np.sort(np.degrees(np.arctan2(bottom[:, 1], bottom[:, 0])) % 360.0)
    shifts = []
    for angle_top, angle_bottom in zip(top_angles, bottom_angles):
        delta = (angle_top - angle_bottom + 180.0) % 360.0 - 180.0
        shifts.append(abs(delta))
    twist = float(np.mean(shifts))
    classification = "antiprism" if twist > 18.0 else "prism"
    return {"classification": classification, "twist_deg": twist}


__all__ = [
    "DEFAULT_CENTROID_OFFSET_FRAC",
    "angular_rmsd_vs_ideals",
    "compute_angular_signature",
    "detect_coordination_number",
    "detect_prism_vs_antiprism",
    "hull_encloses_center",
    "planarity_analysis",
]
