"""Tests for packing-shell polyhedra analysis."""

import numpy as np

from molcrys_kit.analysis.packing_shell import (
    angular_rmsd_vs_ideals,
    detect_coordination_number,
    hull_encloses_center,
)
from molcrys_kit.structures.polyhedra import convex_hull_payload, ideal_polyhedra_for_cn


def test_ideal_polyhedra_catalog_exposes_cn8_cube():
    cn8 = ideal_polyhedra_for_cn(8)

    assert "cube" in cn8
    assert cn8["cube"].shape == (8, 3)
    np.testing.assert_allclose(np.linalg.norm(cn8["cube"], axis=1), np.ones(8))


def test_convex_hull_payload_serializes_faces_and_edges():
    coords = ideal_polyhedra_for_cn(8)["cube"]
    payload = convex_hull_payload(coords)

    assert len(payload["vertices"]) == 8
    assert len(payload["simplices"]) > 0
    assert len(payload["edges"]) > 0


def test_hull_encloses_center_for_symmetric_shell():
    coords = ideal_polyhedra_for_cn(8)["cube"]

    assert hull_encloses_center(coords, np.zeros(3)) is True


def test_coordination_number_expands_until_centered_inside_hull():
    coords = np.array(
        [
            [-4.04, 1.06, 2.40],
            [4.08, 1.07, 2.43],
            [0.00, -4.61, 2.52],
            [0.00, 4.80, -2.62],
            [-3.97, 1.06, -7.76],
            [4.07, 1.06, -7.79],
            [0.00, -4.57, -7.71],
            [-4.04, -7.53, -2.60],
            [4.04, -7.53, -2.60],
            [0.00, 4.76, 7.63],
            [-4.10, 7.74, 2.46],
            [4.10, 7.74, 2.46],
        ],
        dtype=float,
    )
    distances = np.linalg.norm(coords, axis=1)

    result = detect_coordination_number(
        distances,
        coords=coords,
        center=[0.0, 0.0, 0.0],
        enforce_enclosure=True,
    )

    assert result["primary_gap_cn"] == 4
    assert result["coordination_number"] == 12
    assert result["enclosed"] is True
    assert result["enclosure_expanded"] is True


def test_angular_rmsd_matches_ideal_cube():
    coords = ideal_polyhedra_for_cn(8)["cube"]
    result = angular_rmsd_vs_ideals(coords)

    assert result["coordination_number"] == 8
    assert result["best_match"]["name"] == "cube"
    assert result["best_match"]["angular_rmsd"] < 1e-10
