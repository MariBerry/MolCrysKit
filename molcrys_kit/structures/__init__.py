"""
Structural components for MolCrysKit.

This module contains the basic structural classes for representing atoms,
molecules, and crystals.
"""

from .molecule import CrystalMolecule
from .atom import MolAtom
from .crystal import MolecularCrystal
from .polyhedra import all_ideal_polyhedra, convex_hull_payload, ideal_polyhedra_for_cn

# For backward compatibility
Molecule = CrystalMolecule

__all__ = [
    "MolAtom",
    "CrystalMolecule",
    "MolecularCrystal",
    "Molecule",
    "all_ideal_polyhedra",
    "convex_hull_payload",
    "ideal_polyhedra_for_cn",
]
