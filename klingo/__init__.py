# coding: utf-8
"""A minimal framework for creating parameterized airfoil geometries."""

__version__ = "0.1.0-alpha"

from .blade import Blade
from .occ import export_stl, make_compound, make_solid, make_surface
from .profile import NACA4
