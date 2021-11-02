"""Define some wrappers for working with OCC."""

from os import PathLike
from typing import Sequence

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge,
                                     BRepBuilderAPI_MakeSolid,
                                     BRepBuilderAPI_MakeWire,
                                     BRepBuilderAPI_Sewing)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.GeomAPI import GeomAPI_Interpolate
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_HArray1OfPnt
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Shape, topods_Shell
from OCC.Extend.DataExchange import write_stl_file

from klingo.profile import Profile


def sections_to_surface(
    sections: Sequence[Profile],
) -> BRepOffsetAPI_ThruSections:
    """Create a surface out of a sequence of profiles."""
    surface = BRepOffsetAPI_ThruSections(False, False, 1e-18)
    surface.SetMaxDegree(1)

    for section in sections:
        vertices = TColgp_HArray1OfPnt(1, section.coords.shape[0])

        for i, coord in enumerate(section.coords):
            vertices.SetValue(i + 1, gp_Pnt(*coord))

        bspline = GeomAPI_Interpolate(vertices, False, 1e-12)
        bspline.Perform()

        edge = BRepBuilderAPI_MakeEdge(bspline.Curve()).Edge()

        surface.AddWire(BRepBuilderAPI_MakeWire(edge).Wire())

    surface.Build()

    return surface


def surfaces_to_solid(
    surfaces: Sequence[BRepOffsetAPI_ThruSections],
) -> BRepBuilderAPI_MakeSolid:
    """Create a solid out of connected surfaces."""
    sewer = BRepBuilderAPI_Sewing(1e-6)
    for surface in surfaces:
        sewer.Add(surface.Shape())

    sewer.Perform()

    solid = BRepBuilderAPI_MakeSolid()
    solid.Add(topods_Shell(sewer.SewedShape()))

    return solid


def make_compound(surfaces: Sequence) -> TopoDS_Compound:
    """Create a compound out of disconnected surfaces."""
    builder = BRep_Builder()
    compound = TopoDS_Compound()

    builder.MakeCompound(compound)
    for surface in surfaces:
        builder.Add(compound, surface.Shape())

    return compound


def export_stl(
    shape: TopoDS_Shape, filename: PathLike, mode: str = "ascii"
) -> None:
    """Export a given shape to STL."""
    write_stl_file(
        shape,
        filename,
        mode=mode,
        linear_deflection=1e-4,
        angular_deflection=0.03,
    )
