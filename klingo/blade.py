# coding=utf-8
"""Provide tools for managing blade geometries."""

from io import BytesIO
from typing import Sequence

import numpy as np
import stl

from .io import generic_vertex_ordering
from .profile import Airfoil


class Blade:
    """Define a simple, generic blade object."""

    slots = ("sections",)

    def __init__(self, sections: Sequence[Airfoil]) -> None:
        self.sections = sections

    def __getitem__(self, index: int) -> Airfoil:
        return self.sections[index]

    # WARNING: currently only works for sections of same shape.
    def export_stl(
        self,
        name: str,
        file_handler: BytesIO,
        invert: bool = False,
    ) -> BytesIO:
        """Export the surface to stl.

        Parameters
        ----------
        name
            the solid name to be used as prefix.
        file_handler : optional
            the IO handler. If not specified, a new one is created and
            returned.
        invert : optional
            invert the vertices order. default = False (points outwards).

        """
        # Check whether all sections have the same shape.
        if np.any(np.diff([x.coords.shape for x in self.sections], axis=0)):
            raise ValueError(
                "all the coordinates must have the same dimensions"
            )

        # List all points as vertices.
        vertices = np.reshape(
            [section.coords for section in self.sections], (-1, 3)
        )

        # Create a list of solids
        solids = {}

        # It seems odd to attribute -1 to the number of points. However, this
        # way we manage to export an empty file even when the blade has no
        # sections. This seems a much more elegant solution than stack a bunch
        # of if-else statements.
        N = self.sections[0].coords.shape[0] if self.sections else -1

        # Lateral surface
        # --------------------------------------------------------------------
        # Create a generic representation for two consecutive sections. Then,
        # we just need to offset this representation to cover the whole blade.
        lateral_consecutive_secs = np.reshape(
            [
                generic_vertex_ordering(i, N, invert=invert)
                for i in range(N - 1)
            ],
            (-1, 3),
        )

        solids["lateral"] = np.reshape(
            [
                lateral_consecutive_secs + N * i
                for i in range(0, len(self.sections) - 1)
            ],
            (-1, 3),
        )

        # Trailing edge surface
        # --------------------------------------------------------------------
        te_consecutive_secs = np.reshape(
            generic_vertex_ordering(
                0, N, inc_i=N - 1, inc_j=N - 1, invert=invert
            ),
            (-1, 3),
        )

        solids["trailing_edge"] = np.reshape(
            [
                te_consecutive_secs + N * i
                for i in range(0, len(self.sections) - 1)
            ],
            (-1, 3),
        )

        # Top and bottom surfaces
        # --------------------------------------------------------------------
        def top_bottom_secs(invert: bool):
            return np.reshape(
                [
                    generic_vertex_ordering(
                        i,
                        N,
                        j=lambda i, N: N - i - 1,
                        inc_j=-1,
                        invert=invert,
                    )
                    for i in range(N // 2)
                ],
                (-1, 3),
            )

        solids["top"] = top_bottom_secs(False) + N * (len(self.sections) - 1)
        solids["bottom"] = top_bottom_secs(True)

        for sname, faces in solids.items():
            solid = stl.mesh.Mesh(
                np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype)
            )

            for i, face in enumerate(faces):
                for j in range(3):
                    solid.vectors[i][j] = vertices[face[j], :]

            solid.save(f"{name}_{sname}", fh=file_handler, mode=stl.Mode.ASCII)

        return file_handler
