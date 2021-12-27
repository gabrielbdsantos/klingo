# coding=utf-8
"""Provide tools for managing blades."""

from io import BytesIO
from typing import Callable, List, Sequence

import numpy as np
import stl

from .profile import Airfoil


class Blade:
    """Define a simple, generic blade object."""

    slots = ("sections",)

    def __init__(self, sections: Sequence[Airfoil]) -> None:
        self.sections = sections

    def __getitem__(self, index: int) -> Airfoil:
        return self.sections[index]

    # WARN: currently this method only works for sections of same shape.
    def export_stl(
        self,
        name: str,
        file_handler: BytesIO,
        invert: bool = False,
    ) -> str:
        """Export the surface to stl.

        Parameters
        ----------
        name
            the solid name to be used as prefix.
        file_handler : optional
            the IO handler.
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

        solids["top"] = top_bottom_secs(True)
        solids["bottom"] = top_bottom_secs(False) + N * (
            len(self.sections) - 1
        )

        for sname, faces in solids.items():
            solid = stl.mesh.Mesh(
                np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype)
            )

            for i, face in enumerate(faces):
                for j in range(3):
                    solid.vectors[i][j] = vertices[face[j], :]

            solid.save(f"{name}_{sname}", fh=file_handler, mode=stl.Mode.ASCII)

        return file_handler.getvalue().decode()


def generic_vertex_ordering(
    i: int,
    N: int,
    j: Callable = lambda i, N: i + N,
    inc_i: int = 1,
    inc_j: int = 1,
    invert: bool = True,
) -> List:
    """Define the vertices order for each face of the STL solid.

    Faces in STL files consist of three vertices, i.e., triangles. So,
    considering two consecutive blade sections, we can imagine the
    following.

                       (i + 2*inc_i)  (i + inc_i)     i
        ith section ->  ────O────────────O────────────O────
                            .          . .          . .
                            .        .   .        .   .
                            .      .     .      .     .
                            .    .       .    .       .
                            .  .         .  .         .
                            ..           ..           .
        jth section ->  ────O────────────O────────────O────
                       (j + 2*inc_y)  (j + inc_y)     j

    This abstraction works well for blades with consistent, uniform
    sections; i.e., sections should have the same number of points.

    By default, we use clockwise ordering.

    Parameters
    ----------
    i
        the vertex index on the current section.
    N
        an offset value corresponding to the same vertex i on a next
        section.
    j
        a function describing the relationship between i and N.
    inc_i
        the increment for the next vertex in the ith section.
    inc_j
        the increment for the next vertex in the jth section.
    invert
        invert the vertices order: make them counterclockwise.

    """
    return (
        [[i, j(i, N), j(i, N) + inc_j], [i, j(i, N) + inc_j, i + inc_i]]
        if invert
        else [[i, j(i, N) + inc_j, j(i, N)], [i, i + inc_i, j(i, N) + inc_j]]
    )
