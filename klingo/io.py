# coding=utf-8
"""Provide functionalities for managing I/O operations."""

from typing import Callable, List


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
