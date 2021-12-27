# coding=utf-8
"""Provide different three-dimensional airfoil profiles."""

from __future__ import annotations

from typing import Sequence, Type

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class Airfoil:
    """Define a generic three-dimensional airfoil profile."""

    slots = (
        "coords",
        "camber_line",
        "chord_length",
        "max_thickness",
        "max_thickness_location",
        "reference_point",
    )

    def __init__(self) -> None:
        pass

    @classmethod
    def __init_subclass__(cls: Type[Airfoil]) -> None:
        """Initialize subclasses with the following default attributes."""
        cls.coords: NDArray[np.float64] = np.array([])
        cls.camber_line: NDArray[np.float64] = np.array([])
        cls.chord_length: float = 1
        cls.max_thickness: float = 0
        cls.max_thickness_location: NDArray[np.float64] = np.array([])
        cls.reference_point: NDArray[np.float64] = np.zeros(3)

    def scale(self, factor: float) -> None:
        """Scale the airfoil.

        Parameters
        ----------
        factor
            The scaling factor.

        """
        if factor <= 0:
            raise ValueError("Invalid scaling factor.")

        self.coords *= factor
        self.camber_line *= factor
        self.max_thickness_location *= factor
        self.chord_length *= factor

    def translate(self, vector: NDArray[np.float64]) -> None:
        """Translate the airfoil.

        Parameters
        ----------
        vector
            a three-dimensional translation vector.

        """
        vec = np.asfarray(vector)
        if vec.shape != (3,):
            raise TypeError("Invalid translation vector")

        self.coords += vec
        self.camber_line += vec
        self.max_thickness_location += vec

    def rotate(
        self,
        angles: Sequence[float],
        degrees: bool = True,
        rotate_about: NDArray[np.float64] = None,
    ) -> None:
        """Rotate the airfoil about a reference point.

        Create a three-dimensional rotation matrix based on the intrinsic
        Tait-Bryan angles (aka. yaw, pitch, and roll).

        Parameters
        ----------
        angles
            The angle values: [yaw, pitch, roll].
        degrees : optional
            Assume angle values in degrees rather than radians.
        rotate_about : optional
            The point about which to rotate.

        """
        yaw, pitch, roll = angles

        ref_point: NDArray[np.float64] = (
            self.reference_point + 0  # adding zero creates a new object.
            if rotate_about is None
            else np.asfarray(rotate_about)
        )

        def rot(mtx):
            return np.matmul(
                mtx,
                Rotation.from_euler(
                    "zyx", [-yaw, -pitch, -roll], degrees=degrees
                ).as_matrix(),
            )

        self.translate(-ref_point)

        self.coords = rot(self.coords)
        self.camber_line = rot(self.camber_line)
        self.max_thickness_location = rot(self.max_thickness_location)

        self.translate(ref_point)

        def __array__(self) -> NDArray[np.float64]:
            """Enable numpy operations on Airfoil."""
            return self.coords


class NACA4(Airfoil):
    """Define a three-dimensional four-digit NACA airfoil profile."""

    def __init__(
        self,
        max_camber: float,
        max_camber_position: float,
        max_thickness: float,
        chord_length: float = 1.0,
        npoints: int = 100,
        cosine_spacing: bool = True,
        open_trailing_edge: bool = True,
    ) -> None:
        """Generate the coordiantes for a four-digit NACA profile."""
        # Rename the arguments according to the NACA standard.
        m = max_camber
        p = max_camber_position
        t = max_thickness

        # The maximum camber (m) is given by the first digit of the four
        # digits. The value ranges from 0 to 9, which yields a maximum camber
        # from 0% to 9% of the chord length.
        if m < 0 or m >= 0.1:
            raise RuntimeError(
                "The maximum camber should be in the range [0, 0.1)."
            )

        # Similarly, the maximum camber position (p) is given by the second
        # digit. Again, the value ranges from 0 to 9, which yields a maximum
        # camber position from 0% to 90% of the chord length.
        if p < 0 or p >= 1:
            raise RuntimeError(
                "The maximum camber position should be in the range [0, 1)."
            )

        # Finally, the maximum thickness (t) is given by the last two digits.
        # The value ranges from 0 to 99, which yields a maximum camber position
        # from 0% to 99% of the chord length.
        if t <= 0 or t >= 1:
            raise RuntimeError(
                "The maximum thickness should be in the range (0, 1)."
            )

        # Choose whether to use linear or cosine spacing. By default, cosine
        # spacing will be used.
        if cosine_spacing:
            xc = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, npoints)))
        else:
            xc = np.linspace(0, 1, npoints)

        # Calculate the half thickness (yt).
        # Note that using a4 = -0.1015 returns a non-zero thickness at x/c = 1.
        # If a zero-thickness trailing edge is required, then we use a4 =
        # -0.1036, which results in the smallest change to the overall shape of
        # the airfoil.
        a0 = 0.2969
        a1 = -0.1260
        a2 = -0.3516
        a3 = 0.2843
        a4 = -0.1015 if open_trailing_edge else -0.1036
        yt = (
            5
            * t
            * (
                a0 * np.sqrt(xc)
                + a1 * xc
                + a2 * xc ** 2
                + a3 * xc ** 3
                + a4 * xc ** 4
            )
        )

        # Get the position of maximum thickness
        yt_max = yt.max()
        xc_max = xc[np.where(yt == yt_max)]

        # Calculate the camber coordinates corresponding to the maximum
        # thickness of the airfoil.
        if p != 0:
            if xc_max <= p:
                yc_max = m / (p * p) * (2 * p * xc_max - xc_max * xc_max)
            else:
                yc_max = (
                    m
                    / ((1 - p) * (1 - p))
                    * ((1 - 2 * p) + 2 * p * xc_max - xc_max * xc_max)
                )
        else:
            yc_max = [0]

        # Initialize the array for the camber line.
        yc = np.zeros(xc.shape)

        # Calculate both upper and lower airfoil surfaces for cambered
        # airfoils.
        if p != 0:
            # Initialize the array for the camber line derivative.
            dydx = np.zeros(xc.shape)

            i = np.where(xc <= p)
            yc[i] = m / (p * p) * (2 * p * xc[i] - xc[i] * xc[i])
            dydx[i] = 2 * m / (p * p) * (p - xc[i])

            i = np.where(xc > p)
            yc[i] = (
                m
                / ((1 - p) * (1 - p))
                * ((1 - 2 * p) + 2 * p * xc[i] - xc[i] * xc[i])
            )
            dydx[i] = 2 * m / ((1 - p) * (1 - p)) * (p - xc[i])

            theta = np.arctan(dydx)

            xu = xc - yt * np.sin(theta)
            yu = yc + yt * np.cos(theta)
            xl = xc + yt * np.sin(theta)
            yl = yc - yt * np.cos(theta)

        # Calculate both upper and lower airfoil surfaces for non-cambered
        # airfoils.
        else:
            xu = xl = xc
            yu = yt
            yl = -yt

        # Set the lateral coordinates as a continuous array.
        x = np.hstack((xu[::-1], xl[1:]))
        y = np.hstack((yu[::-1], yl[1:]))

        # Assign values
        self.max_thickness = 2 * yt_max
        self.max_thickness_location = np.array([*xc_max, *yc_max, 0])

        def recenter(array: np.ndarray) -> np.ndarray:
            return array - self.reference_point  # type: ignore

        self.camber_line = recenter(
            np.asfarray([xc, yc, np.zeros(xc.shape)]).T
        )
        self.coords = recenter(np.asfarray([x, y, np.zeros(x.shape)]).T)

        self.scale(chord_length)
