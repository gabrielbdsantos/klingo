"""Provide tools for managing blades."""

from typing import Sequence
from profile import Profile


class Blade:
    """Define a simple, generic blade object."""

    def __init__(self, sections = Sequence[Profile]) -> None:
        self.sections = sections
