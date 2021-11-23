"""Provide tools for managing blades."""

from profile import Profile
from typing import Sequence


class Blade:
    """Define a simple, generic blade object."""

    slots = ("sections",)

    def __init__(self, sections: Sequence[Profile]) -> None:
        self.sections = sections

    def __getitem__(self, index: int) -> Profile:
        return self.sections[index]
