from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ReactiveSite:
    atom_idx: int
    partner_idx: Optional[int] = None
    plane_indices: list[int] = field(default_factory=list)
    site_type: str = "generic"

    def __post_init__(self) -> None:
        if self.plane_indices is None:
            object.__setattr__(self, "plane_indices", [])
        else:
            object.__setattr__(self, "plane_indices", list(self.plane_indices))
