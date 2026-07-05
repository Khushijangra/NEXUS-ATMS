"""Immutable SPGRLState dataclass definition."""

from dataclasses import dataclass
from typing import Optional
import torch

@dataclass(frozen=True)
class SPGRLState:
    As: Optional[torch.Tensor] = None
    Ab: Optional[torch.Tensor] = None
    Ft: Optional[torch.Tensor] = None
    Cf: Optional[torch.Tensor] = None
    Gt: Optional[torch.Tensor] = None
    Ct: Optional[torch.Tensor] = None
    Et: Optional[torch.Tensor] = None
