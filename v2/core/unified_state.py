import torch
from .state_types import SPGRLState
from .constants import (
    AS_DIM, AB_DIM, FT_DIM, CF_DIM, GT_DIM, CT_DIM, ET_DIM, ZT_DIM
)

class UnifiedStateBuilder:
    """The central nervous system of SPGRL. Validates, fills, and concatenates the state."""
    
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
    
    def _validate_or_fill(self, tensor: torch.Tensor | None, expected_dim: int, batch_size: int) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(batch_size, expected_dim, device=self.device)
            
        assert tensor.shape[-1] == expected_dim, f"Dimension mismatch: expected {expected_dim}, got {tensor.shape[-1]}"
        return tensor.to(self.device)

    def build(self, state: SPGRLState, batch_size: int = 1) -> torch.Tensor:
        # Validate or fill missing streams with exact dimensional requirements
        As = self._validate_or_fill(state.As, AS_DIM, batch_size)
        Ab = self._validate_or_fill(state.Ab, AB_DIM, batch_size)
        Ft = self._validate_or_fill(state.Ft, FT_DIM, batch_size)
        Cf = self._validate_or_fill(state.Cf, CF_DIM, batch_size)
        Gt = self._validate_or_fill(state.Gt, GT_DIM, batch_size)
        Ct = self._validate_or_fill(state.Ct, CT_DIM, batch_size)
        Et = self._validate_or_fill(state.Et, ET_DIM, batch_size)
        
        # Concatenate into the final 168-dimensional Zt tensor
        Zt = torch.cat([As, Ab, Ft, Cf, Gt, Ct, Et], dim=-1)
        
        # Final safety check
        assert Zt.shape[-1] == ZT_DIM, f"Zt constitution violation: expected {ZT_DIM}, got {Zt.shape[-1]}"
        
        return Zt
