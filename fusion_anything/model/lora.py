#  Copyright (c) Prior Labs GmbH 2026.
"""
LoRA modules for TabPFN v2 architecture.

Applies low-rank adaptation to nn.Linear layers in TabPFNBlock:
  - AlongRowAttention: q_projection, k_projection, v_projection, out_projection
  - AlongColumnAttention: q_projection, k_projection, v_projection, out_projection
  - MLP: fc1, fc2
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA wrapper for nn.Linear: W_eff = W_frozen + lora_B @ lora_A * (alpha / rank).

    The frozen base weight is stored as a buffer so it follows .to(device) calls.
    LoRA weights are float32 for numerical stability; forward casts to input dtype.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        out_dim, in_dim = linear.weight.shape
        self.register_buffer("weight", linear.weight.detach().clone())
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the scaled low-rank update to the frozen linear projection."""
        base_out = F.linear(x, self.weight)
        lora_out = (F.linear(x, self.lora_A) @ self.lora_B.T) * self.scale
        return base_out + lora_out.to(base_out.dtype)


def _replace_with_lora(
    module: nn.Module,
    attr_name: str,
    rank: int,
    alpha: float,
) -> None:
    """Replace a Linear submodule with its LoRA-wrapped counterpart."""
    original = getattr(module, attr_name)
    if not isinstance(original, nn.Linear):
        raise TypeError(f"{attr_name} is not nn.Linear, got {type(original)}")
    setattr(module, attr_name, LoRALinear(original, rank=rank, alpha=alpha))


def apply_lora_to_tabpfn(
    model: nn.Module,
    rank: int = 32,
    alpha: float = 64.0,
    *,
    target_attn_modules: tuple[str, ...] = ("q_projection", "k_projection", "v_projection", "out_projection"),
    target_mlp_layers: tuple[int, ...] = (0, 2),
) -> nn.Module:
    """Inject LoRA into all TabPFNBlock attention and MLP layers.

    Args:
        model: TabPFNV2 instance.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_attn_modules: which projection layers to wrap inside each attention block.
        target_mlp_layers: which MLP sub-layer indices to wrap (0=fc1, 2=fc2).

    Returns:
        The same model, mutated in-place.
    """
    for block_idx, block in enumerate(model.blocks):
        # AlongRowAttention
        row_attn = block.per_sample_attention_between_features
        for name in target_attn_modules:
            _replace_with_lora(row_attn, name, rank=rank, alpha=alpha)

        # AlongColumnAttention
        col_attn = block.per_column_attention_between_cells
        for name in target_attn_modules:
            _replace_with_lora(col_attn, name, rank=rank, alpha=alpha)

        # MLP
        for layer_idx in target_mlp_layers:
            _replace_with_lora(block.mlp, str(layer_idx), rank=rank, alpha=alpha)

    return model


def get_lora_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all LoRA parameters from a model."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params
