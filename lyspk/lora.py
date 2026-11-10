"""LoRA for Lychee-FD's audio encoder, adaptor and Qwen2 trunk.

``LoRALinear`` wraps an existing ``nn.Linear`` (including Step-Audio-2's ``Linear``
subclass, which casts its weight to the input dtype) and adds ``scaling * B(A(x))``.
A is Kaiming-initialised and B is zero, so a freshly wrapped model computes exactly
what the frozen one did. The LoRA weights are float32; the base stays bf16.

``lora_disabled()`` switches every LoRA branch off for the duration of a block. The
trainer uses it to get the frozen model's activations from the same modules (the
preservation term), without keeping a second copy of the model.

Same convention as mpmoshi-oracle/moshi_ms/lora.py: ``scaling`` is passed directly
(mpmoshi uses 2.0), not alpha / rank.

Model deps (lazy): torch
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_STATE = {"enabled": True}


@contextlib.contextmanager
def lora_disabled():
    old = _STATE["enabled"]
    _STATE["enabled"] = False
    try:
        yield
    finally:
        _STATE["enabled"] = old


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scaling: float):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.rank, self.scaling = int(rank), float(scaling)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):          # some callers read .weight (dtype / device checks)
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if not _STATE["enabled"]:
            return y
        z = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return y + (self.scaling * z).to(y.dtype)


def wrap_linears(module: nn.Module, rank: int, scaling: float, names=None) -> int:
    """Replace every ``nn.Linear`` under ``module`` (only children whose attribute name is
    in ``names``, when given) with ``LoRALinear``. Returns how many were wrapped."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and (names is None or name in names):
            setattr(module, name, LoRALinear(child, rank, scaling))
            n += 1
        else:
            n += wrap_linears(child, rank, scaling, names)
    return n


QWEN2_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_state(module: nn.Module) -> dict:
    """Only the LoRA weights, keyed by their full parameter names."""
    return {n: p.detach().cpu() for n, p in module.named_parameters() if "lora_" in n}


def load_lora_state(module: nn.Module, state: dict) -> None:
    own = {n: p for n, p in module.named_parameters() if "lora_" in n}
    missing = sorted(set(own) - set(state))
    extra = sorted(set(state) - set(own))
    if missing or extra:
        raise RuntimeError(f"LoRA layout differs: missing {missing[:3]}... ({len(missing)}), "
                           f"unexpected {extra[:3]}... ({len(extra)})")
    with torch.no_grad():
        for n, p in own.items():
            p.copy_(state[n].to(p.device, p.dtype))


def lora_params(module: nn.Module) -> list:
    return [p for n, p in module.named_parameters() if "lora_" in n]
