"""Readout: the one host-specific part of the speaker head (md section 1).

mpmoshi's ``P1cReadout`` mixes the first 16 layers of one stack. Lychee has two stacks
at different widths that both reach 12.5 Hz:

  enc   the audio encoder: input of block 1 + all 32 block outputs, 1280-d, 50 Hz
        averaged 4 -> 1 to 12.5 Hz                                    [33, T, 1280]
  llm   the Qwen2 trunk at the <audio_patch> slots: the input embedding (adaptor
        output + channel embeddings) + the outputs of layers 1..L, 3584-d  [L+1, T, 3584]

Each group keeps P1cReadout's recipe on its own: softmax-weighted mix (uniform at
init), LayerNorm, then ``Linear -> 384`` (diar) and ``Linear -> 256`` (con). The groups'
projections are summed, so the head sees one 384-d frame either way, and the learned
logits say which depth of which stack the head uses. The layer probe found the voice
in the encoder's early-middle blocks and little of it in the trunk, so the encoder
group is the one that matters most; the trunk group is there because the LoRA can
change that.

Model deps (lazy): torch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupReadout(nn.Module):
    def __init__(self, groups: dict, diar_dim: int = 384, con_dim: int = 256):
        """``groups``: ``{name: (n_layers, dim)}``."""
        super().__init__()
        if not groups:
            raise ValueError("readout needs at least one group")
        self.groups = {k: (int(n), int(d)) for k, (n, d) in groups.items()}
        self.logits = nn.ParameterDict({k: nn.Parameter(torch.zeros(n)) for k, (n, _) in self.groups.items()})
        self.norm = nn.ModuleDict({k: nn.LayerNorm(d) for k, (_, d) in self.groups.items()})
        self.diar = nn.ModuleDict({k: nn.Linear(d, diar_dim) for k, (_, d) in self.groups.items()})
        self.con = nn.ModuleDict({k: nn.Linear(d, con_dim) for k, (_, d) in self.groups.items()})

    def config(self) -> dict:
        return {"groups": {k: list(v) for k, v in self.groups.items()},
                "diar_dim": next(iter(self.diar.values())).out_features,
                "con_dim": next(iter(self.con.values())).out_features}

    def mix(self, name: str, stack: torch.Tensor) -> torch.Tensor:
        """``stack`` [L, T, D] -> [T, D] float32."""
        n = self.groups[name][0]
        if stack.ndim != 3 or int(stack.shape[0]) != n:
            raise ValueError(f"group {name}: expected [{n}, T, D], got {tuple(stack.shape)}")
        w = F.softmax(self.logits[name].float(), dim=0)
        out = None
        for i in range(n):          # layer by layer: no float32 copy of the whole stack
            term = w[i] * stack[i].float()
            out = term if out is None else out + term
        return out

    def forward(self, stacks: dict):
        diar = con = None
        for name in self.groups:
            h = self.norm[name](self.mix(name, stacks[name]))
            d, c = self.diar[name](h), self.con[name](h)
            diar = d if diar is None else diar + d
            con = c if con is None else con + c
        return diar, F.normalize(con, dim=-1)

    def layer_weights(self) -> dict:
        return {k: F.softmax(v.detach().float(), 0).cpu().tolist() for k, v in self.logits.items()}
