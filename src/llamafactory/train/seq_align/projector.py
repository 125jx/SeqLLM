"""
BehaviorProjector -- residual MLP that refines behavior-token embeddings.

    output = input + MLP(input)

Last linear layer is zero-initialized so the projector starts as identity.
"""

import json
import os
from typing import Optional

import torch
import torch.nn as nn

from ...extras.logging import get_logger

logger = get_logger(__name__)

_CONFIG_NAME = "projector_config.json"
_WEIGHTS_NAME = "projector.pt"


class BehaviorProjector(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(dim, dim))
        else:
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(self.hidden_dim, dim))

        self.mlp = nn.Sequential(*layers)
        self._zero_init_last_layer()

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"BehaviorProjector: dim={dim}, hidden={self.hidden_dim}, "
                    f"layers={num_layers}, params={total:,}")

    # ------------------------------------------------------------------
    def _zero_init_last_layer(self):
        """Zero-init the last Linear so residual starts as identity."""
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save_pretrained(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, _WEIGHTS_NAME))
        cfg = {"dim": self.dim, "hidden_dim": self.hidden_dim, "num_layers": self.num_layers}
        with open(os.path.join(save_dir, _CONFIG_NAME), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_dir: str, **overrides) -> "BehaviorProjector":
        with open(os.path.join(load_dir, _CONFIG_NAME)) as f:
            cfg = json.load(f)
        cfg.update(overrides)
        proj = cls(**cfg)
        state = torch.load(os.path.join(load_dir, _WEIGHTS_NAME), map_location="cpu")
        proj.load_state_dict(state)
        logger.info(f"BehaviorProjector loaded from {load_dir}")
        return proj
