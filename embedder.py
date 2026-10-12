"""CNN embedding model (ResNet-18 backbone) + batch inference wrapper."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

log = logging.getLogger(__name__)


class SignatureNet(nn.Module):
    def __init__(self, dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, dim))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:    # x: (B,1,H,W) in 0..1, ink = 1
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return F.normalize(self.head(self.backbone(x)), dim=1)


class Embedder:
    def __init__(self, weights: str | Path | None, dim: int = 256, device: str = "cuda"):
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        has_ckpt = bool(weights) and Path(weights).exists()
        self.net = SignatureNet(dim, pretrained=not has_ckpt)
        if has_ckpt:
            ckpt = torch.load(weights, map_location="cpu")
            self.net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        else:
            log.warning("No fine-tuned embedder weights found - using generic ImageNet features. "
                        "Accuracy against skilled forgeries will be POOR until you run "
                        "training/train_embedder.py")
        self.net.to(self.device).eval()
        self.dim = dim

    @torch.inference_mode()
    def embed(self, canvases: list[np.ndarray], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(canvases), batch):
            x = torch.from_numpy(np.stack(canvases[i:i + batch]))[:, None].to(self.device)
            with torch.autocast(device_type=self.device.split(":")[0], enabled=self.device != "cpu"):
                z = self.net(x)
            out.append(F.normalize(z.float(), dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)
