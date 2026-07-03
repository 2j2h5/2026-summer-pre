import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from semantic_alignment import config as cfg
from semantic_alignment.utils.device import get_device, describe_device

device = get_device(cfg.DEVICE)
describe_device(device)

x = torch.randn(1024, 1024, device=device)
y = x @ x.T
print("Tensor device:", y.device)
print("OK")
