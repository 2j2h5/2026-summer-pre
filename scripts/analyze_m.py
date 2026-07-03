import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_alignment import config as cfg
from semantic_alignment.experiments.analyze_m import analyze_M_from_pair_pts

if __name__ == "__main__":
    analyze_M_from_pair_pts(cfg.OUT_DIR)
