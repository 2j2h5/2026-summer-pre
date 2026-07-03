import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from semantic_alignment import config as cfg
from semantic_alignment.words import WORDS
from semantic_alignment.utils.device import get_device, describe_device
from semantic_alignment.utils.seed import set_seed
from semantic_alignment.core.embedding import precompute_embeddings
from semantic_alignment.core.graph import build_graphs_with_selected_T
from semantic_alignment.experiments.temperature import select_best_T_for_all_models
from semantic_alignment.experiments.validation import run_same_G_validation
from semantic_alignment.experiments.final_ms import run_final_models, run_pairwise, fit_all_M_to_common_S
from semantic_alignment.experiments.common_s import build_common_S_from_results, evaluate_common_S_quality


def main():
    device = get_device(cfg.DEVICE)
    cfg.DEVICE = str(device)
    Path(cfg.OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("PyTorch:", torch.__version__)
    describe_device(device)
    print("N words:", len(WORDS))
    print("K:", cfg.K)
    print("Primary decomposition: G ≈ M S")

    set_seed(cfg.SEED)

    embeddings = precompute_embeddings(WORDS, cfg.MODEL_NAMES, device)
    selected_T = select_best_T_for_all_models(embeddings, cfg)
    graphs = build_graphs_with_selected_T(embeddings, selected_T)

    run_same_G_validation(graphs, cfg)

    results = run_final_models(graphs, cfg)
    run_pairwise(results, WORDS, cfg)

    S_common, aligned_results = build_common_S_from_results(results, cfg)
    evaluate_common_S_quality(S_common, aligned_results, cfg)

    fit_all_M_to_common_S(graphs, S_common, cfg)

    print(f"\nSaved to: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main()
