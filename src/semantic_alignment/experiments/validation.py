import os
import pandas as pd
import numpy as np

from semantic_alignment.experiments.temperature import factorize_model_at_T
from semantic_alignment.core.metrics import align_S, apply_perm_to_S, js_divergence_columns
from semantic_alignment.utils.names import safe_model_name


def run_same_G_validation(graphs, cfg):
    summaries = []

    for model_name in cfg.MODEL_NAMES:
        E = graphs[model_name]["E"]
        T = graphs[model_name]["relation_temperature"]

        print("\n" + "=" * 100)
        print(f"Final same-G validation | {model_name} | T={T}")
        print("=" * 100)

        rows = []
        results = [
            factorize_model_at_T(model_name, E, T, seed, cfg)
            for seed in cfg.FINAL_SEED_STABILITY_SEEDS
        ]

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a = results[i]
                b = results[j]
                perm, _, matched_sim = align_S(a["S"], b["S"])
                S_b_aligned = apply_perm_to_S(b["S"], perm)
                js = js_divergence_columns(a["S"], S_b_aligned).detach().cpu().numpy()

                rows.append({
                    "model_name": model_name,
                    "k": cfg.K,
                    "relation_temperature": T,
                    "seed_a": a["seed"],
                    "seed_b": b["seed"],
                    "matched_S_similarity": matched_sim,
                    "mean_JS": float(js.mean()),
                    "median_JS": float(np.median(js)),
                    "seed_a_S_unique_top1": a["diagnostics"]["S_unique_top1"],
                    "seed_b_S_unique_top1": b["diagnostics"]["S_unique_top1"],
                    "seed_a_S_rank": a["diagnostics"]["S_rank"],
                    "seed_b_S_rank": b["diagnostics"]["S_rank"],
                    "seed_a_recon_G": a["recon"],
                    "seed_b_recon_G": b["recon"],
                })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(cfg.OUT_DIR, f"final_same_G_rows_{safe_model_name(model_name)}.csv"), index=False)

        summaries.append({
            "model_name": model_name,
            "k": cfg.K,
            "relation_temperature": T,
            "same_G_mean_matched_S_similarity": float(df["matched_S_similarity"].mean()),
            "same_G_std_matched_S_similarity": float(df["matched_S_similarity"].std()),
            "same_G_mean_JS": float(df["mean_JS"].mean()),
            "same_G_median_JS": float(df["median_JS"].median()),
        })

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(os.path.join(cfg.OUT_DIR, "final_MS_same_G_summary.csv"), index=False)

    print("\n=== Same-G summary ===")
    print(summary_df.to_string(index=False))
    return summary_df
