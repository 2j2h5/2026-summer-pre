import itertools
import os
import numpy as np
import pandas as pd

from semantic_alignment.core.graph import make_relation_graph
from semantic_alignment.core.factorization_ms import factorize_G_MS
from semantic_alignment.core.metrics import align_S, apply_perm_to_S, js_divergence_columns
from semantic_alignment.utils.names import safe_model_name


def factorize_model_at_T(model_name, E, T, seed, cfg):
    G, C = make_relation_graph(E, temperature=T, symmetrize=True)
    M, S, G_hat, recon, history, diagnostics = factorize_G_MS(G, cfg.K, seed, cfg)
    return {
        "model_name": model_name,
        "relation_temperature": T,
        "seed": seed,
        "G": G,
        "C": C,
        "M": M,
        "S": S,
        "G_hat": G_hat,
        "recon": recon,
        "history": history,
        "diagnostics": diagnostics,
    }


def evaluate_same_G_for_T(model_name, E, T, seeds, cfg):
    if len(seeds) < 2:
        raise ValueError(f"same-G evaluation needs at least 2 seeds, got {seeds}")

    results = [factorize_model_at_T(model_name, E, T, seed, cfg) for seed in seeds]
    rows = []

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

    summary = {
        "model_name": model_name,
        "k": cfg.K,
        "relation_temperature": T,
        "same_G_mean_matched_S_similarity": float(df["matched_S_similarity"].mean()),
        "same_G_std_matched_S_similarity": float(df["matched_S_similarity"].std()),
        "same_G_min_matched_S_similarity": float(df["matched_S_similarity"].min()),
        "same_G_mean_JS": float(df["mean_JS"].mean()),
        "same_G_median_JS": float(df["median_JS"].median()),
        "mean_S_unique_top1": float(pd.concat([df["seed_a_S_unique_top1"], df["seed_b_S_unique_top1"]]).mean()),
        "mean_S_rank": float(pd.concat([df["seed_a_S_rank"], df["seed_b_S_rank"]]).mean()),
        "mean_recon_G": float(pd.concat([df["seed_a_recon_G"], df["seed_b_recon_G"]]).mean()),
    }

    summary["T_selection_score"] = (
        cfg.SCORE_SAMEG_WEIGHT * summary["same_G_mean_matched_S_similarity"]
        - cfg.SCORE_JS_WEIGHT * summary["same_G_mean_JS"]
        + cfg.SCORE_UNIQUE_WEIGHT * summary["mean_S_unique_top1"]
        + cfg.SCORE_RANK_WEIGHT * summary["mean_S_rank"]
        - cfg.SCORE_RECON_WEIGHT * summary["mean_recon_G"]
    )

    return df, summary


def make_fine_T_candidates(best_T, radius, num_points, min_T=0.03, max_T=1.50):
    lo = max(min_T, best_T - radius)
    hi = min(max_T, best_T + radius)
    return sorted(set(float(round(t, 5)) for t in np.linspace(lo, hi, num_points)))


def select_best_T_for_all_models(embeddings, cfg):
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    best_T = {}
    all_summaries = []

    for model_name, E in embeddings.items():
        rows = []
        summaries = []

        print("\n" + "=" * 100)
        print(f"T search | {model_name}")
        print("=" * 100)

        for T in cfg.COARSE_T_CANDIDATES:
            df, summary = evaluate_same_G_for_T(model_name, E, T, cfg.COARSE_T_SELECTION_SEEDS, cfg)
            df["stage"] = "coarse"
            summary["stage"] = "coarse"
            rows.append(df)
            summaries.append(summary)

        coarse_df = pd.DataFrame(summaries).sort_values("T_selection_score", ascending=False)
        coarse_best_T = float(coarse_df.iloc[0]["relation_temperature"])
        fine_Ts = make_fine_T_candidates(coarse_best_T, cfg.FINE_RADIUS, cfg.FINE_NUM_POINTS)

        for T in fine_Ts:
            df, summary = evaluate_same_G_for_T(model_name, E, T, cfg.FINE_T_SELECTION_SEEDS, cfg)
            df["stage"] = "fine"
            summary["stage"] = "fine"
            rows.append(df)
            summaries.append(summary)

        summary_df = pd.DataFrame(summaries).sort_values("T_selection_score", ascending=False)
        best_T[model_name] = float(summary_df.iloc[0]["relation_temperature"])
        all_summaries.extend(summaries)

        pd.concat(rows, ignore_index=True).to_csv(
            os.path.join(cfg.OUT_DIR, f"auto_T_rows_{safe_model_name(model_name)}.csv"),
            index=False,
        )
        summary_df.to_csv(
            os.path.join(cfg.OUT_DIR, f"auto_T_summary_{safe_model_name(model_name)}.csv"),
            index=False,
        )

    best_T_df = pd.DataFrame([
        {"model_name": model_name, "selected_relation_temperature": T}
        for model_name, T in best_T.items()
    ])
    best_T_df.to_csv(os.path.join(cfg.OUT_DIR, "selected_model_relation_temperatures_MS.csv"), index=False)
    pd.DataFrame(all_summaries).to_csv(os.path.join(cfg.OUT_DIR, "auto_T_selection_summary.csv"), index=False)

    print("\n=== Selected T ===")
    print(best_T_df.to_string(index=False))

    return best_T
