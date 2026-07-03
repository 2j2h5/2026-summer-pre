import itertools
import os
import numpy as np
import pandas as pd
import torch

from semantic_alignment.experiments.temperature import factorize_model_at_T
from semantic_alignment.core.factorization_ms import fit_M_given_common_S
from semantic_alignment.core.metrics import (
    align_S,
    apply_perm_to_S,
    apply_perm_to_M,
    js_divergence_columns,
    cosine_rows,
)
from semantic_alignment.utils.names import safe_model_name


def run_final_models(graphs, cfg):
    results = {}

    for i, model_name in enumerate(cfg.MODEL_NAMES):
        print("\n" + "=" * 100)
        print(f"Final MS factorization | {model_name}")
        print("=" * 100)

        r = factorize_model_at_T(
            model_name=model_name,
            E=graphs[model_name]["E"],
            T=graphs[model_name]["relation_temperature"],
            seed=cfg.SEED + i,
            cfg=cfg,
        )
        results[model_name] = r
        r["history"].to_csv(
            os.path.join(cfg.OUT_DIR, f"final_history_{safe_model_name(model_name)}.csv"),
            index=False,
        )

    return results


def run_pairwise(results, words, cfg):
    rows = []
    matrix = pd.DataFrame(np.eye(len(cfg.MODEL_NAMES)), index=cfg.MODEL_NAMES, columns=cfg.MODEL_NAMES)

    for model_a, model_b in itertools.combinations(cfg.MODEL_NAMES, 2):
        a = results[model_a]
        b = results[model_b]

        perm, sim_matrix, matched_sim = align_S(a["S"], b["S"])
        S_b_aligned = apply_perm_to_S(b["S"], perm)
        M_b_aligned = apply_perm_to_M(b["M"], perm)

        js = js_divergence_columns(a["S"], S_b_aligned).detach().cpu().numpy()
        M_factor_cos = cosine_rows(a["M"].T, M_b_aligned.T).diag().mean().item()
        M_word_cos = cosine_rows(a["M"], M_b_aligned).diag().mean().item()

        row = {
            "k": cfg.K,
            "model_a": model_a,
            "model_b": model_b,
            "model_a_relation_temperature": a["relation_temperature"],
            "model_b_relation_temperature": b["relation_temperature"],
            "recon_a_G": a["recon"],
            "recon_b_G": b["recon"],
            "mean_hungarian_matched_S_similarity": matched_sim,
            "mean_JS_between_aligned_S_columns": float(js.mean()),
            "median_JS_between_aligned_S_columns": float(np.median(js)),
            "M_factor_cosine_after_S_alignment": M_factor_cos,
            "M_word_row_cosine_after_S_alignment": M_word_cos,
            "a_S_unique_top1": a["diagnostics"]["S_unique_top1"],
            "b_S_unique_top1": b["diagnostics"]["S_unique_top1"],
            "a_M_unique_top1": a["diagnostics"]["M_unique_top1"],
            "b_M_unique_top1": b["diagnostics"]["M_unique_top1"],
            "a_S_rank": a["diagnostics"]["S_rank"],
            "b_S_rank": b["diagnostics"]["S_rank"],
            "a_M_rank": a["diagnostics"]["M_rank"],
            "b_M_rank": b["diagnostics"]["M_rank"],
        }

        rows.append(row)
        matrix.loc[model_a, model_b] = matched_sim
        matrix.loc[model_b, model_a] = matched_sim

        pair_safe = f"{safe_model_name(model_a)}__VS__{safe_model_name(model_b)}"
        torch.save(
            {
                "k": cfg.K,
                "words": words,
                "model_a": {
                    "model_name": model_a,
                    "relation_temperature": a["relation_temperature"],
                    "M": a["M"].cpu(),
                    "S": a["S"].cpu(),
                    "G": a["G"].cpu(),
                    "C": a["C"].cpu(),
                    "recon": a["recon"],
                    "diagnostics": a["diagnostics"],
                },
                "model_b": {
                    "model_name": model_b,
                    "relation_temperature": b["relation_temperature"],
                    "M": b["M"].cpu(),
                    "M_aligned_to_a": M_b_aligned.cpu(),
                    "S": b["S"].cpu(),
                    "S_aligned_to_a": S_b_aligned.cpu(),
                    "G": b["G"].cpu(),
                    "C": b["C"].cpu(),
                    "recon": b["recon"],
                    "diagnostics": b["diagnostics"],
                },
                "perm_b_to_a": perm,
                "similarity_matrix": sim_matrix,
                "summary": row,
            },
            os.path.join(cfg.OUT_DIR, f"final_MS_pair_k{cfg.K}_{pair_safe}.pt"),
        )

    pair_df = pd.DataFrame(rows).sort_values("mean_hungarian_matched_S_similarity", ascending=False)
    pair_df.to_csv(os.path.join(cfg.OUT_DIR, "final_MS_pairwise_alignment.csv"), index=False)
    matrix.to_csv(os.path.join(cfg.OUT_DIR, "final_MS_pairwise_alignment_matrix.csv"))

    print("\n=== Pairwise S alignment matrix ===")
    print(matrix.to_string())
    print("\n=== Pairwise rows ===")
    print(pair_df.to_string(index=False))
    return pair_df, matrix


def fit_all_M_to_common_S(graphs, S_common, cfg):
    rows = []
    fitted = {}

    for i, model_name in enumerate(cfg.MODEL_NAMES):
        print("\n" + "=" * 100)
        print(f"Fit M to common S* | {model_name}")
        print("=" * 100)

        G = graphs[model_name]["G"]
        M, G_hat, recon, history, diagnostics = fit_M_given_common_S(G, S_common, cfg.SEED + i, cfg)

        fitted[model_name] = {
            "model_name": model_name,
            "relation_temperature": graphs[model_name]["relation_temperature"],
            "M_commonS": M,
            "G_hat_commonS": G_hat,
            "recon_commonS": recon,
            "history": history,
            "diagnostics": diagnostics,
        }

        history.to_csv(os.path.join(cfg.OUT_DIR, f"fit_M_commonS_history_{safe_model_name(model_name)}.csv"), index=False)
        rows.append({
            "model_name": model_name,
            "relation_temperature": graphs[model_name]["relation_temperature"],
            "recon_commonS": recon,
            **diagnostics,
        })

    pd.DataFrame(rows).to_csv(os.path.join(cfg.OUT_DIR, "fit_M_to_commonS_diagnostics.csv"), index=False)

    comp_rows = []
    for a, b in itertools.combinations(cfg.MODEL_NAMES, 2):
        M_a = fitted[a]["M_commonS"]
        M_b = fitted[b]["M_commonS"]

        comp_rows.append({
            "model_a": a,
            "model_b": b,
            "M_factor_cosine_mean": cosine_rows(M_a.T, M_b.T).diag().mean().item(),
            "M_word_row_cosine_mean": cosine_rows(M_a, M_b).diag().mean().item(),
            "recon_a_commonS": fitted[a]["recon_commonS"],
            "recon_b_commonS": fitted[b]["recon_commonS"],
        })

    comp_df = pd.DataFrame(comp_rows).sort_values("M_factor_cosine_mean", ascending=False)
    comp_df.to_csv(os.path.join(cfg.OUT_DIR, "pairwise_M_comparison_under_commonS.csv"), index=False)

    torch.save(
        {
            "S_common": S_common.cpu(),
            "fitted": {
                m: {
                    "model_name": r["model_name"],
                    "relation_temperature": r["relation_temperature"],
                    "M_commonS": r["M_commonS"].cpu(),
                    "G_hat_commonS": r["G_hat_commonS"].cpu(),
                    "recon_commonS": r["recon_commonS"],
                    "diagnostics": r["diagnostics"],
                }
                for m, r in fitted.items()
            },
        },
        os.path.join(cfg.OUT_DIR, "fit_M_to_commonS.pt"),
    )

    print("\n=== M comparison under common S ===")
    print(comp_df.to_string(index=False))
    return fitted, comp_df
