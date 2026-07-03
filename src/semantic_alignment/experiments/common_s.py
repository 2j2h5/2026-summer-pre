import os
import numpy as np
import pandas as pd
import torch

from semantic_alignment.core.metrics import (
    align_S,
    apply_perm_to_S,
    apply_perm_to_M,
    cosine_rows,
    js_divergence_columns,
)


def build_common_S_from_results(results, cfg):
    model_names = list(results.keys())
    ref_model = cfg.COMMON_S_REFERENCE_MODEL or model_names[0]
    S_ref = results[ref_model]["S"]

    aligned_results = {}
    aligned_S_list = []
    rows = []

    for model_name, r in results.items():
        if model_name == ref_model:
            S_aligned = r["S"]
            M_aligned = r["M"]
            matched = 1.0
        else:
            perm, _, matched = align_S(S_ref, r["S"])
            S_aligned = apply_perm_to_S(r["S"], perm)
            M_aligned = apply_perm_to_M(r["M"], perm)

        aligned_results[model_name] = {
            **r,
            "S_aligned": S_aligned,
            "M_aligned": M_aligned,
            "aligned_to": ref_model,
            "S_alignment_to_ref": matched,
        }
        aligned_S_list.append(S_aligned)

        rows.append({
            "model_name": model_name,
            "reference_model": ref_model,
            "S_alignment_to_reference": matched,
        })

    S_common = torch.stack(aligned_S_list, dim=0).mean(dim=0)
    S_common = S_common / (S_common.sum(dim=1, keepdim=True) + 1e-12)

    pd.DataFrame(rows).to_csv(os.path.join(cfg.OUT_DIR, "common_S_alignment_to_reference.csv"), index=False)
    torch.save(
        {"reference_model": ref_model, "S_common": S_common.cpu(), "alignment_rows": rows},
        os.path.join(cfg.OUT_DIR, "common_S.pt"),
    )
    return S_common.detach(), aligned_results


def evaluate_common_S_quality(S_common, aligned_results, cfg):
    rows = []

    for model_name, r in aligned_results.items():
        S_model = r["S_aligned"]
        sim = cosine_rows(S_common, S_model).diag().mean().item()
        js = js_divergence_columns(S_common, S_model).detach().cpu().numpy()

        rows.append({
            "model_name": model_name,
            "mean_diag_cosine_to_common_S": sim,
            "mean_JS_to_common_S": float(js.mean()),
            "median_JS_to_common_S": float(np.median(js)),
        })

    df = pd.DataFrame(rows).sort_values("mean_diag_cosine_to_common_S", ascending=False)
    df.to_csv(os.path.join(cfg.OUT_DIR, "common_S_quality.csv"), index=False)
    print("\n=== Common S Quality ===")
    print(df.to_string(index=False))
    return df
