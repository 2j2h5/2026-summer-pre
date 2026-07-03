from pathlib import Path
import glob
import torch
import pandas as pd

from semantic_alignment.core.metrics import cosine_rows


def analyze_M_from_pair_pts(out_dir: str):
    pt_dir = Path(out_dir)
    diag_dir = pt_dir / "M_diagnostics_from_pair_pts"
    diag_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    factor_rows = []
    word_rows = []

    paths = glob.glob(str(pt_dir / "final_MS_pair_k*.pt"))
    if not paths:
        raise FileNotFoundError(f"No final_MS_pair_k*.pt files found in {pt_dir}")

    for path in paths:
        data = torch.load(path, map_location="cpu")
        a = data["model_a"]
        b = data["model_b"]

        name_a = a["model_name"]
        name_b = b["model_name"]

        M_a = a["M"].float()
        M_b = b.get("M_aligned_to_a", b["M"]).float()

        row_cos = cosine_rows(M_a, M_b).diag()
        factor_cos = cosine_rows(M_a.T, M_b.T).diag()
        diff = M_a - M_b

        rows.append({
            "pt_file": Path(path).name,
            "model_a": name_a,
            "model_b": name_b,
            "M_word_row_cosine_mean": float(row_cos.mean()),
            "M_word_row_cosine_min": float(row_cos.min()),
            "M_factor_cosine_mean": float(factor_cos.mean()),
            "M_factor_cosine_min": float(factor_cos.min()),
            "M_L1_mean_abs_diff": float(diff.abs().mean()),
            "M_RMSE": float((diff ** 2).mean().sqrt()),
            "M_top1_factor_agreement": float((M_a.argmax(dim=1) == M_b.argmax(dim=1)).float().mean()),
        })

        factor_l1 = diff.abs().mean(dim=0)
        factor_rmse = torch.sqrt((diff ** 2).mean(dim=0))
        for idx in range(M_a.shape[1]):
            factor_rows.append({
                "pt_file": Path(path).name,
                "model_a": name_a,
                "model_b": name_b,
                "factor_index": idx,
                "factor_cosine": float(factor_cos[idx]),
                "factor_L1_mean_abs_diff": float(factor_l1[idx]),
                "factor_RMSE": float(factor_rmse[idx]),
            })

        word_l1 = diff.abs().mean(dim=1)
        word_rmse = torch.sqrt((diff ** 2).mean(dim=1))
        for idx in range(M_a.shape[0]):
            word_rows.append({
                "pt_file": Path(path).name,
                "model_a": name_a,
                "model_b": name_b,
                "word_index": idx,
                "word_row_cosine": float(row_cos[idx]),
                "word_L1_mean_abs_diff": float(word_l1[idx]),
                "word_RMSE": float(word_rmse[idx]),
                "top_factor_a": int(M_a[idx].argmax()),
                "top_factor_b": int(M_b[idx].argmax()),
                "top_factor_same": bool(M_a[idx].argmax() == M_b[idx].argmax()),
            })

    pair_df = pd.DataFrame(rows).sort_values("M_factor_cosine_mean", ascending=False)
    factor_df = pd.DataFrame(factor_rows).sort_values(
        ["factor_cosine", "factor_L1_mean_abs_diff"],
        ascending=[True, False],
    )
    word_df = pd.DataFrame(word_rows).sort_values(
        ["word_row_cosine", "word_L1_mean_abs_diff"],
        ascending=[True, False],
    )

    pair_df.to_csv(diag_dir / "M_pairwise_from_pair_pts.csv", index=False)
    factor_df.to_csv(diag_dir / "M_factor_level_differences.csv", index=False)
    word_df.to_csv(diag_dir / "M_word_level_differences.csv", index=False)

    print("\n=== Pairwise M diagnostics ===")
    print(pair_df.to_string(index=False))
    print("\nSaved to:", diag_dir)
    return pair_df, factor_df, word_df
