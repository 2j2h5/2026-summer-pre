import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .metrics import (
    row_entropy,
    usage_entropy_from_rows,
    usage_entropy_from_cols,
    row_orthogonality_loss,
)
from semantic_alignment.utils.seed import set_seed


def factorize_G_MS(G, k, seed, cfg):
    set_seed(seed)

    N = G.shape[0]
    device = G.device

    M_param = torch.randn(N, k, device=device) * 0.01
    S_param = torch.randn(k, N, device=device) * 0.01
    M_param.requires_grad_(True)
    S_param.requires_grad_(True)

    optimizer = torch.optim.Adam([M_param, S_param], lr=cfg.LR)
    history = []

    for step in tqdm(range(cfg.STEPS), desc=f"G≈MS k={k} seed={seed}"):
        M = F.softmax(M_param / cfg.M_TEMPERATURE, dim=1)
        S = F.softmax(S_param / cfg.S_TEMPERATURE, dim=1)
        G_hat = M @ S

        recon_loss = F.mse_loss(G_hat, G)
        m_entropy = row_entropy(M)
        s_entropy = row_entropy(S)
        m_usage = usage_entropy_from_rows(M)
        s_usage = usage_entropy_from_cols(S)
        s_orth = row_orthogonality_loss(S)

        loss = (
            cfg.RECON_WEIGHT * recon_loss
            + cfg.M_ENTROPY_WEIGHT * m_entropy
            + cfg.S_ENTROPY_WEIGHT * s_entropy
            - cfg.M_USAGE_WEIGHT * m_usage
            - cfg.S_USAGE_WEIGHT * s_usage
            + cfg.S_ORTH_WEIGHT * s_orth
            + cfg.M_L2_WEIGHT * (M ** 2).mean()
            + cfg.S_L2_WEIGHT * (S ** 2).mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.LOG_EVERY == 0 or step == cfg.STEPS - 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "recon": float(recon_loss.detach().cpu()),
                "m_entropy": float(m_entropy.detach().cpu()),
                "s_entropy": float(s_entropy.detach().cpu()),
                "m_usage_entropy": float(m_usage.detach().cpu()),
                "s_usage_entropy": float(s_usage.detach().cpu()),
                "s_orth_loss": float(s_orth.detach().cpu()),
            }
            history.append(row)
            print(row)

    with torch.no_grad():
        M = F.softmax(M_param / cfg.M_TEMPERATURE, dim=1)
        S = F.softmax(S_param / cfg.S_TEMPERATURE, dim=1)
        G_hat = M @ S
        recon = F.mse_loss(G_hat, G).item()

        S_top1 = S.argmax(dim=0)
        M_top1 = M.argmax(dim=1)
        sv = torch.linalg.svdvals(S)

        diagnostics = {
            "M_unique_top1": int(M_top1.unique().numel()),
            "S_unique_top1": int(S_top1.unique().numel()),
            "S_rank": int(torch.linalg.matrix_rank(S).item()),
            "M_rank": int(torch.linalg.matrix_rank(M).item()),
            "S_std": float(S.std().item()),
            "M_std": float(M.std().item()),
            "S_row_entropy_mean": float(row_entropy(S).item()),
            "M_row_entropy_mean": float(row_entropy(M).item()),
            "S_usage_entropy": float(usage_entropy_from_cols(S).item()),
            "M_usage_entropy": float(usage_entropy_from_rows(M).item()),
            "S_singular_top1": float(sv[0].item()),
            "S_singular_top2_ratio": float((sv[1] / sv[0]).item()) if len(sv) > 1 else 0.0,
        }

    return M.detach(), S.detach(), G_hat.detach(), recon, pd.DataFrame(history), diagnostics


def fit_M_given_common_S(G, S_common, seed, cfg):
    set_seed(seed)

    N = G.shape[0]
    k = S_common.shape[0]
    device = G.device

    M_param = torch.randn(N, k, device=device) * 0.01
    M_param.requires_grad_(True)
    optimizer = torch.optim.Adam([M_param], lr=cfg.FIT_M_TO_COMMON_S_LR)

    history = []
    S_fixed = S_common.detach()

    for step in tqdm(range(cfg.FIT_M_TO_COMMON_S_STEPS), desc=f"fit M | fixed S* seed={seed}"):
        M = F.softmax(M_param / cfg.FIT_M_TO_COMMON_S_TEMPERATURE, dim=1)
        G_hat = M @ S_fixed

        recon_loss = F.mse_loss(G_hat, G)
        m_entropy = row_entropy(M)
        m_usage = usage_entropy_from_rows(M)

        loss = (
            recon_loss
            + cfg.FIT_M_TO_COMMON_S_ENTROPY_WEIGHT * m_entropy
            - cfg.FIT_M_TO_COMMON_S_USAGE_WEIGHT * m_usage
            + cfg.FIT_M_TO_COMMON_S_L2_WEIGHT * (M ** 2).mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.LOG_EVERY == 0 or step == cfg.FIT_M_TO_COMMON_S_STEPS - 1:
            history.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "recon": float(recon_loss.detach().cpu()),
                "m_entropy": float(m_entropy.detach().cpu()),
                "m_usage_entropy": float(m_usage.detach().cpu()),
            })

    with torch.no_grad():
        M = F.softmax(M_param / cfg.FIT_M_TO_COMMON_S_TEMPERATURE, dim=1)
        G_hat = M @ S_fixed
        recon = F.mse_loss(G_hat, G).item()
        M_top1 = M.argmax(dim=1)
        diagnostics = {
            "M_unique_top1": int(M_top1.unique().numel()),
            "M_rank": int(torch.linalg.matrix_rank(M).item()),
            "M_std": float(M.std().item()),
            "M_row_entropy_mean": float(row_entropy(M).item()),
            "M_usage_entropy": float(usage_entropy_from_rows(M).item()),
        }

    return M.detach(), G_hat.detach(), recon, pd.DataFrame(history), diagnostics
