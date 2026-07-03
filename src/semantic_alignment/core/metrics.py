import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def row_entropy(X, eps=1e-12):
    X = X.clamp_min(eps)
    X = X / (X.sum(dim=1, keepdim=True) + eps)
    return -(X * X.log()).sum(dim=1).mean()


def usage_entropy_from_rows(M, eps=1e-12):
    usage = M.mean(dim=0)
    usage = usage / (usage.sum() + eps)
    return -(usage * torch.log(usage + eps)).sum()


def usage_entropy_from_cols(S, eps=1e-12):
    usage = S.mean(dim=1)
    usage = usage / (usage.sum() + eps)
    return -(usage * torch.log(usage + eps)).sum()


def row_orthogonality_loss(S):
    S_norm = S / (S.norm(dim=1, keepdim=True) + 1e-8)
    C = S_norm @ S_norm.T
    I = torch.eye(S.shape[0], device=S.device)
    return F.mse_loss(C, I)


def cosine_rows(A, B, eps=1e-8):
    A = A / (A.norm(dim=1, keepdim=True) + eps)
    B = B / (B.norm(dim=1, keepdim=True) + eps)
    return A @ B.T


def align_S(S_ref, S_target):
    sim = cosine_rows(S_ref, S_target)
    cost = -sim.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.zeros_like(row_ind)
    perm[row_ind] = col_ind
    matched_sim = sim[row_ind, col_ind].mean().item()
    return perm, sim.detach().cpu().numpy(), matched_sim


def apply_perm_to_S(S, perm):
    perm_t = torch.tensor(perm, dtype=torch.long, device=S.device)
    return S[perm_t, :]


def apply_perm_to_M(M, perm):
    perm_t = torch.tensor(perm, dtype=torch.long, device=M.device)
    return M[:, perm_t]


def js_divergence_columns(A, B, eps=1e-12):
    A = A.clamp_min(eps)
    B = B.clamp_min(eps)

    A = A / (A.sum(dim=0, keepdim=True) + eps)
    B = B / (B.sum(dim=0, keepdim=True) + eps)
    M = 0.5 * (A + B)

    kl_a = (A * (A.log() - M.log())).sum(dim=0)
    kl_b = (B * (B.log() - M.log())).sum(dim=0)
    return 0.5 * (kl_a + kl_b)


def graph_entropy_stats(G, eps=1e-12):
    P = G.clamp_min(eps)
    P = P / (P.sum(dim=1, keepdim=True) + eps)
    entropy = -(P * P.log()).sum(dim=1)
    effective_neighbors = torch.exp(entropy)
    top1 = P.max(dim=1).values
    top5 = P.topk(min(5, P.shape[1]), dim=1).values.sum(dim=1)
    return {
        "graph_row_entropy_mean": float(entropy.mean().item()),
        "graph_effective_neighbors_mean": float(effective_neighbors.mean().item()),
        "graph_top1_mass_mean": float(top1.mean().item()),
        "graph_top5_mass_mean": float(top5.mean().item()),
    }
