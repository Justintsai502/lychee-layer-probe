"""
GPU (torch) versions of metrics.eer_cosine and metrics.eer_wccn: same math, same folds,
same trials, same numbers (float64), but the matrix work runs on the GPU. Used by
analyze_eer.py when torch + CUDA are available; the numpy versions stay the reference
(tests/test_offline.py checks the two agree).

Model deps (lazy): torch
"""

import numpy as np

from lyprobe.metrics import group_folds, trial_masks


def _t(x, dev):
    import torch
    return torch.as_tensor(np.asarray(x, dtype=np.float64), device=dev)


def _masks(spk, sent, dev):
    import torch
    return [torch.as_tensor(m, device=dev) for m in trial_masks(spk, sent)]


def eer(t, n):
    """metrics.eer on torch tensors."""
    import torch
    if t.numel() == 0 or n.numel() == 0:
        return float("nan")
    s = torch.cat([t, n])
    y = torch.cat([torch.ones_like(t), torch.zeros_like(n)])
    y = y[torch.argsort(s, descending=True, stable=True)]
    frr = torch.cat([y.new_ones(1), 1.0 - torch.cumsum(y, 0) / len(t)])
    far = torch.cat([y.new_zeros(1), torch.cumsum(1.0 - y, 0) / len(n)])
    d = frr - far
    k = int(torch.argmin(d.abs()))
    if k + 1 < len(d) and float(d[k]) * float(d[k + 1]) < 0:
        a, b = float(d[k]), float(d[k + 1])
        w = a / (a - b)
        return float(far[k] + w * (far[k + 1] - far[k]))
    return float((frr[k] + far[k]) / 2)


def _unit(Z):
    return Z / (Z.norm(dim=1, keepdim=True) + 1e-12)


def eer_cosine(X, spk, sent, device="cuda"):
    X = _t(X, device)
    Z = _unit((X - X.mean(0)) / (X.std(0, correction=0) + 1e-6))
    C = Z @ Z.T
    tgt, non, hard = _masks(spk, sent, device)
    return eer(C[tgt], C[non]), eer(C[tgt], C[hard])


def wccn_fit(X, y, pca_dim=200, reg=1e-3):
    import torch
    mu, sd = X.mean(0), X.std(0, correction=0) + 1e-6
    Z = (X - mu) / sd
    k = max(1, min(pca_dim, Z.shape[0] - 1, Z.shape[1]))
    if Z.shape[1] > Z.shape[0]:
        s, U = torch.linalg.eigh(Z @ Z.T)
        s, U = s.flip(0)[:k], U.flip(1)[:, :k]
        P = (Z.T @ U) / torch.sqrt(s.clamp_min(1e-12))
    else:
        P = torch.linalg.svd(Z, full_matrices=False)[2][:k].T
    Y = Z @ P
    Sw = torch.zeros((k, k), dtype=Y.dtype, device=Y.device)
    for c in np.unique(y):
        D = Y[torch.as_tensor(y == c, device=Y.device)]
        D = D - D.mean(0)
        Sw += D.T @ D
    Sw += reg * torch.trace(Sw) / k * torch.eye(k, dtype=Y.dtype, device=Y.device)
    ev, V = torch.linalg.eigh(Sw)
    A = P @ (V / torch.sqrt(ev.clamp_min(1e-12)))
    return lambda Xn: ((Xn - mu) / sd) @ A


def eer_wccn(X, spk, sent, n_splits=5, pca_dim=200, seed=0, device="cuda"):
    import torch
    spk, sent = np.asarray(spk), np.asarray(sent)
    if len(np.unique(spk)) < 4:
        return float("nan"), float("nan")
    X = _t(X, device)
    T, N, H = [], [], []
    for tr, te in group_folds(spk, n_splits, seed):
        tr_i, te_i = torch.as_tensor(tr, device=device), torch.as_tensor(te, device=device)
        Z = _unit(wccn_fit(X[tr_i], spk[tr], pca_dim)(X[te_i]))
        C = Z @ Z.T
        tgt, non, hard = _masks(spk[te], sent[te], device)
        T.append(C[tgt])
        N.append(C[non])
        H.append(C[hard])
    T, N, H = torch.cat(T), torch.cat(N), torch.cat(H)
    return eer(T, N), eer(T, H)
