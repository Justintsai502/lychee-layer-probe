"""
Layer metrics on a complete speaker x sentence table (numpy only).

  variance_shares   two-way decomposition: how much of the layer's variance is explained
                    by WHO speaks (acoustic) vs WHAT is said (semantic). Never saturates,
                    and gives the acoustic : semantic ratio directly.
  probe_accuracy    grouped-CV ridge classifier (can the label be read out?)
  similarity_index  s_spk - s_txt on the whole space (which factor organizes it?)
"""

import numpy as np

LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)   # relative to mean(diag(XX^T))


def zscore(X):
    X = np.asarray(X, dtype=np.float64)
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


# ---------------------------------------------------------------- variance decomposition

def variance_shares(X, speakers, sentences):
    """Share of total variance explained by speaker and by sentence.

    Features are z-scored per dimension first, so every dimension counts equally
    (LLM hidden states have a few huge-magnitude dimensions that would otherwise
    dominate). With z-scored X the grand mean is 0 and

        eta_spk  = sum_s n_s |mean_s|^2 / sum_i |x_i|^2
        eta_sent = sum_t n_t |mean_t|^2 / sum_i |x_i|^2

    On a balanced table these two and the residual (interaction + noise) add up to 1.
    Group means pick up some noise even without any real effect, so that floor is
    removed omega^2-style, from the RESIDUAL variance:

        ms_resid = (1 - eta_spk - eta_sent) / (N - k_spk - k_sent + 1)
        excess_spk = max(eta_spk - (k_spk - 1) * ms_resid, 0)          (same for sentence)
        acoustic_share = excess_spk / (excess_spk + excess_sent)   in [0, 1]

    The floor must come from the residual, not from the total: every speaker reads
    every sentence, so the sentence effect cancels inside speaker means, and when the
    sentence explains 80 % of the variance the chance level for speaker is 5x smaller
    than (k - 1) / (N - 1). Using the total clipped real speaker variance to 0.

    1 = everything the layer encodes about these clips is the voice, 0 = it is the words.
    """
    Z = zscore(X)
    spk, sent = np.asarray(speakers), np.asarray(sentences)
    total = float((Z ** 2).sum())

    def explained(labels):
        ss = 0.0
        for g in np.unique(labels):
            m = labels == g
            ss += m.sum() * float((Z[m].mean(0) ** 2).sum())
        return ss / total

    out = shares_from_eta(explained(spk), explained(sent), len(Z),
                          len(np.unique(spk)), len(np.unique(sent)))
    return out


def shares_from_eta(eta_spk, eta_sent, n, k_spk, k_sent):
    """Excess shares and acoustic_share from the two eta values of a balanced table."""
    df_resid = max(n - k_spk - k_sent + 1, 1)
    ms_resid = max(1.0 - eta_spk - eta_sent, 0.0) / df_resid
    null_spk, null_sent = (k_spk - 1) * ms_resid, (k_sent - 1) * ms_resid
    ex_spk, ex_sent = max(eta_spk - null_spk, 0.0), max(eta_sent - null_sent, 0.0)
    denom = ex_spk + ex_sent
    return {
        "eta_spk": eta_spk, "eta_sent": eta_sent,
        "null_spk": null_spk, "null_sent": null_sent,
        "excess_spk": ex_spk, "excess_sent": ex_sent,
        "acoustic_share": ex_spk / denom if denom > 0 else float("nan"),
    }


# ---------------------------------------------------------------- grouped CV probe

def group_folds(groups, n_splits, seed=0):
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    n_splits = min(n_splits, len(uniq))
    if n_splits < 2:
        raise ValueError(f"need >= 2 groups for cross-validation, got {len(uniq)}")
    order = uniq[np.random.default_rng(seed).permutation(len(uniq))]
    fold_of = {g: i % n_splits for i, g in enumerate(order)}
    fold = np.array([fold_of[g] for g in groups])
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(n_splits)]


def _zfit(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(a - mu) / sd for a in (train,) + others]


class DualRidge:
    """Ridge in the dual (n << d): one eigendecomposition serves every lambda."""

    def __init__(self, X, Y):
        self.X, self.Ymean = X, Y.mean(0)
        K = X @ X.T
        self.scale = float(np.trace(K)) / len(K)
        s, U = np.linalg.eigh(K)
        self.s, self.U = np.clip(s, 0, None), U
        self.UtY = U.T @ (Y - self.Ymean)

    def predict(self, Xte, lam):
        alpha = self.U @ (self.UtY / (self.s + lam * self.scale)[:, None])
        return (Xte @ self.X.T) @ alpha + self.Ymean


def probe_accuracy(X, y, groups, n_splits=5, inner_splits=3, seed=0):
    """Grouped CV accuracy of a ridge classifier. Returns (mean, std)."""
    X = np.asarray(X, dtype=np.float64)
    _, yi = np.unique(np.asarray(y), return_inverse=True)
    Y = np.eye(yi.max() + 1)[yi]
    groups = np.asarray(groups)
    accs = []
    for tr, te in group_folds(groups, n_splits, seed):
        Xtr, Xte = _zfit(X[tr], X[te])
        inner = group_folds(groups[tr], inner_splits, seed + 1) if len(set(groups[tr].tolist())) >= 2 else []
        scores = np.zeros(len(LAMBDAS))
        scores[LAMBDAS.index(1.0)] = 1e-9
        for itr, ite in inner:
            a, b = _zfit(X[tr][itr], X[tr][ite])
            model = DualRidge(a, Y[tr][itr])
            for j, lam in enumerate(LAMBDAS):
                scores[j] += np.mean(model.predict(b, lam).argmax(1) == yi[tr][ite])
        lam = LAMBDAS[int(np.argmax(scores))]
        accs.append(np.mean(DualRidge(Xtr, Y[tr]).predict(Xte, lam).argmax(1) == yi[te]))
    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------- similarity index

def similarity_index(X, speakers, sentences):
    """(s_spk, s_txt, s_spk - s_txt) on z-scored, L2-normalized features."""
    c = cosine_profile(X, speakers, sentences, standardize=True)
    return c["s_spk"], c["s_txt"], c["s_spk"] - c["s_txt"]


def cosine_profile(X, speakers, sentences, standardize=True):
    """Mean pairwise cosine for the three kinds of pairs on the speaker x sentence table.

      s_spk   same speaker,      different sentence
      s_txt   different speaker, same sentence
      s_none  different speaker, different sentence   (the baseline: nothing shared)

    standardize=True z-scores every dimension first. Raw LLM hidden states share one
    dominant direction (anisotropy), so raw cosines sit near 1 for every pair and the
    differences between pair types are compressed; the raw version is kept for reference.
    """
    Z = zscore(X) if standardize else np.asarray(X, dtype=np.float64).copy()
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    C = Z @ Z.T
    spk, sent = np.asarray(speakers), np.asarray(sentences)
    same_spk = spk[:, None] == spk[None, :]
    same_sent = sent[:, None] == sent[None, :]
    return {"s_spk": float(C[same_spk & ~same_sent].mean()),
            "s_txt": float(C[~same_spk & same_sent].mean()),
            "s_none": float(C[~same_spk & ~same_sent].mean())}
