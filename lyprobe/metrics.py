"""
Layer metrics on a complete speaker x sentence table (numpy only).

  variance_shares   two-way decomposition: how much of the layer's variance is explained
                    by WHO speaks (acoustic) vs WHAT is said (semantic). Never saturates,
                    and gives the acoustic : semantic ratio directly.
  probe_accuracy    grouped-CV ridge classifier (can the label be read out?)
  similarity_index  s_spk - s_txt on the whole space (which factor organizes it?)
  eer_cosine        speaker-verification EER, cosine on z-scored features, no training
  eer_wccn          speaker-verification EER on unseen speakers with a WCCN backend
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


# ---------------------------------------------------------------- speaker verification EER

def eer(target_scores, nontarget_scores):
    """Equal error rate: the threshold where false accepts = false rejects. 0 = perfect,
    0.5 = chance. Can exceed 0.5 when non-target pairs score HIGHER than target pairs."""
    t = np.asarray(target_scores, dtype=np.float64)
    n = np.asarray(nontarget_scores, dtype=np.float64)
    if len(t) == 0 or len(n) == 0:
        return float("nan")
    s = np.concatenate([t, n])
    y = np.concatenate([np.ones(len(t)), np.zeros(len(n))])
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    # threshold just below the k-th highest score: accept the top k
    frr = np.concatenate([[1.0], 1.0 - np.cumsum(y) / len(t)])
    far = np.concatenate([[0.0], np.cumsum(1 - y) / len(n)])
    k = int(np.argmin(np.abs(frr - far)))
    # interpolate between the two thresholds around the crossing
    if k + 1 < len(frr) and (frr[k] - far[k]) * (frr[k + 1] - far[k + 1]) < 0:
        a, b = frr[k] - far[k], frr[k + 1] - far[k + 1]
        w = a / (a - b)
        return float(far[k] + w * (far[k + 1] - far[k]))
    return float((frr[k] + far[k]) / 2)


def trial_masks(spk, sent):
    """Upper-triangle pair masks: target (same speaker, different sentence), nontarget
    (different speaker, different sentence) and hard nontarget (different speaker, SAME
    sentence). Same-speaker same-sentence pairs do not exist (one take per cell)."""
    spk, sent = np.asarray(spk), np.asarray(sent)
    same_spk = spk[:, None] == spk[None, :]
    same_sent = sent[:, None] == sent[None, :]
    upper = np.triu(np.ones(same_spk.shape, bool), k=1)
    return upper & same_spk & ~same_sent, upper & ~same_spk & ~same_sent, upper & ~same_spk & same_sent


def _unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def eer_cosine(X, spk, sent):
    """Training-free EER: cosine on z-scored features (statistics from all clips, no
    labels). Returns (eer, eer_hard)."""
    Z = _unit(zscore(X))
    C = Z @ Z.T
    tgt, non, hard = trial_masks(spk, sent)
    return eer(C[tgt], C[non]), eer(C[tgt], C[hard])


def wccn_fit(X, y, pca_dim=200, reg=1e-3):
    """PCA (on z-scored X) then within-class covariance normalization (WCCN): whiten the
    variation INSIDE each speaker (different sentences, noise), keep every direction.

    Chosen over LDA: LDA keeps only the (n_train_speakers - 1) = 19 directions that
    separate the training speakers, and new speakers fall partly outside that span
    (on synthetic voice-only data LDA gives EER 0.12 where cosine gives 0.02; WCCN 0.02).
    Returns a projector function."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    k = max(1, min(pca_dim, Z.shape[0] - 1, Z.shape[1]))
    if Z.shape[1] > Z.shape[0]:
        # n << D (600 clips x 3584 dims): PCA from the n x n Gram matrix, same result as
        # the SVD but one matrix product + a small eigh instead of a 600 x 3584 SVD
        s, U = np.linalg.eigh(Z @ Z.T)
        s, U = s[::-1][:k], U[:, ::-1][:, :k]
        P = (Z.T @ U) / np.sqrt(np.clip(s, 1e-12, None))
    else:
        P = np.linalg.svd(Z, full_matrices=False)[2][:k].T
    Y = Z @ P
    Sw = np.zeros((k, k))
    for c in np.unique(y):
        D = Y[y == c] - Y[y == c].mean(0)
        Sw += D.T @ D
    Sw += reg * np.trace(Sw) / k * np.eye(k)
    ev, V = np.linalg.eigh(Sw)
    A = P @ (V / np.sqrt(np.clip(ev, 1e-12, None)))
    return lambda Xn: ((Xn - mu) / sd) @ A


def eer_wccn(X, spk, sent, n_splits=5, pca_dim=200, seed=0):
    """EER on UNSEEN speakers: WCCN backend trained on the other speakers' clips
    (speaker-disjoint folds, 20 train / 5 test speakers), cosine scoring of trials among
    the held-out speakers, scores pooled over folds. Returns (eer, eer_hard)."""
    X = np.asarray(X, dtype=np.float64)
    spk, sent = np.asarray(spk), np.asarray(sent)
    if len(np.unique(spk)) < 4:
        return float("nan"), float("nan")
    T, N, H = [], [], []
    for tr, te in group_folds(spk, n_splits, seed):
        Z = _unit(wccn_fit(X[tr], spk[tr], pca_dim)(X[te]))
        C = Z @ Z.T
        tgt, non, hard = trial_masks(spk[te], sent[te])
        T.append(C[tgt])
        N.append(C[non])
        H.append(C[hard])
    T, N, H = np.concatenate(T), np.concatenate(N), np.concatenate(H)
    return eer(T, N), eer(T, H)
