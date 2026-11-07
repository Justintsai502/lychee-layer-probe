#!/usr/bin/env python3
"""
Acoustic vs semantic content at every point of the network.

Per point (and per baseline feature):
  eta_spk / eta_sent   share of variance explained by speaker / by sentence
  acoustic_share       speaker part of the above-chance explained variance, in [0, 1]
  spk_acc              grouped-CV ridge probe for speaker (held-out sentences)
  spk_index            s_spk - s_txt similarity index
  sent_acc             sentence probe, only with --content-probe (saturates on 30 sentences)

Works on any features file with pooled [N, P, D]: Lychee files carry point names and
branch tracks; the BayLing file from bayling-speaker-probe is read as a plain stack
(point layer_k, track trunk), so both models can be compared with one metric.

Output (--out): metrics.csv, summary.md, curves.png (if matplotlib is installed)

Hard deps: numpy
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import (PROJECT_ROOT, RESULTS_DIR, SIBLING_BASELINES, complete_grid,  # noqa: E402
                            setup_project_cache)
from lyprobe.metrics import cosine_profile, probe_accuracy, variance_shares  # noqa: E402


# ---------------------------------------------------------------- loading

def load_features(path):
    d = np.load(path, allow_pickle=False)
    P = d["pooled"].shape[1]
    if "point_names" in d.files:
        names, depths, tracks = d["point_names"].tolist(), d["point_depths"].tolist(), d["point_track"].tolist()
    else:
        names, depths, tracks = [f"layer_{i}" for i in range(P)], list(range(P)), ["trunk"] * P
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    model = meta.get("model") or ("BayLing-Duplex" if "context" in meta else Path(path).stem)
    return {"model": model, "pooled": d["pooled"], "utt_id": d["utt_id"], "speaker": d["speaker"],
            "sentence_id": d["sentence_id"], "names": names, "depths": depths, "tracks": tracks}


def load_baselines(path):
    if not path or not Path(path).exists():
        return {}
    d = np.load(path, allow_pickle=False)
    return {k[5:]: (d[k], d["utt_id"]) for k in d.files if k.startswith("feat_")}


def align(feat, baselines):
    """Restrict everything to the shared utterances that form a complete grid."""
    shared = set(feat["utt_id"].tolist())
    for _, u in baselines.values():
        shared &= set(u.tolist())
    keep = np.array([u in shared for u in feat["utt_id"]])
    grid, spk_set, sent_set = complete_grid(feat["speaker"][keep], feat["sentence_id"][keep])
    utts = feat["utt_id"][keep][grid]
    pos = {u: i for i, u in enumerate(feat["utt_id"].tolist())}
    idx = np.array([pos[u] for u in utts])
    base = {}
    for name, (F, u) in baselines.items():
        bpos = {x: i for i, x in enumerate(u.tolist())}
        base[name] = F[np.array([bpos[x] for x in utts])].astype(np.float32)
    return (feat["pooled"][idx].astype(np.float32), feat["speaker"][idx], feat["sentence_id"][idx],
            base, spk_set, sent_set)


# ---------------------------------------------------------------- metrics

def evaluate(X, spk, sent, folds, seed, content_probe, probe):
    r = variance_shares(X, spk, sent)
    r["spk_acc"] = probe_accuracy(X, spk, sent, folds, seed=seed)[0] if probe else float("nan")
    r["sent_acc"] = probe_accuracy(X, sent, spk, folds, seed=seed)[0] if content_probe else float("nan")
    c = cosine_profile(X, spk, sent, standardize=True)
    raw = cosine_profile(X, spk, sent, standardize=False)
    r.update(c)
    r.update({f"{k}_raw": v for k, v in raw.items()})
    r["spk_index"] = c["s_spk"] - c["s_txt"]
    return r


COLS = ["model", "point", "track", "depth", "acoustic_share", "eta_spk", "eta_sent",
        "excess_spk", "excess_sent", "spk_acc", "sent_acc", "spk_index",
        "s_spk", "s_txt", "s_none", "s_spk_raw", "s_txt_raw", "s_none_raw"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=PROJECT_ROOT / "work/features/lychee.npz")
    ap.add_argument("--baselines", type=Path, default=SIBLING_BASELINES,
                    help="baselines.npz from bayling-speaker-probe (ECAPA, Mimi); skipped if missing")
    ap.add_argument("--out", type=Path, default=None,
                    help="csv / summary / png folder (default: work/results/<name>)")
    ap.add_argument("--json", type=Path, default=None,
                    help="result JSON to commit (default: results/<name>.json)")
    ap.add_argument("--name", default=None, help="result name (default: from the model, e.g. lychee, bayling)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-probe", action="store_true", help="skip the speaker probe (faster)")
    ap.add_argument("--content-probe", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()
    setup_project_cache()

    if not cfg.features.exists():
        raise SystemExit(f"features not found: {cfg.features}")
    feat = load_features(cfg.features)
    name = cfg.name or feat["model"].split("-")[0].split(" ")[0].lower()
    cfg.out = cfg.out or PROJECT_ROOT / "work/results" / name
    cfg.json = cfg.json or RESULTS_DIR / f"{name}.json"
    baselines = load_baselines(cfg.baselines)
    if not baselines:
        print(f"no baselines at {cfg.baselines}; continuing without")
    F, spk, sent, base, spk_set, sent_set = align(feat, baselines)
    print(f"{feat['model']}: {len(spk)} clips | {len(spk_set)} speakers x {len(sent_set)} sentences "
          f"| {F.shape[1]} points")
    degenerate = len(spk_set) < 3 or len(sent_set) < 3
    if degenerate:
        # 1 speaker: the probe is trivially 1.0 and there is no speaker variance, so
        # acoustic_share is 0/0 rounded by float noise. Fine for checking the pipeline only.
        print("WARNING: fewer than 3 speakers or sentences - every number below is meaningless "
              "(pipeline check only); the result JSON is not written to results/")
        if cfg.json.parent.resolve() == RESULTS_DIR.resolve():
            cfg.json = PROJECT_ROOT / "work/results" / f"{name}_DEGENERATE.json"

    rows = []
    for p, (name, depth, track) in enumerate(zip(feat["names"], feat["depths"], feat["tracks"])):
        r = evaluate(F[:, p], spk, sent, cfg.folds, cfg.seed, cfg.content_probe, not cfg.no_probe)
        rows.append(dict(model=feat["model"], point=name, track=track, depth=depth, **r))
        print(f"  {name:>11}  acoustic_share {r['acoustic_share']:.3f}  spk_acc {r['spk_acc']:.3f}", flush=True)
    for name, X in base.items():
        r = evaluate(X, spk, sent, cfg.folds, cfg.seed, cfg.content_probe, not cfg.no_probe)
        rows.append(dict(model="baseline", point=name, track="baseline", depth="", **r))
        print(f"  {name:>11}  acoustic_share {r['acoustic_share']:.3f}  spk_acc {r['spk_acc']:.3f}", flush=True)

    cfg.out.mkdir(parents=True, exist_ok=True)
    with open(cfg.out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    write_summary(cfg.out / "summary.md", rows, feat["model"], len(spk), len(spk_set), len(sent_set))
    plot(cfg.out / "curves.png", rows, feat["model"], len(spk_set))
    write_json(cfg.json, rows, feat["model"], cfg.features, len(spk), len(spk_set), len(sent_set))
    print(f"wrote {cfg.out}/metrics.csv, summary.md")
    tracked = cfg.json.parent.resolve() == RESULTS_DIR.resolve()
    print(f"wrote {cfg.json}" + ("  <- commit and push this" if tracked else "  (not for pushing)"))


def write_json(path, rows, model, features, n, n_spk, n_sent):
    clean = lambda v: None if isinstance(v, float) and v != v else v  # noqa: E731  NaN -> null
    out = {
        "model": model,
        "features": str(features),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_clips": n, "n_speakers": n_spk, "n_sentences": n_sent,
        "chance_speaker_probe": 1 / n_spk,
        "acoustic_share_method": "omega2: chance floor from residual variance (k - 1) * ms_resid",
        "points": [{k: clean(r[k]) for k in COLS if k != "model"}
                   for r in rows if r["track"] != "baseline"],
        "baselines": [{k: clean(r[k]) for k in COLS if k not in ("model", "track", "depth")}
                      for r in rows if r["track"] == "baseline"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- report

def write_summary(path, rows, model, n, n_spk, n_sent):
    pts = [r for r in rows if r["track"] != "baseline"]
    base = [r for r in rows if r["track"] == "baseline"]
    f = lambda v: "-" if v != v else f"{v:.3f}"  # noqa: E731
    lines = [f"# Acoustic vs semantic per layer: {model}", "",
             f"- {n} clips: {n_spk} speakers x {n_sent} sentences",
             f"- chance: speaker probe {1 / n_spk:.3f}",
             "- acoustic_share: 1 = the layer encodes only the voice, 0 = only the words", "",
             "| point | track | depth | acoustic_share | eta_spk | eta_sent | spk_acc | spk_index |",
             "|---|---|---|---|---|---|---|---|"]
    for r in pts:
        lines.append(f"| {r['point']} | {r['track']} | {r['depth']} | {f(r['acoustic_share'])} | "
                     f"{f(r['eta_spk'])} | {f(r['eta_sent'])} | {f(r['spk_acc'])} | {f(r['spk_index'])} |")
    trunk = [r for r in pts if r["track"] == "trunk"]
    if trunk:
        hi = max(trunk, key=lambda r: r["acoustic_share"] if r["acoustic_share"] == r["acoustic_share"] else -1)
        lo = min(trunk, key=lambda r: r["acoustic_share"] if r["acoustic_share"] == r["acoustic_share"] else 2)
        lines += ["", f"- most acoustic trunk point: **{hi['point']}** ({f(hi['acoustic_share'])})",
                  f"- most semantic trunk point: **{lo['point']}** ({f(lo['acoustic_share'])})"]
    ends = {r["track"]: r for r in pts}            # last point of each track
    if len(ends) > 1:
        lines += ["", "| track end | acoustic_share | spk_acc |", "|---|---|---|"]
        for t, r in ends.items():
            lines.append(f"| {r['point']} | {f(r['acoustic_share'])} | {f(r['spk_acc'])} |")
    if base:
        lines += ["", "## Baselines", "", "| feature | acoustic_share | eta_spk | eta_sent | spk_acc |",
                  "|---|---|---|---|---|"]
        for r in base:
            lines.append(f"| {r['point']} | {f(r['acoustic_share'])} | {f(r['eta_spk'])} | "
                         f"{f(r['eta_sent'])} | {f(r['spk_acc'])} |")
    lines += ["", "Variance shares are measured on utterance means pooled over the user's speech;",
              "'semantic' here means sentence identity and 'acoustic' means speaker identity.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot(path, rows, model, n_spk):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped curves.png")
        return
    pts = [r for r in rows if r["track"] != "baseline"]
    base = [r for r in rows if r["track"] == "baseline"]
    by = {r["point"]: r for r in pts}
    colors = {"trunk": "#2a78d6", "control": "#eb6834", "speech": "#1baf7a", "merge": "#4a3aa7"}

    def track_line(track):
        rs = [r for r in pts if r["track"] == track]
        if not rs or track == "trunk":
            return rs
        start = {"control": rs[0]["depth"] - 1, "speech": rs[0]["depth"] - 1}.get(track)
        if track == "merge":
            prev = [r for r in pts if r["track"] == "speech"]
            return prev[-1:] + rs
        root = by.get(f"trunk_{start}")
        return ([root] if root else []) + rs

    panels = [("acoustic_share", "acoustic share  (speaker / (speaker + sentence))", (0, 1)),
              ("excess_spk", "variance explained above chance", None),
              ("spk_acc", "speaker probe accuracy", (0, 1.02))]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (key, label, ylim) in zip(axes, panels):
        for track in ("trunk", "control", "speech", "merge"):
            rs = track_line(track)
            if rs:
                ax.plot([r["depth"] for r in rs], [r[key] for r in rs], marker="o", ms=3,
                        color=colors[track], label=track)
                if key == "excess_spk":
                    ax.plot([r["depth"] for r in rs], [r["excess_sent"] for r in rs], ls="--",
                            color=colors[track], alpha=.7, label=f"{track} (sentence)")
        for r in base:
            if r[key] == r[key]:
                ax.axhline(r[key], color="gray", ls=":", lw=.9)
                ax.annotate(r["point"], (0, r[key]), fontsize=7, color="gray", va="bottom")
        if key == "spk_acc":
            ax.axhline(1 / n_spk, color="k", lw=.8)
        ax.set(xlabel="depth (layers from the input embedding)", ylabel=label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle(model)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
