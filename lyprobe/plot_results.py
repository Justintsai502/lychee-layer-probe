#!/usr/bin/env python3
"""
Plot the committed result JSONs (results/*.json) - runs on a laptop, no model needed.

  python lyprobe/plot_results.py                       # every results/*.json
  python lyprobe/plot_results.py results/lychee.json   # one file

Writes docs/fig_acoustic_share.png and docs/fig_speaker_probe.png.
Lychee's tree is drawn as trunk + branches (control from layer 20, speech + merge from
layer 24); other models (BayLing) as a single trunk.

Hard deps: numpy, matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

TRACK_COLORS = {"trunk": "#2a78d6", "control": "#eb6834", "speech": "#1baf7a", "merge": "#4a3aa7"}
OTHER_MODEL = "#7c7b76"


def track_lines(points):
    """{track: [points in order]} with each branch starting at the trunk point it reads."""
    by = {p["point"]: p for p in points}
    tracks = {}
    for p in points:
        tracks.setdefault(p["track"], []).append(p)
    for t, ps in tracks.items():
        if t in ("control", "speech"):
            root = by.get(f"trunk_{ps[0]['depth'] - 1}")
            if root:
                tracks[t] = [root] + ps
    if "merge" in tracks and "speech" in tracks:
        tracks["merge"] = [tracks["speech"][-1]] + tracks["merge"]
    return tracks


def draw(ax, results, key, ylabel):
    for res in results:
        lines = track_lines(res["points"])
        is_tree = len(lines) > 1
        for t, ps in lines.items():
            xs = [p["depth"] for p in ps]
            ys = [p[key] for p in ps]
            if all(v is None for v in ys):
                continue
            color = TRACK_COLORS.get(t, OTHER_MODEL) if is_tree else OTHER_MODEL
            label = f"{res['model']} {t}" if is_tree else res["model"]
            ax.plot(xs, ys, marker="o", ms=3, lw=2, color=color, label=label,
                    ls="-" if is_tree else "--")
    base = results[0].get("baselines", [])
    for b in base:
        if b.get(key) is not None:
            ax.axhline(b[key], color="#9a998f", ls=":", lw=1)
            ax.annotate(b["point"], (0.2, b[key]), fontsize=8, color="#7c7b76", va="bottom")
    if key == "spk_acc":
        ax.axhline(results[0]["chance_speaker_probe"], color="k", lw=.8)
        ax.annotate("chance", (0.2, results[0]["chance_speaker_probe"]), fontsize=8, va="bottom")
    tree = next((r for r in results if len(track_lines(r["points"])) > 1), None)
    if tree:
        for t in ("control", "speech"):
            ps = [p for p in tree["points"] if p["track"] == t]
            if ps:
                ax.axvline(ps[0]["depth"] - 1, color=TRACK_COLORS[t], lw=.8, alpha=.4)
    ax.set_xlabel("depth (layers from the input embedding)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="best")


def head_series(res):
    """The three 4-layer heads aligned on 'layers into the head' (0 = the branch input).

    text    = trunk layers after the speech branch point (the text head reads trunk end)
    speech  = stoken layers, then merge layers continuing at 5..8
    control = control layers; its input is an EARLIER trunk layer than the other two
    """
    by = {p["point"]: p for p in res["points"]}
    sp = [p for p in res["points"] if p["track"] == "speech"]
    ct = [p for p in res["points"] if p["track"] == "control"]
    mg = [p for p in res["points"] if p["track"] == "merge"]
    s_at, c_at = sp[0]["depth"] - 1, ct[0]["depth"] - 1
    n_trunk = max(p["depth"] for p in res["points"] if p["track"] == "trunk")
    text = [by[f"trunk_{d}"] for d in range(s_at, n_trunk + 1)]
    return {
        f"text head (trunk {s_at + 1}-{n_trunk})": ("trunk", text),
        f"speech head (from trunk {s_at})": ("speech", [by[f"trunk_{s_at}"]] + sp),
        "merge (after speech)": ("merge", [sp[-1]] + mg),
        f"control head (from trunk {c_at})": ("control", [by[f"trunk_{c_at}"]] + ct),
    }, s_at, c_at


def plot_heads(plt, res, path):
    series, s_at, c_at = head_series(res)
    panels = [("acoustic_share", "acoustic share  (1 = voice only)"),
              ("excess_spk", "voice: variance explained by speaker"),
              ("excess_sent", "content: variance explained by sentence"),
              ("spk_acc", "speaker probe accuracy")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, (key, ylabel) in zip(axes, panels):
        for label, (track, ps) in series.items():
            start = len(series[next(iter(series))][1]) - 1 if track == "merge" else 0
            xs = list(range(start, start + len(ps)))
            ax.plot(xs, [p[key] for p in ps], marker="o", ms=4, lw=2, color=TRACK_COLORS[track],
                    ls="--" if track == "merge" else "-", label=label)
        if key == "spk_acc":
            ax.axhline(res["chance_speaker_probe"], color="k", lw=.8)
        ax.axvline(4, color="#9a998f", lw=.8, ls=":")
        ax.set_xticks(range(0, 9))
        ax.set_xlabel("layers into the head  (0 = branch input)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{res['model']}: the three 4-layer heads side by side  "
                 f"(text & speech read trunk {s_at}, control reads trunk {c_at})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def plot_cosine(plt, results, path):
    """Row 1: mean cosine of the three pair types per layer (z-scored features).
    Row 2: zoom on the same-speaker bonus, s_spk - s_none (how much more alike two clips
    of one voice are than two unrelated clips). Raw-feature cosines stay in the JSON but
    are not plotted: anisotropy puts every pair near 1 and s_spk on top of s_none."""
    if not all("s_spk" in r["points"][0] for r in results):
        print("result JSONs have no s_spk yet (re-run analysis on the server); skipped fig_cosine.png")
        return
    fig, axes = plt.subplots(2, len(results), figsize=(8 * len(results), 9), squeeze=False,
                             sharey="row")
    base = {b["point"]: b for b in results[0].get("baselines", []) if "s_spk" in b}
    for col, res in enumerate(results):
        lines = track_lines(res["points"])
        tree = len(lines) > 1
        trunk = [p for p in res["points"] if p["track"] == "trunk"]
        tx = [p["depth"] for p in trunk]
        ax, bx = axes[0][col], axes[1][col]
        for t, ps in lines.items():
            xs = [p["depth"] for p in ps]
            color = TRACK_COLORS.get(t, OTHER_MODEL) if tree else TRACK_COLORS["trunk"]
            lab = f" ({t})" if tree else ""
            ax.plot(xs, [p["s_spk"] for p in ps], marker="o", ms=3, lw=2, color=color,
                    label="same speaker, diff sentence" + lab)
            bx.plot(xs, [p["s_spk"] - p["s_none"] for p in ps], marker="o", ms=3, lw=2,
                    color=color, label=t if tree else res["model"])
        ax.plot(tx, [p["s_txt"] for p in trunk], ls="--", lw=1.6, color="#7c7b76",
                label="diff speaker, same sentence" + (" (trunk)" if tree else ""))
        ax.plot(tx, [p["s_none"] for p in trunk], ls=":", lw=1.6, color="#2b2b2b",
                label="diff speaker, diff sentence" + (" (trunk)" if tree else ""))
        ax.set_title(f"{res['model']}: mean cosine by pair type (z-scored features)", fontsize=10)
        ref = "   ".join(f"{k} {v['s_spk'] - v['s_none']:.2f}" for k, v in base.items())
        bx.set_title(f"{res['model']}: same-speaker bonus  s_spk - s_none\n"
                     f"(reference, off scale: {ref})", fontsize=9)
        for a in (ax, bx):
            a.axhline(0, color="k", lw=.6)
            a.set_xlabel("depth (layers from the input embedding)")
            a.grid(alpha=.3)
        ax.legend(fontsize=7, loc="center right")
        bx.legend(fontsize=7, loc="upper right")
    axes[0][0].set_ylabel("mean cosine similarity")
    axes[1][0].set_ylabel("cosine gain over unrelated pairs")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs")
    cfg = ap.parse_args()

    files = cfg.files or sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no result JSONs in {RESULTS_DIR}")
    results = [json.loads(Path(f).read_text()) for f in files]
    # the tree model first so its baselines and branch markers are used
    results.sort(key=lambda r: -len(track_lines(r["points"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.out.mkdir(parents=True, exist_ok=True)
    figs = [("acoustic_share", "acoustic share   (1 = voice only, 0 = words only)", (0, 1.02),
             "fig_acoustic_share.png"),
            ("spk_acc", "speaker probe accuracy", (0, 1.02), "fig_speaker_probe.png")]
    for key, ylabel, ylim, fname in figs:
        fig, ax = plt.subplots(figsize=(10, 5))
        draw(ax, results, key, ylabel)
        ax.set_ylim(*ylim)
        n = results[0]
        ax.set_title(f"{n['n_speakers']} speakers x {n['n_sentences']} sentences ({n['n_clips']} clips)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(cfg.out / fname, dpi=160)
        plt.close(fig)
        print(f"wrote {cfg.out / fname}")

    tree = next((r for r in results if len(track_lines(r["points"])) > 1), None)
    if tree:
        plot_heads(plt, tree, cfg.out / "fig_heads.png")
    plot_cosine(plt, results, cfg.out / "fig_cosine.png")

    # the two parts behind acoustic_share, side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, key, ylabel in ((axes[0], "excess_spk", "voice: variance explained by speaker"),
                            (axes[1], "excess_sent", "content: variance explained by sentence")):
        draw(ax, results, key, ylabel)
        ax.set_ylim(0, 1.0 if key == "excess_sent" else None)
    fig.suptitle("Above-chance share of each layer's variance (omega^2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(cfg.out / "fig_variance.png", dpi=160)
    plt.close(fig)
    print(f"wrote {cfg.out / 'fig_variance.png'}")


if __name__ == "__main__":
    sys.exit(main())
