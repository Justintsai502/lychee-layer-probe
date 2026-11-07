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
