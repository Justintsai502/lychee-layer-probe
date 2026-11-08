#!/usr/bin/env python3
"""
Plot results/*_eer.json -> docs/fig_eer.png (laptop, no model).

One panel per model. x = depth: negative = the audio encoder (before the LLM), 0 = the
LLM input embedding. y = speaker-verification EER (lower = more speaker information,
0.5 = chance). Solid = cosine (no training), dashed = WCCN backend on unseen speakers.

  python lyprobe/plot_eer.py

Hard deps: numpy, matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lyprobe.common import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from lyprobe.plot_results import TRACK_COLORS, track_lines  # noqa: E402

COLORS = dict(TRACK_COLORS, encoder="#c03a8b")
BASE_STYLE = {"ecapa": "#9a998f", "mimi_acoustic": "#1baf7a", "mimi_semantic": "#7c7b76",
              "mimi_all": "#4a3aa7"}


def lines_with_encoder(points):
    """Encoder points as their own track, joined to the LLM's first point."""
    enc = [p for p in points if p["track"] == "encoder"]
    llm = [p for p in points if p["track"] != "encoder"]
    lines = track_lines(llm) if llm else {}
    if enc:
        first = min((p for p in llm if p["track"] == "trunk"), key=lambda p: p["depth"], default=None)
        lines = {"encoder": enc + ([first] if first else []), **lines}
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs/fig_eer.png")
    cfg = ap.parse_args()
    files = cfg.files or sorted(RESULTS_DIR.glob("*_eer.json"))
    if not files:
        raise SystemExit(f"no *_eer.json in {RESULTS_DIR}")
    results = [json.loads(Path(f).read_text()) for f in files]
    results.sort(key=lambda r: -len(lines_with_encoder(r["points"])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(9 * len(results), 6.0), squeeze=False,
                             sharey=True)
    for ax, res in zip(axes[0], results):
        lines = lines_with_encoder(res["points"])
        for t, ps in lines.items():
            xs = [p["depth"] for p in ps]
            c = COLORS.get(t, "#2a78d6")
            ax.plot(xs, [100 * p["eer_cos"] for p in ps], marker="o", ms=3, lw=2, color=c,
                    label=f"{t}: cosine")
            ax.plot(xs, [100 * p["eer_wccn"] for p in ps], ls="--", lw=1.4, color=c,
                    label=f"{t}: WCCN, unseen speakers")
        for b in res.get("baselines", []):
            col = BASE_STYLE.get(b["point"], "#9a998f")
            ax.axhline(100 * b["eer_cos"], color=col, ls=":", lw=1)
            ax.annotate(f"{b['point']} {100 * b['eer_cos']:.1f}%", (0.995, 100 * b["eer_cos"]),
                        xycoords=("axes fraction", "data"), fontsize=7, color=col, ha="right",
                        va="top" if b["point"] == "mimi_acoustic" else "bottom")
        ax.axhline(50, color="k", lw=.8)
        ax.annotate("chance 50%", (0.005, 50), xycoords=("axes fraction", "data"), fontsize=8, va="bottom")
        if "encoder" in lines:
            ax.axvline(-0.5, color=COLORS["encoder"], lw=.8, alpha=.5)
            ax.annotate("audio encoder  |  LLM", (-0.5, 55), fontsize=8, ha="center",
                        color=COLORS["encoder"])
        ax.set_title(f"{res['model']}: speaker-verification EER per layer "
                     f"({res['n_speakers']} spk x {res['n_sentences']} sent)", fontsize=10)
        ax.set_xlabel("depth (negative = audio encoder, 0 = LLM input)")
        ax.set_ylim(0, 60)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False)
    axes[0][0].set_ylabel("EER %   (lower = more speaker information)")
    fig.tight_layout()
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.out, dpi=160)
    print(f"wrote {cfg.out}")


if __name__ == "__main__":
    sys.exit(main())
