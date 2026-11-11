"""Forest plot of the diag battery's paired deltas (lyspk/diag.py summary.json), AMI and IF apart.

  python scripts/plot_spk_diag.py results/spk/mt/diag_step1200/summary.json   # -> diag_pairs.png beside it
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
PAIRS = [("gold-empty", "cache worth at all"), ("gold-wrong", "identity (banks)"),
         ("wrong-empty", "occupancy (banks)"), ("gold-rotated", "identity (live)"),
         ("self-gold", "cost of self-fill"), ("first-gold", "earliest vs latest"),
         ("spread-any voice", "this voice vs anyone"), ("one xK-empty", "one frame, copied"),
         ("clean-edge", "mid-turn vs handover")]
SUBS = [("ami", "AMI", BLUE), ("tt", "IF", ORANGE)]


def main(path):
    s = json.loads(Path(path).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.4), sharey=True, facecolor=SURF)
    n = len(PAIRS)
    for ax, metric in zip(axes, ("frame", "turn", "change")):
        ax.set_facecolor(SURF)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        for j, (sub, lab, col) in enumerate(SUBS):
            for i, (pair, _) in enumerate(PAIRS):
                g = s["paired"][sub][pair][metric]
                if g["delta"] is None:
                    continue
                y = n - 1 - i + (0.16 if j == 0 else -0.16)
                sig = g["lo"] is not None and (g["lo"] > 0 or g["hi"] < 0)
                if g["lo"] is not None:
                    ax.plot([g["lo"], g["hi"]], [y, y], color=col, lw=2, solid_capstyle="round")
                ax.plot(g["delta"], y, "o", ms=8, color=col if sig else SURF, mec=col, mew=2,
                        label=lab if i == 0 else None)
        ax.set_title(f"{metric} accuracy, A − B", loc="left", fontsize=10.5, color=INK)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"{p}\n{d}" for p, d in reversed(PAIRS)], fontsize=8.5, color=INK)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    nb = s["n_groups_by"]
    fig.suptitle(f"Cache ablation at step {s['step']} ({s['head_mode']}): {nb.get('ami', 0)} AMI meetings, "
                 f"{nb.get('tt', 0)} IF scenes", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "95% interval from 10,000 resamples of recordings; filled = interval excludes zero",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = Path(path).with_name("diag_pairs.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/spk/mt/diag_step1200/summary.json")
