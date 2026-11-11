"""Plot hspk_cos.json (scripts/hspk_cos.py): same- vs different-speaker cosine, and per-window AUC.

  python scripts/plot_hspk_cos.py results/spk/mt/diag_step1200/hspk_cos.json   # -> hspk_cos.png beside it
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Arial Unicode MS", "Heiti TC", "DejaVu Sans"]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SUBS = [("ami", "AMI", BLUE), ("tt", "IF", ORANGE)]


def style(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main(path):
    d = json.loads(Path(path).read_text())
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURF,
                                     gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})
    # 1-2: mean cosine same vs diff, frame and segment
    for ax, unit, title in ((a1, "frame", "單一 frame"), (a2, "segment", "0.96 秒 segment")):
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        for j, (sub, lab, col) in enumerate(SUBS):
            r = d["pooled"][sub][unit]
            y = j
            ax.plot([r["cos_diff"], r["cos_same"]], [y, y], color=col, lw=2.5, alpha=0.35)
            ax.plot(r["cos_diff"], y, "o", ms=9, color=SURF, mec=col, mew=2)
            ax.plot(r["cos_same"], y, "o", ms=9, color=col, mec=SURF, mew=1.2)
            ax.annotate(f"差 {r['gap']:.2f}   AUC {r['auc']:.2f}", ((r["cos_diff"] + r["cos_same"]) / 2, y),
                        xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9, color=INK2)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([s[1] for s in SUBS], fontsize=10.5, color=INK)
        ax.set_ylim(1.6, -0.6)
        ax.set_xlim(0, 1)
        ax.set_title(f"{title}：平均 cos", loc="left", fontsize=11, color=INK)
    a1.plot([], [], "o", color=INK2, label="同一人")
    a1.plot([], [], "o", color=SURF, mec=INK2, mew=2, label="不同人")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
    # 3: per-window segment AUC
    style(a3)
    a3.grid(axis="y", color=GRID, lw=0.8)
    a3.axhline(0.5, color=INK2, lw=0.9, ls=(0, (3, 3)))
    rng = np.random.default_rng(0)
    for j, (sub, lab, col) in enumerate(SUBS):
        aucs = [w["segment"]["auc"] for w in d["per_window"] if w["dataset"] == sub and w["segment"]["auc"] is not None]
        x = j + rng.uniform(-0.12, 0.12, len(aucs))
        a3.plot(x, aucs, "o", ms=8, color=col, mec=SURF, mew=1.2, alpha=0.9)
        med = float(np.median(aucs))
        a3.plot([j - 0.25, j + 0.25], [med, med], color=INK, lw=2)
        a3.annotate(f"中位數 {med:.2f}（{len(aucs)} 窗）", (j + 0.27, med), va="center", fontsize=9, color=INK2)
    a3.set_xticks([0, 1])
    a3.set_xticklabels([s[1] for s in SUBS], fontsize=10.5, color=INK)
    a3.set_xlim(-0.5, 1.9)
    a3.set_ylim(0.35, 1.02)
    a3.set_title("每個窗內的 segment AUC（0.5 = 分不開）", loc="left", fontsize=11, color=INK)
    fig.suptitle(f"head 輸入特徵（readout，{d['dim']} 維）的說話者 cos 相似度，只在同一窗內比較",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.01, "AUC = 隨機抽一對同一人、一對不同人，同一人 cos 較高的機率。少於 2 位單人說話者的窗沒有 AUC。",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = Path(path).with_name("hspk_cos.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/spk/mt/diag_step1200/hspk_cos.json")
