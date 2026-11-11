"""Per-variant accuracy of the diag battery (lyspk/diag.py), every live cache and fixed bank, AMI and IF apart.

  python scripts/plot_spk_banks.py results/spk/mt/diag_step1200   # -> bank_acc.png in that dir

Dots are pooled accuracies (sum ok / sum n, as summary.json); bars are 95% intervals from
resampling recordings (per_window.json). Dashed lines mark each subset's empty-cache level.
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
GROUPS = [
    ("即時 cache", [("self", "自己寫入"), ("gold", "正解 cache"), ("rotated", "正解但 slot 對調"),
                  ("silenced", "self，拿掉 agent 段")]),
    ("沒有 cache", [("empty", "空的")]),
    ("同一人、取樣位置不同", [("first", "最早的幀"), ("one turn", "單一段連續話"), ("recent", "最近的幀"),
                    ("late half", "後半段散取"), ("spread", "全段散取")]),
    ("對照組", [("clean", "只取段落中間"), ("edge", "只取交界/重疊旁"), ("one xK", "一幀複製 K 次"),
             ("one only", "只有一幀"), ("any voice", "任何人的聲音"), ("wrong", "給成別人的聲音")]),
]
SUBS = [("ami", "AMI", BLUE), ("tt", "IF", ORANGE)]
METRICS = [("frame", "frame_ok", "n_single", "單人 frame"), ("turn", "turn_mode_ok", "turn_n", "turn（mode）"),
           ("change", "change_ok", "change_n", "轉換 frame")]


def pooled_ci(rows, ok, n, boot=4000, seed=0):
    groups = sorted({r["group"] for r in rows})
    o = np.array([sum(r[ok] for r in rows if r["group"] == g) for g in groups], float)
    m = np.array([sum(r[n] for r in rows if r["group"] == g) for g in groups], float)
    if m.sum() == 0:
        return None, None, None
    idx = np.random.default_rng(seed).integers(0, len(groups), (boot, len(groups)))
    den = m[idx].sum(1)
    rs = o[idx].sum(1)[den > 0] / den[den > 0]
    return o.sum() / m.sum(), np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def main(d):
    d = Path(d)
    s = json.loads((d / "summary.json").read_text())
    pw = json.loads((d / "per_window.json").read_text())
    rows = []  # (y, key, label) top to bottom with gaps between groups
    y = 0.0
    heads = []
    for gname, items in GROUPS:
        heads.append((y, gname))
        y += 0.9
        for key, lab in items:
            rows.append((y, key, f"{key}  ·  {lab}"))
            y += 1
        y += 0.5
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.36 * y + 1.6), sharey=True, facecolor=SURF)
    for ax, (mname, ok, n, title) in zip(axes, METRICS):
        ax.set_facecolor(SURF)
        ax.grid(axis="x", color=GRID, lw=0.8)
        for j, (sub, lab, col) in enumerate(SUBS):
            e = pooled_ci([r for r in pw["empty"] if r["dataset"] == sub], ok, n)[0]
            if e is not None:
                ax.axvline(e, color=col, lw=1, ls=(0, (3, 3)), alpha=0.6)
            for yy, key, _ in rows:
                v, lo, hi = pooled_ci([r for r in pw[key] if r["dataset"] == sub], ok, n)
                if v is None:
                    continue
                yo = yy + (-0.17 if j == 0 else 0.17)
                ax.plot([lo, hi], [yo, yo], color=col, lw=2, alpha=0.45, solid_capstyle="round")
                ax.plot(v, yo, "o", ms=6.5, color=col, mec=SURF, mew=1.2, label=lab if key == "self" else None)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylim(y - 0.3, -0.6)
    axes[0].set_yticks([r[0] for r in rows])
    axes[0].set_yticklabels([r[2] for r in rows], fontsize=8.8, color=INK)
    for yy, g in heads:
        axes[0].annotate(g, (0, yy), xycoords=("axes fraction", "data"), xytext=(-8, 0), textcoords="offset points",
                         ha="right", va="center", fontsize=9.5, color=INK2, fontweight="bold")
    axes[2].legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="lower right")
    nb = s["n_groups_by"]
    fig.suptitle(f"各 cache / bank 下的正確率，step {s['step']}（{nb.get('ami', 0)} 場 AMI 會議、"
                 f"{nb.get('tt', 0)} 個 IF 場景，theta {s['theta_act']}）", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.01, "點 = 合併正確率；橫條 = 對錄音重抽 4000 次的 95% 區間；虛線 = 該資料集 empty 的水準。"
             "bank 都是 oracle（可取到整窗任何位置），只比 bank 之間，不和即時 cache 比絕對值。",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.96))
    out = d / "bank_acc.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/spk/mt/diag_step1200")
