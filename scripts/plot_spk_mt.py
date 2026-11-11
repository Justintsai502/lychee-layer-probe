"""Plot a speaker-head + dialogue-loss run from the compact summary printed on the server.

  python scripts/plot_spk_mt.py results/spk/mt/summary.txt      # -> results/spk/mt/train_curves.png (+ eval_curves.png)

Summary lines: "T <step>,..." (10-step means of train.jsonl, header "TRAIN step,...") and optionally
"E <step>,... | layer weights" (eval.jsonl, header "EVAL step,..."). Validation points are drawn on the
matching train panels.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
RESUME = 200


def parse(path):
    tk, ek, T, E = None, None, [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("TRAIN "):
            tk = line.split(" ", 1)[1].split(",")
        elif line.startswith("EVAL "):
            ek = line.split(" ", 1)[1].split("|")[0].strip().split(",")
        elif line.startswith("T "):
            T.append([float(x) for x in line[2:].split(",")])
        elif line.startswith("E "):
            E.append([float(x) for x in line[2:].split("|")[0].strip().split(",")])
    t = {k: np.array(v) for k, v in zip(tk, np.array(T).T)}
    e = {k: np.array(v) for k, v in zip(ek, np.array(E).T)} if E else {}
    return t, e


def smooth(y, k=5):
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.axvline(RESUME, color=INK2, lw=0.8, ls=(0, (3, 3)), alpha=0.6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1210)


def line(ax, x, y, color, label):
    ax.plot(x, y, color=color, lw=1, alpha=0.3)
    ax.plot(x, smooth(y), color=color, lw=2, label=label)


def main(path):
    t, e = parse(path)
    x = t["step"]
    out = Path(path).parent
    panels = [
        ("Speaker head: diar loss", [("diar_sort", BLUE, "sort"), ("diar_pil", ORANGE, "PIL")], None, None),
        ("Contrastive loss / chance", [("con_rel", BLUE, "train")], None, None),
        ("Agent text CE", [("dlg_text", BLUE, "train")], "text", None),
        ("Agent speech-token CE", [("dlg_stoken", BLUE, "train")], "stoken", None),
        ("Control CE", [("dlg_control", BLUE, "train")], "control", None),
        ("Control accuracy", [("ctrl_acc", BLUE, "train")], "ctrl_acc", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), facecolor=SURF)
    for ax, (title, series, ekey, _) in zip(axes.flat, panels):
        style(ax, title)
        for key, color, label in series:
            line(ax, x, t[key], color, label)
        if ekey and e and ekey in e and np.isfinite(e[ekey]).any():
            ax.plot(e["step"], e[ekey], color=ORANGE, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label="val")
        if len([l for l in ax.get_lines() if not l.get_label().startswith('_')]) > 1:
            ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="best")
    for ax in axes[1]:
        ax.set_xlabel("optimizer step (8 crops each)", fontsize=8.5, color=INK2)
    fig.suptitle("lychee_c5_id4 + dialogue loss on AMI / IF / IF-SD (trunk L14, LoRA, 1200 steps)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.005, "thin = 10-step mean, bold = 50-step rolling mean; dashed line = resume at step 200",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out / "train_curves.png", dpi=150)
    print(f"wrote {out / 'train_curves.png'}")

    if not e:
        return
    lw, final = {}, {}
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("LW "):
            k, v = ln[3:].split(":", 1)
            lw[k.strip()] = np.array([float(x) for x in v.split()])
        elif ln.startswith("FINAL:"):          # "self single 0.725 change 0.563 der 0.364 | gold single ..."
            for part in ln[6:].split("|"):
                w = part.split()
                if w and w[0] in ("self", "gold", "empty"):
                    final[w[0] + "_single"], final[w[0] + "_change"] = float(w[2]), float(w[4])
                    if w[0] == "self":
                        final["der"] = float(w[6])
                elif w and w[0] == "rotation_follow":
                    final["rot_follow"] = float(w[1])
                elif w and w[0] == "ctrl":
                    final["start_acc"] = float(w[-1])
    xs = e["step"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), facecolor=SURF)

    def pts(ax, key, color, label):
        ax.plot(xs, e[key], color=color, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=label)
        if key in final:                       # full-val number of the final checkpoint, drawn off the curve
            ax.plot([1260], [final[key]], marker="D", ms=7, color=color, mec=SURF, mew=1.5, ls="none")

    for ax, title, keys in (
            (axes[0, 0], "Speaker acc, single-speaker frames", "single"),
            (axes[0, 1], "Speaker acc, at speaker changes", "change")):
        style(ax, title)
        for pre, c, lab in (("self", BLUE, "self-fill cache"), ("gold", ORANGE, "gold cache"), ("empty", AQUA, "empty cache")):
            pts(ax, f"{pre}_{keys}", c, lab)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    style(axes[0, 2], "DER (self-fill), lower is better")
    pts(axes[0, 2], "der", BLUE, "der")
    style(axes[1, 0], "Rotation follow (> 0 = follows cache)")
    axes[1, 0].axhline(0, color=INK2, lw=0.8)
    pts(axes[1, 0], "rot_follow", BLUE, "rot")
    style(axes[1, 1], "Control accuracy at turn starts")
    pts(axes[1, 1], "start_acc", BLUE, "start")
    for ax in list(axes[0]) + [axes[1, 0], axes[1, 1]]:
        ax.set_xlim(0, 1300)
        ax.set_xlabel("optimizer step", fontsize=8.5, color=INK2)
    ax = axes[1, 2]
    style(ax, "Readout layer weights (step 1200)")
    ax.lines[0].remove()                       # no resume marker here
    if lw:
        for k, c in (("enc", BLUE), ("llm", ORANGE)):
            if k in lw:
                w = lw[k]
                ax.plot(np.linspace(0, 1, len(w)), w * len(w), color=c, lw=2, marker="o", ms=4, mec=SURF,
                        label=f"{k} ({len(w)} layers)")
        ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("relative depth (0 = first layer)", fontsize=8.5, color=INK2)
        ax.set_ylabel("weight x n_layers (1 = uniform)", fontsize=8.5, color=INK2)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.suptitle("Validation during training (first 16 val recordings, 60 s each)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.text(0.01, 0.005, "circles = every 50 steps on the 16-clip subset; diamond = final checkpoint on all of val",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out / "eval_curves.png", dpi=150)
    print(f"wrote {out / 'eval_curves.png'}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/spk/mt/summary.txt")
