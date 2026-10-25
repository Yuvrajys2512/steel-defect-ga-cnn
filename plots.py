"""
Figures: GA progress, GA vs random search, the winning architecture, confusion matrix,
and a grid of test predictions. All functions return matplotlib Figures (no pyplot
global state, so the web demo can create them repeatedly without leaking memory).
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from data import PRETTY

BLUE, LIGHT, ORANGE, GREY = "#1565c0", "#90caf9", "#ef6c00", "#9e9e9e"


def ga_progress(generations, title="Genetic algorithm: validation accuracy per generation"):
    """generations: list of {"gen", "accs": [...], "best_acc", "mean_acc"}."""
    fig = Figure(figsize=(8, 4.2))
    ax = fig.subplots()
    rng = np.random.default_rng(0)
    for g in generations:
        a = np.array(g["accs"]) * 100
        ax.scatter(g["gen"] + rng.uniform(-0.15, 0.15, len(a)), a, s=10, color=GREY, alpha=0.45,
                   label="individual" if g["gen"] == 0 else None, zorder=1)
    x = [g["gen"] for g in generations]
    ax.plot(x, [g["mean_acc"] * 100 for g in generations], "-o", color=LIGHT, lw=2, ms=4,
            label="population average", zorder=2)
    ax.plot(x, [g["best_acc"] * 100 for g in generations], "-o", color=BLUE, lw=2.5, ms=5,
            label="best individual", zorder=3)
    b = generations[-1]["best_acc"] * 100
    ax.annotate(f"{b:.1f}%", (x[-1], b), textcoords="offset points", xytext=(0, 8), ha="center",
                color=BLUE, fontweight="bold")
    lo = min(min(g["accs"]) for g in generations) * 100
    ax.set_ylim(max(0, min(lo, generations[0]["mean_acc"] * 100) - 5), 101)
    ax.set_xlabel("generation  (0 = random population)")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def search_comparison(ga_trace, rs_trace):
    """Best-so-far validation accuracy vs number of networks trained: GA vs random search."""
    fig = Figure(figsize=(7, 3.8))
    ax = fig.subplots()
    for trace, name, col in ((ga_trace, "genetic algorithm", BLUE), (rs_trace, "random search", ORANGE)):
        acc = np.maximum.accumulate(np.array(trace) * 100)
        ax.plot(np.arange(1, len(acc) + 1), acc, color=col, lw=2.2, label=f"{name} (best {acc[-1]:.1f}%)")
        ax.scatter(np.arange(1, len(trace) + 1), np.array(trace) * 100, s=6, color=col, alpha=0.25)
    ax.set_ylim(max(0, min(min(ga_trace), min(rs_trace)) * 100 - 3), 101)
    ax.set_xlabel("networks trained (same compute budget)")
    ax.set_ylabel("best validation accuracy so far (%)")
    ax.set_title("GA vs random search")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def architecture(genome, img=64, n_classes=6, params=None, title="Winning architecture"):
    """Layer diagram: box height ~ spatial size, box width ~ number of channels."""
    layers = [("Input", f"1 @ {img}x{img}", img, 1, "#b0bec5")]
    s = img
    for i, f in enumerate(genome.used, 1):
        layers.append((f"Conv {genome.kernel}x{genome.kernel}\n+BN+ReLU", f"{f} @ {s}x{s}", s, f, "#1e88e5"))
        s //= 2
        layers.append(("MaxPool\n2x2", f"{f} @ {s}x{s}", s, f, "#26a69a"))
    c = genome.used[-1]
    layers.append(("Global\navg pool", f"{c}", 1, c, "#7e57c2"))
    if genome.dense:
        layers.append((f"Dense\n+ReLU", f"{genome.dense}", 1, genome.dense, "#fb8c00"))
    layers.append(("Softmax", f"{n_classes} classes", 1, n_classes, "#e53935"))

    fig = Figure(figsize=(max(8, 1.15 * len(layers)), 4.0))
    ax = fig.subplots()
    x = 0.0
    for name, shape, sp, ch, col in layers:
        h = 0.6 + 2.4 * np.log2(max(sp, 1)) / np.log2(img)
        w = 0.25 + 0.12 * np.log2(max(ch, 1))
        ax.add_patch(FancyBboxPatch((x, -h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=col, ec="#263238", lw=1, alpha=0.9))
        ax.text(x + w / 2, 1.85, name, ha="center", va="bottom", fontsize=7.5)
        ax.text(x + w / 2, -1.85, shape, ha="center", va="top", fontsize=7, color="#37474f")
        x += w + 0.45
    ax.annotate("", xy=(x - 0.3, -2.6), xytext=(0, -2.6), arrowprops=dict(arrowstyle="->", color=GREY))
    ax.text(x / 2, -2.85, "S-cell layers (conv) and C-cell layers (pool) alternate, as in Fukushima's Neocognitron",
            ha="center", va="top", fontsize=7.5, color="#546e7a", style="italic")
    ax.set_xlim(-0.3, x)
    ax.set_ylim(-3.3, 2.6)
    ax.axis("off")
    sub = genome.describe() + (f" | {params:,} parameters" if params else "")
    ax.set_title(f"{title}\n{sub}", fontsize=9.5)
    fig.tight_layout()
    return fig


def confusion(cm, title="Test-set confusion matrix"):
    cm = np.asarray(cm)
    fig = Figure(figsize=(5.6, 4.8))
    ax = fig.subplots()
    ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(len(PRETTY)), PRETTY, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(PRETTY)), PRETTY, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    acc = np.trace(cm) / cm.sum() * 100
    ax.set_title(f"{title} ({acc:.1f}% correct)", fontsize=10)
    fig.tight_layout()
    return fig


def prediction_grid(images, true, pred, conf, cols=6, title="Predictions on held-out test images"):
    rows = int(np.ceil(len(images) / cols))
    fig = Figure(figsize=(2.0 * cols, 2.35 * rows), layout="constrained")     # keeps titles off the row above
    axes = np.atleast_1d(fig.subplots(rows, cols)).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, im, t, p, c in zip(axes, images, true, pred, conf):
        ax.imshow(im, cmap="gray")
        ok = t == p
        ax.set_title(f"{PRETTY[p]} {c * 100:.0f}%" + ("" if ok else f"\n(true: {PRETTY[t]})"),
                     fontsize=8, color="#2e7d32" if ok else "#c62828")
    fig.suptitle(title, fontsize=10)
    return fig
