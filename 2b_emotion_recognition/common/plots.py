"""Plot helpers (matplotlib only)."""
import os

import numpy as np

from common.data import EMOTIONS

SHORT = {"pleasant_surprise": "pleas.surp."}


def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1e-9)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = [SHORT.get(e, e) for e in EMOTIONS]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, "%.2f" % norm[i, j], ha="center", va="center", color="white" if norm[i, j] > 0.55 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
