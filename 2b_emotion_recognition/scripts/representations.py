"""Separability of the emotion classes in the learned representations.

For the held-out test clips, the vectors produced by
    * the temporal-modelling block  (speech BiGRU + attention pooling),
    * the contextual-modelling block (text BiGRU + attention pooling),
    * the fusion block
are projected to 2-D with t-SNE and coloured by emotion (top row) and by speaker / target word (bottom row).
Separability is also quantified: silhouette score w.r.t. the emotion labels, and 5-fold k-NN accuracy.

    python scripts/representations.py --cache <cache> [--seed 0] [--fusion_mode concat]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import EMOTIONS  # noqa: E402
from common.engine import load_trained, predict  # noqa: E402

PALETTE = ["#d7301f", "#8c510a", "#6a51a3", "#fdae61", "#737373", "#1b9e77", "#2171b5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", default="word")
    ap.add_argument("--fusion_mode", default="concat")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    model, bundle, split, _ = load_trained("fusion", a.protocol, a.seed, a.cache, a.fusion_mode, a.device)
    idx = split["test"]
    _, reps = predict(model, bundle, idx, return_reps=True)
    y = bundle.y[idx].cpu().numpy()
    spk = np.array([bundle.meta[i]["speaker"] for i in idx])
    words = np.array([bundle.meta[i]["word"] for i in idx])
    blocks = [("temporal", "Temporal-modelling block (speech)"), ("contextual", "Contextual-modelling block (text)"), ("fusion", "Fusion block (%s)" % a.fusion_mode)]
    chance = 1 / 7

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    rows = []
    for c, (key, title) in enumerate(blocks):
        Z = StandardScaler().fit_transform(reps[key])
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
        sil_emo = silhouette_score(Z, y)
        two_spk = len(set(spk)) > 1            # cross-speaker protocols have a single speaker in the test set
        sil_spk = silhouette_score(Z, spk == "YAF") if two_spk else float("nan")
        sil_word = np.nan
        knn = cross_val_score(KNeighborsClassifier(5), Z, y, cv=5).mean()
        knn_spk = cross_val_score(KNeighborsClassifier(5), Z, spk, cv=5).mean() if two_spk else float("nan")
        rows.append(dict(block=key, silhouette_emotion=float(sil_emo), silhouette_speaker=float(sil_spk), knn5_emotion_acc=float(knn), knn5_speaker_acc=float(knn_spk)))
        for e in range(7):
            m = y == e
            axes[0, c].scatter(emb[m, 0], emb[m, 1], s=12, color=PALETTE[e], label=EMOTIONS[e], alpha=0.85, linewidths=0)
        axes[0, c].set_title("%s\nsilhouette(emotion) = %.2f   5-NN emotion acc. = %.2f" % (title, sil_emo, knn), fontsize=10)
        if two_spk:
            for s_name, col in (("OAF", "#4575b4"), ("YAF", "#d73027")):
                m = spk == s_name
                axes[1, c].scatter(emb[m, 0], emb[m, 1], s=12, color=col, label=s_name, alpha=0.8, linewidths=0)
            axes[1, c].set_title("same projection, coloured by speaker (5-NN speaker acc. = %.2f)" % knn_spk, fontsize=9)
        else:                                  # second row: same projection coloured by the word (shows what the text block encodes)
            for w_i, w in enumerate(sorted(set(words))):
                m = words == w
                axes[1, c].scatter(emb[m, 0], emb[m, 1], s=10, color=plt.cm.tab20(w_i % 20), alpha=0.8, linewidths=0)
            axes[1, c].set_title("same projection, coloured by target word (%d words)" % len(set(words)), fontsize=9)
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].legend(fontsize=8, markerscale=1.6, loc="best")
    if len(set(spk)) > 1:
        axes[1, 0].legend(fontsize=8, markerscale=1.6, loc="best")
    fig.suptitle("t-SNE of the learned representations of the %d test clips (%s protocol, seed %d); chance level for 7 classes = %.2f" % (len(idx), a.protocol, a.seed, chance), fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "Results", "representations"), exist_ok=True)
    out = os.path.join(ROOT, "Results", "representations", "tsne_blocks_%s_seed%d.png" % (a.protocol, a.seed))
    fig.savefig(out, dpi=130)
    json.dump(rows, open(os.path.join(ROOT, "Results", "representations", "separability_%s_seed%d.json" % (a.protocol, a.seed)), "w"), indent=1)
    print("saved", out)
    for r in rows:
        print("%-11s silhouette(emotion) %.3f | silhouette(speaker) %.3f | 5-NN emotion acc %.3f | 5-NN speaker acc %.3f" % (r["block"], r["silhouette_emotion"], r["silhouette_speaker"], r["knn5_emotion_acc"], r["knn5_speaker_acc"]))

    # the same three blocks from the unimodal models (text: separability by target WORD, which is all it can encode)
    from common.engine import load_trained as lt
    for modality, key in (("speech", "temporal"), ("text", "contextual")):
        m2, b2, sp2, _ = lt(modality, a.protocol, a.seed, a.cache, "concat", a.device)
        _, r2 = predict(m2, b2, sp2["test"], return_reps=True)
        Z = StandardScaler().fit_transform(r2[key])
        print("unimodal %-6s %-10s silhouette(emotion) %.3f | 5-NN emotion acc %.3f" % (modality, key, silhouette_score(Z, y), cross_val_score(KNeighborsClassifier(5), Z, y, cv=5).mean()))


if __name__ == "__main__":
    main()
