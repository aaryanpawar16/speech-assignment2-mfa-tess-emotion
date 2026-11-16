"""Sanity check for the fusion code: a text modality that DOES carry (noisy) emotion information.

TESS text is the same carrier phrase for every emotion, so no fusion model can profit from it.  To check that the
fusion pipelines are able to exploit an informative text modality, each clip gets a second sentence that states an
emotion: "Say the word fat. I am terrified."  With probability P the stated emotion is the true one, otherwise it is a
different, randomly chosen emotion (so text-only accuracy is about P).  This is a SYNTHETIC experiment about the code,
not a TESS result.

    python scripts/make_synthetic_text.py --cache <cache> [--p 0.6]     # writes <cache>/text_synth_p060.npz and data/synthetic_text_p060.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import EMOTIONS, load_metadata  # noqa: E402

CUE = {"angry": "I am furious", "disgust": "That is disgusting", "fear": "I am terrified", "happy": "I am so happy",
       "neutral": "It is fine", "pleasant_surprise": "What a lovely surprise", "sad": "I am so sad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--p", type=float, default=0.6)
    a = ap.parse_args()
    meta = load_metadata(os.path.join(ROOT, "data", "metadata.csv"))
    rng = np.random.default_rng(0)
    texts, stated = [], []
    for m in meta:
        e = m["emotion"] if rng.random() < a.p else str(rng.choice([x for x in EMOTIONS if x != m["emotion"]]))
        stated.append(e)
        texts.append("%s. %s." % (m["text"].rstrip("."), CUE[e]))
    acc = np.mean([s == m["emotion"] for s, m in zip(stated, meta)])
    print("clips %d | fraction of clips whose stated emotion is the true one: %.3f | example: %r (true: %s)" % (len(meta), acc, texts[0], meta[0]["emotion"]))
    tag = "synth_p%03d" % round(a.p * 100)
    with open(os.path.join(ROOT, "data", "synthetic_text_p%03d.csv" % round(a.p * 100)), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "true_emotion", "stated_emotion", "text"])
        for m, s, t in zip(meta, stated, texts):
            w.writerow([m["file"], m["emotion"], s, t])
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    feats = {}
    t0 = time.time()
    with torch.no_grad():
        for t in sorted(set(texts)):
            enc = tok(t, return_tensors="pt", padding="max_length", truncation=True, max_length=16)
            feats[t] = (bert(**enc).last_hidden_state[0].numpy().astype(np.float16), enc["attention_mask"][0].numpy().astype(bool))
    X = np.stack([feats[t][0] for t in texts])
    M = np.stack([feats[t][1] for t in texts])
    np.savez_compressed(os.path.join(a.cache, "text_%s.npz" % tag), X=X, mask=M)
    print("saved text_%s.npz" % tag, X.shape, "| %d unique texts | %.0fs" % (len(set(texts)), time.time() - t0))


if __name__ == "__main__":
    main()
