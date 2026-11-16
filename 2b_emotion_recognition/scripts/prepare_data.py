"""Build the metadata table and cache the features used by all three pipelines.

    python scripts/prepare_data.py --tess "<path to 'TESS Toronto emotional speech set data'>" --cache <folder>

Writes
    data/metadata.csv                 path-independent table: speaker, word, emotion, text  (+ file name)
    <cache>/speech_logmel.npz         (N, 300, 80) float16 log-mel + valid lengths
    <cache>/text_distilbert.npz       frozen DistilBERT last-layer states per clip text: (N, L, 768) + mask
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import MAX_FRAMES, N_MELS, build_metadata, pad_or_crop, save_metadata, speech_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--skip_text", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

    meta = build_metadata(args.tess)
    print("clips after de-duplication:", len(meta))
    from collections import Counter
    print("per speaker:", dict(Counter(m["speaker"] for m in meta)), "| per emotion:", dict(Counter(m["emotion"] for m in meta)))
    out = []
    for m in meta:
        d = dict(m)
        d["file"] = os.path.basename(d.pop("path"))
        out.append(d)
    import csv
    with open(os.path.join(ROOT, "data", "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "word", "emotion", "text"])
        w.writeheader()
        w.writerows(out)

    # ----- speech: trimmed, normalised 16 kHz audio -> 80-bin log-mel, 10 ms hop
    t0 = time.time()
    X = np.zeros((len(meta), MAX_FRAMES, N_MELS), np.float16)
    L = np.zeros(len(meta), np.int32)
    for i, m in enumerate(meta):
        f = speech_features(m["path"])
        x, mask = pad_or_crop(f)
        X[i] = x
        L[i] = mask.sum()
        if (i + 1) % 500 == 0:
            print("  speech features %d/%d (%.0fs)" % (i + 1, len(meta), time.time() - t0), flush=True)
    np.savez_compressed(os.path.join(args.cache, "speech_logmel.npz"), X=X, lengths=L)
    print("speech: frames per clip after trimming: mean %.0f, max %d (limit %d)" % (L.mean(), L.max(), MAX_FRAMES))

    # ----- text: frozen DistilBERT contextual embeddings of the carrier phrase
    if not args.skip_text:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert = AutoModel.from_pretrained("distilbert-base-uncased").eval()
        texts = [m["text"] for m in meta]
        uniq = sorted(set(texts))
        feats = {}
        with torch.no_grad():
            for t in uniq:
                enc = tok(t, return_tensors="pt", padding="max_length", truncation=True, max_length=10)
                feats[t] = (bert(**enc).last_hidden_state[0].numpy().astype(np.float16), enc["attention_mask"][0].numpy().astype(bool),
                            tok.convert_ids_to_tokens(enc["input_ids"][0]))
        T = np.stack([feats[t][0] for t in texts]); M = np.stack([feats[t][1] for t in texts])
        np.savez_compressed(os.path.join(args.cache, "text_distilbert.npz"), X=T, mask=M)
        print("text: %d unique texts, tokens e.g. %s, array %s" % (len(uniq), feats[uniq[0]][2], T.shape))
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
