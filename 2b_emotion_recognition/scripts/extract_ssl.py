"""Cache frozen wav2vec2 features as an alternative speech front-end (extra experiment).

For every clip: the same trimmed, peak-normalised 16 kHz audio as the log-mel pipeline -> wav2vec2-base (frozen) ->
the MEAN of the hidden states of layers 6-9 (the middle layers; chosen a priori from the literature on paralinguistic
information in self-supervised models, not tuned on any test result) -> (frames at 50 Hz, 768) per clip.

    python scripts/extract_ssl.py --tess <TESS folder> --cache <cache>        # writes <cache>/speech_w2v.npz
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import build_metadata, load_audio, preprocess_audio  # noqa: E402

MAX_FRAMES = 150          # 3 s at 50 Hz (same duration limit as the log-mel features)
LAYERS = (6, 7, 8, 9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    a = ap.parse_args()
    import torch
    from transformers import Wav2Vec2Model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2Model.from_pretrained(a.model).to(dev).eval()
    meta = build_metadata(a.tess)
    X = np.zeros((len(meta), MAX_FRAMES, 768), np.float16)
    L = np.zeros(len(meta), np.int32)
    t0 = time.time()
    with torch.no_grad():
        for i, m in enumerate(meta):
            y = preprocess_audio(load_audio(m["path"])).astype(np.float32)
            y = (y - y.mean()) / (y.std() + 1e-7)                       # the feature extractor of wav2vec2-base normalises each clip
            hs = model(torch.from_numpy(y)[None].to(dev), output_hidden_states=True).hidden_states
            h = torch.stack([hs[l] for l in LAYERS]).mean(0)[0].cpu().numpy()
            n = min(len(h), MAX_FRAMES)
            X[i, :n] = h[:n]
            L[i] = n
            if (i + 1) % 400 == 0:
                print("  %d/%d (%.0fs)" % (i + 1, len(meta), time.time() - t0), flush=True)
    np.savez_compressed(os.path.join(a.cache, "speech_w2v.npz"), X=X, lengths=L)
    print("saved speech_w2v.npz", X.shape, "frames per clip: mean %.0f, max %d" % (L.mean(), L.max()), "| %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
