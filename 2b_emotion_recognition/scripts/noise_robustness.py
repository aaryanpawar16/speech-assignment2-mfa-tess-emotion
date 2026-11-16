"""When does fusion help?  Test the trained speech-only and fusion models with degraded speech.

White Gaussian noise is added to the (trimmed, normalised) test waveforms at several signal-to-noise ratios.
The text input is unchanged, so a fusion model can only gain if the text carries emotion information.

    python scripts/noise_robustness.py --cache <cache> --tess <TESS folder> [--seeds 0 1 2]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import build_metadata, logmel, pad_or_crop, preprocess_audio, load_audio  # noqa: E402
from common.engine import load_trained, metrics, predict  # noqa: E402

SNRS = [None, 30, 20, 10, 5, 0]


def noisy_features(paths, snr_db, rng):
    out = []
    for p in paths:
        y = preprocess_audio(load_audio(p))
        if snr_db is not None:
            noise = rng.standard_normal(len(y)).astype(np.float32)
            y = y + noise * np.sqrt((y ** 2).mean() / (10 ** (snr_db / 10)) / (noise ** 2).mean())
        out.append(pad_or_crop(logmel(y))[0])
    return torch.from_numpy(np.stack(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tess", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--protocol", default="word")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    paths = {os.path.basename(m["path"]): m["path"] for m in build_metadata(a.tess)}
    res = {}
    for seed in a.seeds:
        models = {k: load_trained(k, a.protocol, seed, a.cache, "concat", a.device) for k in ("speech", "text", "fusion")}
        _, bundle, split, _ = models["speech"]
        idx = split["test"]
        files = [paths[bundle.meta[i]["file"]] for i in idx]
        y = bundle.y[idx].cpu().numpy()
        for snr in SNRS:
            rng = np.random.default_rng(1000 + seed)
            X = noisy_features(files, snr, rng).to(bundle.device)
            for k, (model, b, _, _) in models.items():
                logits = predict(model, b, idx, speech_override=X)
                res.setdefault((k, snr), []).append(metrics(y, logits)["accuracy"])
        print("seed", seed, "done", flush=True)
    os.makedirs(os.path.join(ROOT, "Results", "tables"), exist_ok=True)
    with open(os.path.join(ROOT, "Results", "tables", "noise_robustness.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "snr_db", "accuracy_mean_%", "accuracy_std_%", "seeds"])
        for (k, snr), v in sorted(res.items(), key=lambda kv: (kv[0][0], 99 if kv[0][1] is None else -kv[0][1])):
            w.writerow([k, "clean" if snr is None else snr, "%.2f" % (100 * np.mean(v)), "%.2f" % (100 * np.std(v)), len(v)])
    print("%-8s" % "SNR dB" + "".join("%10s" % ("clean" if s is None else s) for s in SNRS))
    for k in ("speech", "text", "fusion"):
        print("%-8s" % k + "".join("%9.1f%%" % (100 * np.mean(res[(k, s)])) for s in SNRS))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [40 if s is None else s for s in SNRS]
    for k, c in (("speech", "#2171b5"), ("fusion", "#d7301f"), ("text", "#737373")):
        m = np.array([100 * np.mean(res[(k, s)]) for s in SNRS])
        sd = np.array([100 * np.std(res[(k, s)]) for s in SNRS])
        ax.errorbar(xs, m, yerr=sd, marker="o", label=k + ("-only" if k != "fusion" else " (concat)"), color=c, capsize=3)
    ax.axhline(100 / 7, ls=":", color="k", lw=0.8)
    ax.text(0.5, 100 / 7 + 1.5, "chance", fontsize=8)
    ax.set_xticks(xs, ["clean" if s is None else str(s) for s in SNRS])
    ax.set_xlabel("signal-to-noise ratio of the speech (dB)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Robustness to additive white noise (word-disjoint split)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "Results", "plots", "noise_robustness.png"), dpi=130)


if __name__ == "__main__":
    main()
