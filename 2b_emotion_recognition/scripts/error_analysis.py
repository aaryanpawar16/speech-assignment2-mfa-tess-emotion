"""Error analysis: pick failure cases and explain them with measurable acoustic cues.

TESS contains every word, by the same speaker, in all 7 emotions.  For each failure we therefore compare the
misclassified clip with the SAME speaker saying the SAME word in the true emotion and in the predicted emotion:
mel-spectrogram, F0 contour, and a few descriptors (duration, mean F0, F0 range, RMS level).

    python scripts/error_analysis.py --cache <cache> --tess <TESS folder> --protocol spk_OAF2YAF --seeds 0 1 2 --n 5
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import EMOTIONS, build_metadata, load_audio, logmel, preprocess_audio  # noqa: E402
from common.engine import run_dir  # noqa: E402


def descriptors(path):
    import librosa
    y = preprocess_audio(load_audio(path))
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=700, sr=16000, frame_length=1024)
    f = f0[voiced & ~np.isnan(f0)]
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    return dict(dur=len(y) / 16000, f0_mean=float(np.mean(f)) if len(f) else float("nan"), f0_range=float(np.percentile(f, 95) - np.percentile(f, 5)) if len(f) > 5 else float("nan"),
                rms_db=float(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9)), rms_var=float(np.std(20 * np.log10(rms + 1e-6))), y=y, f0=f0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tess", required=True)
    ap.add_argument("--protocol", default="spk_OAF2YAF")
    ap.add_argument("--modality", default="speech")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {(m["speaker"], m["word"], m["emotion"]): m["path"] for m in build_metadata(a.tess)}
    errs = []
    for seed in a.seeds:
        fn = os.path.join(run_dir(a.modality, a.protocol, seed), "test_predictions.csv")
        for r in csv.DictReader(open(fn, encoding="utf-8")):
            if r["true"] != r["pred"]:
                r["seed"] = seed
                errs.append(r)
    from collections import Counter
    pair = Counter((r["true"], r["pred"]) for r in errs)
    print("failures:", len(errs), "| most common confusions:", pair.most_common(6))
    # choose cases: the top confusion pairs first, the most confident error of each pair
    chosen, used = [], set()
    for (t, p), _ in pair.most_common():
        cand = sorted([r for r in errs if r["true"] == t and r["pred"] == p], key=lambda r: -float(r["confidence"]))
        for r in cand:
            key = (r["speaker"], r["word"], r["true"])
            if key not in used:
                chosen.append(r)
                used.add(key)
                break
        if len(chosen) >= a.n:
            break
    os.makedirs(os.path.join(ROOT, "Results", "plots"), exist_ok=True)
    md = ["# Error analysis (%s, protocol %s, seeds %s)" % (a.modality, a.protocol, a.seeds), "",
          "Total failures over %d seeds: %d. Most frequent confusions (true -> predicted): %s." % (len(a.seeds), len(errs), ", ".join("%s->%s (%d)" % (t, p, c) for (t, p), c in pair.most_common(5))), ""]
    for k, r in enumerate(chosen, 1):
        spk, word, t, p = r["speaker"], r["word"], r["true"], r["pred"]
        d = {e: descriptors(paths[(spk, word, e)]) for e in {t, p}}
        # class-level reference for this speaker: mean F0 / level over all of its clips in the two classes
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), gridspec_kw=dict(height_ratios=[1.5, 1]))
        for c, (e, ttl) in enumerate(((t, "TRUE: %s" % t), (p, "PREDICTED: %s" % p))):
            S = logmel(d[e]["y"])
            axes[0, c].imshow(S.T, origin="lower", aspect="auto", cmap="magma")
            axes[0, c].set_title("%s - %s says '%s' (%s)" % (ttl, spk, word, e), fontsize=9)
            axes[0, c].set_ylabel("mel bin")
            axes[1, c].plot(np.arange(len(d[e]["f0"])) * 0.0128, d[e]["f0"], color="k", lw=1)
            axes[1, c].set_ylim(70, 700)
            axes[1, c].set_ylabel("F0 (Hz)")
            axes[1, c].set_xlabel("time (s)")
        fig.suptitle("Failure %d: %s clip '%s' (speaker %s), confidence %s" % (k, t, word, spk, r["confidence"]), fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, "Results", "plots", "failure_%d_%s_%s_%s.png" % (k, spk, word, t)), dpi=120)
        plt.close(fig)
        dt, dp = d[t], d[p]
        md += ["## Failure %d: %s, word '%s', true **%s**, predicted **%s** (confidence %s, seed %s)" % (k, spk, word, t, p, r["confidence"], r["seed"]), "",
               "| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |", "|---|---|---|---|---|---|"]
        for e, dd in ((t, dt), (p, dp)):
            md.append("| same word, %s%s | %.2f | %.0f | %.0f | %.1f | %.1f |" % (e, " (true)" if e == t else " (predicted)", dd["dur"], dd["f0_mean"], dd["f0_range"], dd["rms_db"], dd["rms_var"]))
        md += ["", "Class probabilities: " + ", ".join("%s %.2f" % (e.replace("p_", ""), float(r[e])) for e in r if e.startswith("p_")), ""]
    open(os.path.join(ROOT, "Results", "error_analysis_%s_%s.md" % (a.modality, a.protocol)), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
