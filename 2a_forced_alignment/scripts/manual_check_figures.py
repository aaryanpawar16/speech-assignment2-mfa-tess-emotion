"""One zoomed figure per boundary of outputs/analysis/manual_check_template.csv, to make the hand check quick.

Each figure shows +-0.30 s around the boundary: waveform, wide-band spectrogram, minor ticks every 10 ms, and the word-start times of
A3 (blue, main model) and A6 (red, english_mfa).  Read the time where the word really starts (onset of energy / first glottal pulse of the
word), write it into the praat_time_s column, then run  extras_analysis.py --manual <csv>.   Figures: outputs/manual_check/.

    python scripts/manual_check_figures.py --root .
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--half", type=float, default=0.30)
    ap.add_argument("--blind", action="store_true", help="no A3/A6 lines and no times in the title (unbiased annotation); written to outputs/manual_check_blind/")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = os.path.join(root, "outputs", "manual_check_blind" if a.blind else "manual_check")
    os.makedirs(out, exist_ok=True)
    for old in glob.glob(os.path.join(out, "*.png")):
        os.remove(old)
    rows = list(csv.DictReader(open(os.path.join(root, "outputs", "analysis", "manual_check_template.csv"), encoding="utf-8")))
    for n, r in enumerate(rows, 1):
        wav = glob.glob(os.path.join(root, "corpus_norm", "**", r["utterance"] + ".wav"), recursive=True)[0]
        y, sr = sf.read(wav)
        y = y if y.ndim == 1 else y[:, 0]
        t3, t6 = float(r["A3_time_s"]), float(r["A6_time_s"])
        c = (t3 + t6) / 2
        lo, hi = max(0.0, c - a.half), min(len(y) / sr, c + a.half)
        seg = y[int(lo * sr):int(hi * sr)]
        f, t, S = spectrogram(seg, sr, nperseg=int(0.005 * sr), noverlap=int(0.0045 * sr), nfft=1024)
        fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
        ax[0].plot(lo + np.arange(len(seg)) / sr, seg, lw=0.5, color="#333")
        ax[1].pcolormesh(lo + t, f, 10 * np.log10(S + 1e-12), cmap="Greys", shading="auto", vmin=-100, vmax=-35)
        ax[1].set_ylim(0, 6000)
        for x in ax:
            if not a.blind:
                x.axvline(t3, color="#1f5fbf", lw=1.4)
                x.axvline(t6, color="#c0392b", lw=1.4, ls="--")
            x.set_xlim(lo, hi)
        ax[1].set_xticks(np.arange(np.ceil(lo * 100) / 100, hi, 0.05))
        ax[1].set_xticks(np.arange(np.ceil(lo * 100) / 100, hi, 0.01), minor=True)
        ax[1].grid(which="both", axis="x", alpha=0.25)
        ax[1].set_xlabel("time (s)")
        ax[0].set_title(("#%d  %s  |  word '%s' (word index %s): where does the word start?" % (n, r["utterance"], r["word"], r["word_index"])) if a.blind else
                        ("#%d  %s  |  word '%s' (word index %s): start  |  blue = A3 %.3f s, red dashed = A6 %.3f s" % (n, r["utterance"], r["word"], r["word_index"], t3, t6)), fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "%02d_%s_w%s_%s.png" % (n, r["utterance"], r["word_index"], r["word"])), dpi=110)
        plt.close(fig)
    print("wrote %d figures to %s" % (len(rows), out))


if __name__ == "__main__":
    main()
