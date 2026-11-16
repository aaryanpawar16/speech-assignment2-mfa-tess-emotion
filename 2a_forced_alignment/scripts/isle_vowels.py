"""Do the non-native (ISLE) utterances realise the vowel contrasts that the aligner assumes?

The three ISLE files are minimal-pair sentences ("I said WHITE not BAIT", "... NEW not NO", "... BAD not BED").
A forced aligner can only place the *canonical* phones it is given (AY vs EY, UW vs OW, AE vs EH).  If the
speaker does not produce the contrast, the alignment is still produced but is acoustically wrong for one of the
two words.  This script measures F1/F2 (Praat, via parselmouth) inside the vowel intervals of the alignment.

    python isle_vowels.py --root ..
"""
import argparse
import csv
import glob
import os

import numpy as np
import parselmouth
import tgt

REF = "A3_norm_custom_arpa"
TARGET = {"white": "AY", "bait": "EY", "new": "UW", "no": "OW", "bad": "AE", "bed": "EH"}


def read(path):
    tg = tgt.io.read_textgrid(path, include_empty_intervals=True)
    return {t.name: [(i.start_time, i.end_time, i.text.strip()) for i in t.intervals] for t in tg.tiers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    out = os.path.join(root, "outputs", "analysis")
    os.makedirs(out, exist_ok=True)
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "outputs", REF, "ISLE*", "*.TextGrid"))):
        u = os.path.splitext(os.path.basename(p))[0]
        wav = glob.glob(os.path.join(root, "corpus_norm", "ISLE*", u + ".wav"))[0]
        snd = parselmouth.Sound(wav)
        tg = read(p)
        f0 = snd.to_pitch().selected_array["frequency"]
        f0 = f0[f0 > 0]
        ceiling = 5500 if np.median(f0) > 165 else 5000            # Praat's rule of thumb: female / male formant ceiling
        form = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=ceiling, window_length=0.025, time_step=0.005)
        for (ws, we, w) in tg["words"]:
            if w.lower() not in TARGET:
                continue
            ph = [x for x in tg["phones"] if x[0] >= ws - 1e-6 and x[1] <= we + 1e-6 and "".join(c for c in x[2] if not c.isdigit()) == TARGET[w.lower()]]
            if not ph:
                continue
            s, e, lab = ph[0]
            ts = [s + (e - s) * q for q in (0.25, 0.5, 0.75)]
            f1 = [form.get_value_at_time(1, t) for t in ts]
            f2 = [form.get_value_at_time(2, t) for t in ts]
            rows.append([u, w.lower(), lab, "%.0f" % ((e - s) * 1000), "%.0f" % np.median(f0), ceiling] + ["%.0f" % v for v in f1 + f2])
    hdr = ["utterance", "word", "vowel", "dur_ms", "speaker_median_F0", "formant_ceiling", "F1_25%", "F1_50%", "F1_75%", "F2_25%", "F2_50%", "F2_75%"]
    with open(os.path.join(out, "isle_vowel_formants.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print("%-12s %-6s %-5s %6s %6s | F1 25/50/75%%        | F2 25/50/75%%" % ("utterance", "word", "vowel", "ms", "F0"))
    for r in rows:
        print("%-12s %-6s %-5s %6s %6s | %5s %5s %5s | %5s %5s %5s" % (r[0][-8:], r[1], r[2], r[3], r[4], r[6], r[7], r[8], r[9], r[10], r[11]))
    by = {r[1]: r for r in rows}
    print()
    for a, b in (("white", "bait"), ("new", "no"), ("bad", "bed")):
        if a in by and b in by:
            d1 = abs(float(by[a][7]) - float(by[b][7]))
            d2 = abs(float(by[a][10]) - float(by[b][10]))
            print("%-5s vs %-5s: |dF1| = %4.0f Hz, |dF2| = %4.0f Hz at the vowel midpoint" % (a, b, d1, d2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cols = {"white": "#d73027", "bait": "#fc8d59", "new": "#1a9850", "no": "#91cf60", "bad": "#4575b4", "bed": "#91bfdb"}
    for r in rows:
        f2 = [float(r[9]), float(r[10]), float(r[11])]
        f1 = [float(r[6]), float(r[7]), float(r[8])]
        ax.plot(f2, f1, "-o", color=cols[r[1]], ms=4, lw=1.2)
        ax.annotate("%s [%s]" % (r[1], r[2]), (f2[1], f1[1]), textcoords="offset points", xytext=(5, 4), fontsize=9, color=cols[r[1]])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.set_title("ISLE speaker: vowels of the minimal-pair words\n(25% -> 50% -> 75% of the aligned vowel interval)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(root, "outputs", "figures"), exist_ok=True)
    fig.savefig(os.path.join(root, "outputs", "figures", "isle_vowel_chart.png"), dpi=130)


if __name__ == "__main__":
    main()
