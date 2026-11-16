"""How differently do the two TESS speakers realise each emotion?  (explains the unseen-speaker results)

Mean duration, F0 mean, F0 range (5-95 percentile) and level over 40 random clips per speaker x emotion
(trimmed, normalised audio; Praat pitch via parselmouth).  Writes Results/tables/acoustic_profile.csv.

    python scripts/acoustic_profile.py --tess <TESS folder>
"""
import argparse
import collections
import csv
import os
import random
import sys

import numpy as np
import parselmouth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.data import build_metadata, load_audio, preprocess_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tess", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    random.seed(0)
    by = collections.defaultdict(list)
    for m in build_metadata(a.tess):
        by[(m["speaker"], m["emotion"])].append(m)
    rows = []
    for k in sorted(by):
        d, f, r, lv = [], [], [], []
        for m in random.sample(by[k], a.n):
            y = preprocess_audio(load_audio(m["path"]))
            d.append(len(y) / 16000)
            pi = parselmouth.Sound(y.astype(float), 16000).to_pitch(pitch_floor=75, pitch_ceiling=600).selected_array["frequency"]
            pi = pi[pi > 0]
            if len(pi) > 5:
                f.append(pi.mean())
                r.append(np.percentile(pi, 95) - np.percentile(pi, 5))
            lv.append(20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9))
        rows.append([k[0], k[1], "%.2f" % np.mean(d), "%.0f" % np.mean(f), "%.0f" % np.mean(r), "%.1f" % np.mean(lv)])
        print(*rows[-1])
    os.makedirs(os.path.join(ROOT, "Results", "tables"), exist_ok=True)
    with open(os.path.join(ROOT, "Results", "tables", "acoustic_profile.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["speaker", "emotion", "duration_s", "F0_mean_Hz", "F0_range_5_95_Hz", "level_dB"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
