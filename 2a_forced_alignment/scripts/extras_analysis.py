"""Extra analyses of the alignments.

1. Pronunciation variants (A7): which vowel did the aligner choose for the minimal-pair words, canonical or the confusable one?
2. Acoustic reference at pause boundaries: for every boundary between silence and speech, an independent estimate from the
   short-time energy of the waveform (NOT a hand annotation) and the signed/absolute error of each configuration against it.
3. Manual check (optional): writes a template with boundaries to correct in Praat; with --manual <csv> the errors of every
   configuration against YOUR corrections are computed.

    python scripts/extras_analysis.py --root .
    python scripts/extras_analysis.py --root . --manual outputs/analysis/manual_check_filled.csv
"""
import argparse
import csv
import glob
import os

import numpy as np
import soundfile as sf
import tgt

CONFIGS = ["A1_naive_stock_arpa", "A2_norm_stock_arpa", "A3_norm_custom_arpa", "A4_norm_custom_arpa_ft",
           "A5_norm_custom_adapted", "A6_norm_custom_mfa", "A7_norm_variants_arpa", "A8_trained_on_data_only"]
CANON = {"white": "AY", "bait": "EY", "new": "UW", "no": "OW", "bad": "AE", "bed": "EH"}
NONSPEECH = ("", "sil", "sp", "spn", "<eps>")


def load(path):
    tg = tgt.io.read_textgrid(path, include_empty_intervals=True)
    return {t.name: [(i.start_time, i.end_time, i.text.strip()) for i in t.intervals] for t in tg.tiers}


def load_config(root, cfg):
    return {os.path.splitext(os.path.basename(p))[0]: load(p) for p in glob.glob(os.path.join(root, "outputs", cfg, "**", "*.TextGrid"), recursive=True)}


def strip(ph):
    return "".join(c for c in ph if not c.isdigit())


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------------------------------ 1. variants
def variant_choice(root, out):
    data = load_config(root, "A7_norm_variants_arpa")
    base = load_config(root, "A3_norm_custom_arpa")
    rows = []
    for u in sorted(data):
        if not u.startswith("ISLE"):
            continue
        for (s, e, w) in data[u]["words"]:
            if w.lower() not in CANON:
                continue
            ph = [strip(x[2]) for x in data[u]["phones"] if x[0] >= s - 1e-6 and x[1] <= e + 1e-6 and x[2] not in NONSPEECH]
            vowel = [p for p in ph if p in ("AY", "EY", "UW", "OW", "AE", "EH")]
            chosen = vowel[0] if vowel else "?"
            rows.append([u[-8:], w.lower(), CANON[w.lower()], chosen, "canonical" if chosen == CANON[w.lower()] else "VARIANT", " ".join(ph), "%.0f" % ((e - s) * 1000)])
    write_csv(os.path.join(out, "variant_choice_A7.csv"), ["utterance", "word", "canonical_vowel", "vowel_chosen", "choice", "phones", "word_ms"], rows)
    print("\n== A7: vowel chosen by the aligner when canonical and confusable pronunciations are both allowed")
    for r in rows:
        print("  %-9s %-6s canonical %-3s chosen %-3s -> %s" % tuple(r[:5]))
    return rows


# ------------------------------------------------------------------------------------------ 2. energy reference
def energy_db(wav, sr, win=0.025, hop=0.005):
    n, h = int(win * sr), int(hop * sr)
    frames = [wav[i:i + n] for i in range(0, max(1, len(wav) - n), h)]
    e = np.array([np.mean(f ** 2) + 1e-12 for f in frames])
    db = 10 * np.log10(e)
    k = np.ones(3) / 3                                      # 15 ms smoothing
    return np.convolve(db, k, mode="same"), hop


def energy_boundary(db, hop, t_hint, kind, search=0.15):
    """Time where the energy crosses halfway between the silence level and the speech level next to the hinted pause boundary.

    kind "onset": silence -> speech (the boundary is the START of a word after a pause); "offset": speech -> silence."""
    i0, i1 = max(0, int((t_hint - search) / hop)), min(len(db) - 1, int((t_hint + search) / hop))
    seg = db[i0:i1 + 1]
    if len(seg) < 6:
        return None
    lo, hi = np.percentile(seg, 10), np.percentile(seg, 90)
    if hi - lo < 12:                                          # no clear energy step in the window (e.g. whispered onset)
        return None
    thr = (lo + hi) / 2
    above = seg > thr
    idx = np.where(~above[:-1] & above[1:])[0] if kind == "onset" else np.where(above[:-1] & ~above[1:])[0]
    if not len(idx):
        return None
    j = idx[np.argmin(np.abs((i0 + idx) * hop - t_hint))]
    return (i0 + j + 0.5) * hop


def pause_boundaries(tiers, min_pause=0.08):
    """(time, kind) for every boundary between a silence interval (or utterance edge) and a word; pauses >= min_pause only at interior positions."""
    words = tiers["words"]
    out = []
    for k, (s, e, w) in enumerate(words):
        if w != "":
            continue
        if e - s < min_pause:
            continue
        if k + 1 < len(words) and words[k + 1][2] != "":
            out.append((e, "onset"))
        if k > 0 and words[k - 1][2] != "":
            out.append((s, "offset"))
    return out


def energy_reference(root, out, configs):
    wavs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(root, "corpus_norm", "**", "*.wav"), recursive=True)}
    ref = load_config(root, "A3_norm_custom_arpa")
    # candidate boundaries come from every configuration's pauses (so a config that places a pause elsewhere is still evaluated at ITS boundary,
    # against the energy step nearest to it); the reference time is the energy crossing, searched within +-150 ms.
    all_cfg = {c: load_config(root, c) for c in configs}
    rows, per_cfg = [], {c: [] for c in configs}
    for u, path in wavs.items():
        wav, sr = sf.read(path)
        db, hop = energy_db(wav if wav.ndim == 1 else wav[:, 0], sr)
        for c in configs:
            if u not in all_cfg[c]:
                continue
            for t, kind in pause_boundaries(all_cfg[c][u]):
                tr = energy_boundary(db, hop, t, kind)
                if tr is None:
                    continue
                per_cfg[c].append(t - tr)
                rows.append([c, u[-12:], kind, "%.3f" % t, "%.3f" % tr, "%.0f" % ((t - tr) * 1000)])
    write_csv(os.path.join(out, "pause_boundary_vs_energy.csv"), ["config", "utterance", "kind", "textgrid_s", "energy_s", "error_ms"], rows)
    print("\n== pause boundaries vs. short-time-energy reference (signed error = TextGrid - energy, ms)")
    summ = []
    for c in configs:
        e = np.array(per_cfg[c]) * 1000
        if len(e) == 0:
            continue
        summ.append([c, len(e), "%.1f" % np.mean(e), "%.1f" % np.mean(np.abs(e)), "%.1f" % np.median(np.abs(e)), "%.0f" % np.max(np.abs(e))])
        print("  %-26s n=%3d  mean %6s  mean|e| %5s  median|e| %5s  max|e| %4s" % tuple(summ[-1]))
    write_csv(os.path.join(out, "pause_boundary_vs_energy_summary.csv"), ["config", "n_boundaries", "mean_error_ms", "mean_abs_error_ms", "median_abs_error_ms", "max_abs_error_ms"], summ)


# ------------------------------------------------------------------------------------------ 3. manual check
def manual_template(root, out, n=12):
    """Boundaries where the two acoustic models disagree most (A3 vs A6): the most informative ones to correct by hand."""
    a3, a6 = load_config(root, "A3_norm_custom_arpa"), load_config(root, "A6_norm_custom_mfa")
    cand = []
    for u in sorted(a3):
        if u not in a6:
            continue
        wa = [x for x in a3[u]["words"] if x[2] not in NONSPEECH]
        wb = [x for x in a6[u]["words"] if x[2] not in NONSPEECH]
        if len(wa) != len(wb):
            continue
        for k, (x, y) in enumerate(zip(wa, wb)):
            if x[2].lower() == y[2].lower():
                cand.append((abs(x[0] - y[0]), u, k, x[2], x[0], y[0]))
    cand.sort(reverse=True)
    pick = [c for c in cand if c[0] > 0.02][:n // 2]                       # largest disagreements
    rest = [c for c in cand if c[0] <= 0.02]
    step = max(1, len(rest) // (n - len(pick)))
    pick += rest[::step][:n - len(pick)]                                     # plus a spread of agreeing boundaries
    rows = [[u, k, w, "word_start", "%.3f" % a, "%.3f" % b, ""] for d, u, k, w, a, b in pick]
    path = os.path.join(out, "manual_check_template.csv")
    write_csv(path, ["utterance", "word_index", "word", "boundary", "A3_time_s", "A6_time_s", "praat_time_s"], rows)
    print("\nmanual-check template:", path, "(%d boundaries; fill the praat_time_s column, save as manual_check_filled.csv, rerun with --manual)" % len(rows))


def manual_eval(root, out, path, configs):
    gold = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["praat_time_s"].strip()]
    data = {c: load_config(root, c) for c in configs}
    print("\n== error against YOUR hand-corrected word starts (%d boundaries, ms)" % len(gold))
    rows = []
    for c in configs:
        errs = []
        for g in gold:
            tg = data[c].get(g["utterance"])
            if not tg:
                continue
            w = [x for x in tg["words"] if x[2] not in NONSPEECH]
            k = int(g["word_index"])
            if k < len(w) and w[k][2].lower() == g["word"].lower():
                errs.append((w[k][0] - float(g["praat_time_s"])) * 1000)
        if errs:
            e = np.array(errs)
            rows.append([c, len(e), "%.1f" % e.mean(), "%.1f" % np.abs(e).mean(), "%.0f" % np.abs(e).max()])
            print("  %-26s n=%2d mean %6s  mean|e| %5s  max|e| %4s" % tuple(rows[-1]))
    write_csv(os.path.join(out, "manual_check_errors.csv"), ["config", "n", "mean_error_ms", "mean_abs_error_ms", "max_abs_error_ms"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--manual", default=None, help="CSV with the praat_time_s column filled in")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    out = os.path.join(root, "outputs", "analysis")
    os.makedirs(out, exist_ok=True)
    configs = [c for c in CONFIGS if os.path.isdir(os.path.join(root, "outputs", c))]
    if "A7_norm_variants_arpa" in configs:
        variant_choice(root, out)
    energy_reference(root, out, configs)
    if a.manual:
        manual_eval(root, out, a.manual, configs)
    else:
        manual_template(root, out)


if __name__ == "__main__":
    main()
