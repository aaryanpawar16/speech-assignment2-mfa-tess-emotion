"""Praat-style figures of the alignments: waveform + spectrogram + word tier + phone tier.

    python visualize.py --root ..        -> <root>/outputs/figures/*.png   and   <root>/outputs/praat_view/ (wav + TextGrid pairs)

The figures are drawn with matplotlib so they can be regenerated without Praat; the same TextGrids can be opened
in Praat (see README, "Inspecting the output in Praat").
"""
import argparse
import glob
import os
import shutil

import librosa
import matplotlib
import numpy as np
import soundfile as sf
import tgt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REF = "A3_norm_custom_arpa"
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
COL = {"vowel": "#f4a582", "cons": "#92c5de", "sil": "#eeeeee"}


def read(path):
    tg = tgt.io.read_textgrid(path, include_empty_intervals=True)
    return {t.name: [(i.start_time, i.end_time, i.text.strip()) for i in t.intervals] for t in tg.tiers}


def kind(label):
    base = "".join(c for c in label if not c.isdigit())
    return "sil" if label in ("", "sil", "sp") else ("vowel" if base in VOWELS else "cons")


def draw_tier(ax, tier, t0, t1, label, fs=8, phone=False):
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    for s, e, t in tier:
        if e < t0 or s > t1:
            continue
        ax.axvline(s, color="k", lw=0.5)
        ax.axvline(e, color="k", lw=0.5)
        if phone:
            ax.axvspan(max(s, t0), min(e, t1), color=COL[kind(t)], alpha=0.8)
        if t:
            xm = (max(s, t0) + min(e, t1)) / 2
            ax.text(xm, 0.5, t, ha="center", va="center", fontsize=fs if (e - s) > 0.03 or not phone else fs - 2, rotation=0 if (e - s) > 0.035 else 90, clip_on=True)
    ax.set_xticks([])


def draw(wav, tiers, t0, t1, title, path, extra=()):
    """tiers: dict name->intervals of the main system; extra: [(label, word_tier)] drawn below for comparison."""
    y, sr = sf.read(wav, dtype="float32")
    seg = y[int(t0 * sr): int(t1 * sr)]
    n_rows = 2 + 2 + len(extra)
    fig = plt.figure(figsize=(max(9, 2.2 * (t1 - t0)), 1.2 * n_rows + 2.6))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=[1.1, 2.0] + [0.7, 0.9] + [0.7] * len(extra), hspace=0.05)
    a0 = fig.add_subplot(gs[0])
    a0.plot(np.linspace(t0, t0 + len(seg) / sr, len(seg)), seg, lw=0.5, color="k")
    a0.set_xlim(t0, t1)
    a0.set_xticks([])
    a0.set_ylabel("waveform", rotation=0, ha="right", va="center", fontsize=9)
    a1 = fig.add_subplot(gs[1])
    S = librosa.amplitude_to_db(np.abs(librosa.stft(seg, n_fft=int(0.025 * sr), hop_length=int(0.005 * sr), win_length=int(0.025 * sr))), ref=np.max, top_db=70)
    a1.imshow(S[: int(S.shape[0] * 8000 / (sr / 2))], origin="lower", aspect="auto", cmap="Greys", extent=[t0, t0 + len(seg) / sr, 0, 8000])
    a1.set_xlim(t0, t1)
    a1.set_xticks([])
    a1.set_ylabel("Hz", rotation=0, ha="right", va="center", fontsize=9)
    aw = fig.add_subplot(gs[2])
    draw_tier(aw, tiers["words"], t0, t1, "words", fs=9)
    ap = fig.add_subplot(gs[3])
    draw_tier(ap, tiers["phones"], t0, t1, "phones", phone=True)
    last = ap
    for k, (label, tier) in enumerate(extra):
        ax = fig.add_subplot(gs[4 + k])
        draw_tier(ax, tier, t0, t1, label, fs=8)
        last = ax
    last.set_xticks(np.linspace(t0, t1, 7))
    last.set_xticklabels(["%.2f" % v for v in np.linspace(t0, t1, 7)], fontsize=8)
    last.set_xlabel("time (s)", fontsize=9)
    fig.suptitle(title, fontsize=10)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def find_word(tier, word):
    return [(s, e) for s, e, t in tier if t.lower() == word]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    fig_dir = os.path.join(root, "outputs", "figures")
    wavs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(root, "corpus_norm", "**", "*.wav"), recursive=True)}
    tgs = {}
    for cfg in ("A1_naive_stock_arpa", "A2_norm_stock_arpa", REF, "A6_norm_custom_mfa", "A7_norm_variants_arpa", "A8_trained_on_data_only"):
        for p in glob.glob(os.path.join(root, "outputs", cfg, "**", "*.TextGrid"), recursive=True):
            tgs.setdefault(cfg, {})[os.path.splitext(os.path.basename(p))[0]] = read(p)
    ref = tgs[REF]

    # 1. ISLE utterances (whole, non-native minimal pairs)
    for u in sorted(ref):
        if u.startswith("ISLE"):
            dur = ref[u]["words"][-1][1]
            draw(wavs[u], ref[u], 0, dur, "%s - %s (words and ARPA phones)" % (u, REF), os.path.join(fig_dir, "%s.png" % u))
    # 2. first 7 seconds of each news-reading file
    for u in sorted(ref):
        if u.startswith("F2B"):
            draw(wavs[u], ref[u], 0, 7.0, "%s, first 7 s - %s" % (u, REF), os.path.join(fig_dir, "%s_first7s.png" % u))
    # 3. out-of-vocabulary words: stock dictionary (<unk>/spn) vs custom G2P dictionary vs a different acoustic model
    for u, word in (("F2BJRLP1", "melnicove"), ("F2BJRLP2", "dukakis"), ("F2BJRLP2", "maffy"), ("F2BJRLP1", "wbur")):
        spans = find_word(ref[u]["words"], word) or find_word(tgs["A1_naive_stock_arpa"][u]["words"], word)
        if not spans:
            # the acronym only exists as letters in the spoken-form variant: use the letter sequence "b" "u" "r"
            spans = find_word(ref[u]["words"], "r")
            spans = spans[-1:] if spans else []
        if not spans:
            continue
        s, e = spans[0]
        t0, t1 = max(0.0, s - 1.2), e + 0.9
        extra = [("A1 naive+stock words", tgs["A1_naive_stock_arpa"][u]["words"]), ("A2 norm+stock words", tgs["A2_norm_stock_arpa"][u]["words"]),
                 ("A6 english_mfa words", tgs["A6_norm_custom_mfa"][u]["words"])]
        draw(wavs[u], ref[u], t0, t1, "%s around '%s': A3 (custom dictionary) phones/words and three other systems' word tiers" % (u, word),
             os.path.join(fig_dir, "compare_%s_%s.png" % (u, word)), extra=extra)

    # 3b. variants (A7) and the model trained on six clips only (A8)
    if "A7_norm_variants_arpa" in tgs and "A8_trained_on_data_only" in tgs:
        for u in sorted(ref):
            if u.startswith("ISLE") and u in tgs["A7_norm_variants_arpa"] and u in tgs["A8_trained_on_data_only"]:
                dur = ref[u]["words"][-1][1]
                draw(wavs[u], ref[u], 0, dur, "%s: A3 words/phones, A7 phones (pronunciation variants allowed), A8 words/phones (model trained on the six clips only)" % u,
                     os.path.join(fig_dir, "variants_and_scratch_%s.png" % u[-8:]),
                     extra=[("A7 variants: phones", tgs["A7_norm_variants_arpa"][u]["phones"]), ("A8 scratch: words", tgs["A8_trained_on_data_only"][u]["words"]),
                            ("A8 scratch: phones", tgs["A8_trained_on_data_only"][u]["phones"])])
    # 4. Praat view folder: wav + TextGrid (main configuration) side by side
    pv = os.path.join(root, "outputs", "praat_view")
    os.makedirs(pv, exist_ok=True)
    for p in glob.glob(os.path.join(root, "outputs", REF, "**", "*.TextGrid"), recursive=True):
        u = os.path.splitext(os.path.basename(p))[0]
        shutil.copyfile(p, os.path.join(pv, u + ".TextGrid"))
        shutil.copyfile(wavs[u], os.path.join(pv, u + ".wav"))
    print("figures:", len(glob.glob(os.path.join(fig_dir, "*.png"))), "in", fig_dir, "| Praat pairs:", len(glob.glob(os.path.join(pv, "*.wav"))), "in", pv)


if __name__ == "__main__":
    main()
