"""Quantitative inspection of the MFA TextGrids produced by run_pipeline.sh.

There is no hand-labelled gold alignment for these six recordings, so quality is judged with
proxies that do not need one:

  * agreement between independently configured aligners (boundary differences, ms);
  * MFA's own per-utterance log-likelihood / phone-duration deviation / SNR;
  * "squeezed" phones (3 frames = 30 ms is the shortest the HMM topology allows) which signal a
    skipped or forced-through phone;
  * how out-of-vocabulary words are handled and which G2P pronunciation variant the aligner chose;
  * whether pauses coincide with punctuation in the transcript.

    python analyze_alignments.py --root ..           (writes <root>/outputs/analysis/*.csv and summary.txt)
"""
import argparse
import csv
import difflib
import glob
import os
import re
from collections import defaultdict

import numpy as np
import tgt

CONFIGS = ["A1_naive_stock_arpa", "A2_norm_stock_arpa", "A3_norm_custom_arpa", "A4_norm_custom_arpa_ft",
           "A5_norm_custom_adapted", "A6_norm_custom_mfa", "A7_norm_variants_arpa", "A8_trained_on_data_only", "A9_norm_trained_g2p"]
REF = "A3_norm_custom_arpa"
FRAME = 0.010                                   # MFA feature frame shift: every boundary is a multiple of 10 ms
OOV_WORDS = ["dukakis", "politicize", "maffy", "melnicove", "wbur"]


def load(path):
    tg = tgt.io.read_textgrid(path, include_empty_intervals=True)
    out = {}
    for tier in tg.tiers:
        out[tier.name] = [(i.start_time, i.end_time, i.text.strip()) for i in tier.intervals]
    return out


def utt_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_config(root, cfg):
    d = {}
    for p in glob.glob(os.path.join(root, "outputs", cfg, "**", "*.TextGrid"), recursive=True):
        d[utt_name(p)] = load(p)
    return d


def speech(tier):
    return [x for x in tier if x[2] not in ("", "sil", "sp", "spn", "<eps>")]


def matched_words(a, b):
    """Index pairs of words with identical labels in the two word sequences (difflib on the label lists)."""
    la, lb = [x[2].lower() for x in a], [x[2].lower() for x in b]
    sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
    pairs = []
    for blk in sm.get_matching_blocks():
        pairs += [(blk.a + k, blk.b + k) for k in range(blk.size)]
    return pairs


def pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    out_dir = os.path.join(root, "outputs", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    data = {c: load_config(root, c) for c in CONFIGS if os.path.isdir(os.path.join(root, "outputs", c))}
    lines = []
    P = lambda s="": (print(s), lines.append(s))

    # ------------------------------------------------------------------ MFA statistics + phone/word counts
    rows, per_utt = [], []
    for cfg, utts in data.items():
        ana = {}
        fn = os.path.join(root, "outputs", cfg, "alignment_analysis.csv")
        if os.path.exists(fn):
            for r in csv.DictReader(open(fn, encoding="utf-8")):
                ana[r["file"]] = r
        n_words = n_phones = n_short = n_unk = 0
        dev, ll, sll = [], [], []
        for u, tg in utts.items():
            ph, wd = speech(tg["phones"]), speech(tg["words"])
            short = [p for p in ph if p[1] - p[0] <= 0.0301]
            unk = [w for w in tg["words"] if w[2].lower() in ("<unk>", "spn")] + [p for p in tg["phones"] if p[2] == "spn"]
            n_words += len(wd); n_phones += len(ph); n_short += len(short); n_unk += len(unk)
            a = ana.get(u, {})
            f = lambda k: float(a[k]) if a.get(k) not in (None, "") else float("nan")
            dev.append(f("phone_duration_deviation")); ll.append(f("overall_log_likelihood")); sll.append(f("speech_log_likelihood"))
            per_utt.append([cfg, u, len(wd), len(ph), len(short), len(unk), f("overall_log_likelihood"), f("speech_log_likelihood"),
                            f("phone_duration_deviation"), f("snr")])
        rows.append([cfg, len(utts), n_words, n_phones, n_short, "%.1f" % (100 * n_short / max(1, n_phones)), n_unk,
                     "%.2f" % np.nanmean(ll), "%.2f" % np.nanmean(sll), "%.2f" % np.nanmean(dev)])
    write_csv(os.path.join(out_dir, "summary_by_config.csv"),
              ["config", "utterances", "words", "phones", "phones_<=30ms", "pct_phones_<=30ms", "unk_or_spn_items",
               "mean_overall_loglik", "mean_speech_loglik", "mean_phone_dur_deviation"], rows)
    write_csv(os.path.join(out_dir, "per_utterance.csv"),
              ["config", "utterance", "words", "phones", "phones<=30ms", "unk_or_spn", "overall_loglik", "speech_loglik", "phone_dur_deviation", "snr"], per_utt)
    P("== Summary by configuration")
    P("%-26s %4s %6s %6s %8s %7s %6s %9s %9s %8s" % ("config", "utts", "words", "phones", "<=30ms", "%", "unk", "loglik", "speechLL", "durdev"))
    for r in rows:
        P("%-26s %4d %6d %6d %8d %7s %6d %9s %9s %8s" % tuple(r))

    # ------------------------------------------------------------------ boundary agreement against the reference configuration
    ref = data.get(REF)
    agree = []
    P("\n== Word- and phone-boundary agreement with %s (ms; 'within' = share of boundaries differing by at most that much)" % REF)
    P("%-26s %-6s %6s %8s %8s %8s %8s %8s %8s" % ("config", "level", "n", "mean", "median", "p90", "max", "<=20ms", "<=50ms"))
    for cfg, utts in data.items():
        if cfg == REF or ref is None:
            continue
        wd_diff, ph_diff = [], []
        for u in sorted(set(utts) & set(ref)):
            a, b = utts[u]["words"], ref[u]["words"]
            for ia, ib in matched_words(speech(a), speech(b)):
                sa, sb = speech(a)[ia], speech(b)[ib]
                wd_diff += [abs(sa[0] - sb[0]), abs(sa[1] - sb[1])]
            if cfg != "A6_norm_custom_mfa":                                       # same ARPA phone set -> compare phones too
                pa, pb = speech(utts[u]["phones"]), speech(ref[u]["phones"])
                la, lb = [x[2] for x in pa], [x[2] for x in pb]
                for blk in difflib.SequenceMatcher(a=la, b=lb, autojunk=False).get_matching_blocks():
                    for k in range(blk.size):
                        ph_diff += [abs(pa[blk.a + k][0] - pb[blk.b + k][0]), abs(pa[blk.a + k][1] - pb[blk.b + k][1])]
        for level, d in (("word", wd_diff), ("phone", ph_diff)):
            if not d:
                continue
            d = np.array(d) * 1000
            agree.append([cfg, level, len(d), "%.1f" % d.mean(), "%.1f" % np.median(d), "%.1f" % pct(d, 90), "%.0f" % d.max(),
                          "%.1f" % (100 * (d <= 20).mean()), "%.1f" % (100 * (d <= 50).mean())])
            P("%-26s %-6s %6d %8.1f %8.1f %8.1f %8.0f %7.1f%% %7.1f%%" % (cfg, level, len(d), d.mean(), np.median(d), pct(d, 90), d.max(),
                                                                       100 * (d <= 20).mean(), 100 * (d <= 50).mean()))
    write_csv(os.path.join(out_dir, "boundary_agreement_vs_%s.csv" % REF),
              ["config", "level", "n_boundaries", "mean_ms", "median_ms", "p90_ms", "max_ms", "pct_within_20ms", "pct_within_50ms"], agree)

    # ------------------------------------------------------------------ out-of-vocabulary words
    P("\n== Out-of-vocabulary words: how each configuration handled them")
    oov_rows = []
    for cfg, utts in data.items():
        for u, tg in utts.items():
            wds, phs = tg["words"], tg["phones"]
            for (s, e, t) in wds:
                if t.lower() in OOV_WORDS or t.lower() in ("<unk>",):
                    inside = [p[2] for p in phs if p[0] >= s - 1e-6 and p[1] <= e + 1e-6]
                    oov_rows.append([cfg, u, t, "%.2f" % s, "%.2f" % e, "%.0f" % ((e - s) * 1000), " ".join(inside)])
    write_csv(os.path.join(out_dir, "oov_words.csv"), ["config", "utterance", "word_label", "start", "end", "dur_ms", "phones"], oov_rows)
    for r in oov_rows:
        if r[0] in ("A1_naive_stock_arpa", "A2_norm_stock_arpa", REF, "A6_norm_custom_mfa"):
            P("%-24s %-14s %-12s %5s-%5s %6s ms  %s" % (r[0], r[1][-8:], r[2], r[3], r[4], r[5], r[6]))

    # ------------------------------------------------------------------ squeezed phones in the reference configuration
    if ref:
        sq = []
        for u, tg in ref.items():
            for (s, e, t) in speech(tg["phones"]):
                if e - s <= 0.0301:
                    word = next((w[2] for w in tg["words"] if w[0] - 1e-6 <= s and e <= w[1] + 1e-6), "")
                    sq.append([u, t, word, "%.2f" % s, "%.0f" % ((e - s) * 1000)])
        STOPS = {"P", "B", "T", "D", "K", "G", "CH", "JH"}
        VOW = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
        cls = lambda t: ("stop/affricate" if t in STOPS else "vowel" if "".join(c for c in t if not c.isdigit()) in VOW else "other consonant")
        for r in sq:
            r.append(cls(r[1]))
        write_csv(os.path.join(out_dir, "squeezed_phones_%s.csv" % REF), ["utterance", "phone", "word", "start", "dur_ms", "class"], sq)
        n_ph = sum(len(speech(tg["phones"])) for tg in ref.values())
        P("\n== %d of %d phones (%.1f%%) sit at the 30 ms minimum in %s" % (len(sq), n_ph, 100 * len(sq) / max(1, n_ph), REF))
        from collections import Counter
        P("   by class: " + ", ".join("%s %d" % kv for kv in Counter(r[5] for r in sq).most_common()) +
          "  (short stops are normal; squeezed vowels / sonorants / fricatives are the suspicious ones)")
        # clusters: >= 3 squeezed phones within 150 ms of each other = a region where the alignment is probably unreliable
        clusters, cur = [], []
        for r in sorted(sq, key=lambda r: (r[0], float(r[3]))):
            if cur and (r[0] != cur[-1][0] or float(r[3]) - float(cur[-1][3]) > 0.15):
                clusters.append(cur)
                cur = []
            cur.append(r)
        clusters.append(cur)
        clusters = [c for c in clusters if len(c) >= 3]
        P("   clusters of >=3 squeezed phones within 150 ms: %d" % len(clusters))
        for c in clusters:
            P("     %-10s %6.2f-%6.2f s  words: %s  phones: %s" % (c[0][0][:8], float(c[0][3]), float(c[-1][3]), " ".join(dict.fromkeys(r[2] for r in c)), " ".join(r[1] for r in c)))
        P("   first 12 squeezed phones: utterance / phone / word / start")
        for r in sq[:12]:
            P("   %-34s %-4s %-14s %6s" % tuple(r[:4]))

    # ------------------------------------------------------------------ pauses vs punctuation (news-reading speaker only)
    if ref:
        P("\n== Pauses vs punctuation (%s): a pause = silence >= 150 ms after a word" % REF)
        prow = []
        for u, tg in sorted(ref.items()):
            if not u.startswith("F2B"):
                continue
            lab = os.path.join(root, "corpus_norm", "F2B", u + ".lab")
            text = open(lab, encoding="utf-8").read()
            toks = []                                       # (word, punctuation-after)
            for raw in text.split():
                parts = re.split(r"-", re.sub(r"^[^\w']+", "", raw))
                for k, part in enumerate(parts):
                    clean = re.sub(r"[^\w']+$", "", part).lower().replace("'s", "'s")
                    punct = bool(re.search(r"[.,;:!?]$", part)) if k == len(parts) - 1 else False
                    if clean:
                        toks.append((clean, punct))
            wd = tg["words"]
            sw = [(i, w) for i, w in enumerate(wd) if w[2] not in ("",)]
            labels = [w[2].lower() for _, w in sw]
            sm = difflib.SequenceMatcher(a=[t[0] for t in toks], b=labels, autojunk=False)
            n_p = n_p_hit = n_pause = n_pause_p = 0
            last_word = sw[-1][0]                           # the silence after the last word is the end of the recording, not a pause
            for blk in sm.get_matching_blocks():
                for k in range(blk.size):
                    punct = toks[blk.a + k][1]
                    i, w = sw[blk.b + k]
                    if i == last_word:
                        continue                            # sentence-final mark: not an interior boundary
                    sil = (wd[i + 1][1] - wd[i + 1][0]) if i + 1 < len(wd) and wd[i + 1][2] == "" else 0.0
                    n_p += int(punct); n_p_hit += int(punct and sil >= 0.15)
                    n_pause += int(sil >= 0.15); n_pause_p += int(sil >= 0.15 and punct)
            prow.append([u, n_p, n_p_hit, n_pause, n_pause_p])
            P("   %-10s interior punctuation marks %2d, followed by a pause >=150 ms: %2d | interior pauses >=150 ms: %2d, of which at punctuation: %2d" % (u, n_p, n_p_hit, n_pause, n_pause_p))
        write_csv(os.path.join(out_dir, "pauses_vs_punctuation.csv"), ["utterance", "interior_punct_marks", "punct_with_pause", "interior_pauses", "pauses_at_punct"], prow)

    # ------------------------------------------------------------------ ISLE (non-native minimal pairs): word/phone table
    if ref:
        P("\n== ISLE utterances (%s): word and phone timings" % REF)
        irows = []
        for u, tg in sorted(ref.items()):
            if not u.startswith("ISLE"):
                continue
            for (s, e, t) in tg["words"]:
                if not t:
                    continue
                inside = [(p[2], p[1] - p[0]) for p in tg["phones"] if p[0] >= s - 1e-6 and p[1] <= e + 1e-6]
                irows.append([u, t, "%.2f" % s, "%.2f" % e, " ".join("%s(%d)" % (n, round(d * 1000)) for n, d in inside)])
                P("   %-4s %-7s %5.2f-%5.2f  %s" % (u[-8:-6] if False else u[-9:-6], t, s, e, " ".join("%s(%d)" % (n, round(d * 1000)) for n, d in inside)))
        write_csv(os.path.join(out_dir, "isle_words_phones.csv"), ["utterance", "word", "start", "end", "phones(ms)"], irows)

    open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nwritten:", out_dir)


if __name__ == "__main__":
    main()
