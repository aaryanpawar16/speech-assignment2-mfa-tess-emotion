"""Evaluate the G2P model trained in the extra experiment A9 (mfa train_g2p on a 25k-word sample of the stock dictionary).

1. Held-out words of the stock dictionary (not in the 25k training sample): word error rate (exact phone string, with and without stress
   digits) and phone error rate (Levenshtein distance / reference length) of the model's best pronunciation.
   (MFA's own `--evaluate` printed WER/LER = 100 for this model, which contradicts the outputs; the numbers below are computed here.)
2. The corpus out-of-vocabulary words: pronunciations of the trained model vs. the pretrained english_us_arpa G2P model (stress digits ignored).

    python scripts/g2p_eval.py --root .
"""
import argparse
import csv
import os


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]


def read(path):
    out = {}
    for ln in open(path, encoding="utf-8").read().splitlines():
        if ln.strip():
            c = ln.split("\t")
            out.setdefault(c[0], []).append(c[-1].split())
    return out


def strip(p):
    return [x.rstrip("012") for x in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    d = os.path.join(os.path.abspath(a.root), "dictionaries")
    ref, pred = read(os.path.join(d, "g2p_heldout_ref.dict")), read(os.path.join(d, "g2p_heldout_pred.dict"))
    words = [w for w in ref if w in pred]
    wer = sum(pred[w][0] != ref[w][0] for w in words) / len(words)
    wer_ns = sum(strip(pred[w][0]) != strip(ref[w][0]) for w in words) / len(words)
    per = sum(lev(pred[w][0], ref[w][0]) for w in words) / sum(len(ref[w][0]) for w in words)
    per_ns = sum(lev(strip(pred[w][0]), strip(ref[w][0])) for w in words) / sum(len(ref[w][0]) for w in words)
    print("held-out words: %d | WER %.1f%% (%.1f%% ignoring stress) | PER %.1f%% (%.1f%% ignoring stress)" % (len(words), 100 * wer, 100 * wer_ns, 100 * per, 100 * per_ns))
    oov_t, oov_p = read(os.path.join(d, "oov_trained_g2p.dict")), read(os.path.join(d, "oov_english_us_arpa.dict"))
    rows = []
    for w in sorted(oov_t):
        if w in oov_p:
            t1, p1 = strip(oov_t[w][0]), strip(oov_p[w][0])
            best = min(lev(strip(t), strip(p)) for t in oov_t[w] for p in oov_p[w])
            rows.append([w, " ".join(oov_t[w][0]), " ".join(oov_p[w][0]), lev(t1, p1), best])
            print("  %-11s trained: %-24s pretrained: %-24s edit(top1) %d, edit(best of 3x3) %d" % tuple(rows[-1]))
    with open(os.path.join(os.path.abspath(a.root), "outputs", "analysis", "g2p_trained_vs_pretrained.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["heldout_words", len(words), "WER_pct", "%.1f" % (100 * wer), "WER_nostress_pct", "%.1f" % (100 * wer_ns), "PER_pct", "%.1f" % (100 * per), "PER_nostress_pct", "%.1f" % (100 * per_ns)])
        w.writerow([])
        w.writerow(["oov_word", "trained_top1", "pretrained_top1", "edit_top1", "edit_best_of_3x3"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
