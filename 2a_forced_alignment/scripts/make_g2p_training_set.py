"""Random subset of the stock pronunciation dictionary, used to TRAIN a small G2P model (extra experiment A9).

The full english_us_arpa dictionary has ~208k entries; training a pynini G2P model on all of them needs far more memory/time than a laptop
WSL session offers, so a seeded random sample of N words (all pronunciations of a sampled word are kept) is used.  The out-of-vocabulary
words of the corpus are NOT in the dictionary, so none of them can leak into the training set.

    python scripts/make_g2p_training_set.py --dict <stock english_us_arpa.dict> --n 25000 --out dictionaries/g2p_train_subset.dict
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", required=True)
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    by_word = {}
    for ln in open(a.dict, encoding="utf-8").read().splitlines():
        if ln.strip():
            by_word.setdefault(ln.split("\t")[0], []).append(ln)
    words = sorted(w for w in by_word if w.isalpha())              # plain alphabetic words (no apostrophes / digits)
    random.Random(a.seed).shuffle(words)
    keep = sorted(words[:a.n])
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for w in keep:
            f.write("\n".join(by_word[w]) + "\n")
    # 500 held-out words (single pronunciation, not in the training sample) for scripts/g2p_eval.py
    import os
    held = sorted(w for w in words[a.n:] if len(by_word[w]) == 1)
    random.Random(a.seed + 1).shuffle(held)
    held = sorted(held[:500])
    d = os.path.dirname(os.path.abspath(a.out))
    open(os.path.join(d, 'g2p_heldout_words.txt'), 'w', newline='\n').write('\n'.join(held) + '\n')
    open(os.path.join(d, 'g2p_heldout_ref.dict'), 'w', newline='\n').write('\n'.join(by_word[w][0] for w in held) + '\n')
    print("dictionary words %d | alphabetic %d | sampled %d | lines %d -> %s" % (len(by_word), len(words), len(keep), sum(len(by_word[w]) for w in keep), a.out))


if __name__ == "__main__":
    main()
