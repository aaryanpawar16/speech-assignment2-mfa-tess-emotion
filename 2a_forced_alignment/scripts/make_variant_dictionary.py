"""Dictionary with pronunciation VARIANTS for the minimal-pair words of the non-native speaker.

The ISLE sentences are "I said WHITE not BAIT", "... NEW not NO", "... BAD not BED".  A learner may merge the two
vowels of a pair.  If the dictionary offers both the canonical and the confusable vowel for each word, the aligner picks
(per occurrence, by acoustic likelihood) the variant that fits the audio better, which turns the alignment into a
pronunciation-variant detector.  Written to dictionaries/english_us_arpa_variants.dict = custom dictionary + variants.

    python scripts/make_variant_dictionary.py
"""
import argparse
import os

# word -> additional pronunciations (ARPAbet, stress digit on the vowel)
VARIANTS = {
    "white": ["W EY1 T"],        # canonical W AY1 T   (the speaker might say "bait")
    "bait": ["B AY1 T"],
    "new": ["N OW1"],            # canonical N UW1
    "no": ["N UW1"],
    "bad": ["B EH1 D"],          # canonical B AE1 D
    "bed": ["B AE1 D"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    src = os.path.join(a.root, "dictionaries", "english_us_arpa_custom.dict")
    dst = os.path.join(a.root, "dictionaries", "english_us_arpa_variants.dict")
    with open(src, encoding="utf-8") as f:
        lines = f.read().splitlines()
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
        for w, prons in VARIANTS.items():
            for p in prons:
                f.write("%s\t%s\n" % (w, p))
    print("wrote", dst, "| %d extra variants" % sum(len(v) for v in VARIANTS.values()))


if __name__ == "__main__":
    main()
