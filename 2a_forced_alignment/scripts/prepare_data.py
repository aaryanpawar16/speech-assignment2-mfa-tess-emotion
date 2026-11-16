"""Organise the provided audio + transcripts into the folder layout the Montreal Forced Aligner expects.

MFA wants, for every utterance, an audio file and a transcript with the same basename, and treats every
sub-directory as one speaker:

    corpus_<variant>/<speaker>/<utt>.wav
    corpus_<variant>/<speaker>/<utt>.lab

Two transcript variants are written (so their effect on the alignment can be compared):

  naive : the transcript exactly as provided, only whitespace/newlines collapsed.
  norm  : "spoken form" - dotted acronyms and the radio-station call sign are spelled out letter by
          letter, because the speaker says them that way (verified with an independent ASR):
          S.J.C.'s -> S J C's,  WBUR's -> W B U R's ;  "seventy- six" -> "seventy six".

    python prepare_data.py --data ../data --out ..
"""
import argparse
import os
import re
import shutil

ACRONYMS = {"S.J.C.": "S J C", "WBUR": "W B U R"}           # applied to the news-reading speaker only


def speaker_of(stem):
    return "ISLE_SESS0131" if stem.startswith("ISLE_") else stem[:3]      # F2BJRLP1 -> F2B


def clean_naive(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_norm(text):
    text = clean_naive(text)
    for a, spoken in ACRONYMS.items():
        text = text.replace(a, spoken)
    text = re.sub(r"(\w)- (\w)", r"\1 \2", text)                 # "seventy- six" -> "seventy six"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="..")
    args = ap.parse_args()
    wav_dir, txt_dir = os.path.join(args.data, "wav"), os.path.join(args.data, "transcripts")
    txts = {os.path.splitext(f)[0]: os.path.join(txt_dir, f) for f in os.listdir(txt_dir)}
    n = 0
    for variant, fn in (("naive", clean_naive), ("norm", clean_norm)):
        for f in sorted(os.listdir(wav_dir)):
            stem = os.path.splitext(f)[0]
            spk_dir = os.path.join(args.out, "corpus_" + variant, speaker_of(stem))
            os.makedirs(spk_dir, exist_ok=True)
            shutil.copyfile(os.path.join(wav_dir, f), os.path.join(spk_dir, stem + ".wav"))
            text = open(txts[stem], encoding="utf-8", errors="replace").read()
            with open(os.path.join(spk_dir, stem + ".lab"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(fn(text) + "\n")
            n += 1
    print("wrote %d utterances (2 variants) under %s" % (n // 2, os.path.abspath(args.out)))
    for variant in ("naive", "norm"):
        p = os.path.join(args.out, "corpus_" + variant, "F2B", "F2BJRLP1.lab")
        print("[%s] %s" % (variant, open(p, encoding="utf-8").read()[:150].strip()))


if __name__ == "__main__":
    main()
