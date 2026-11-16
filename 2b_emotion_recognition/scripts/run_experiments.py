"""Run the full experiment grid (train + test for every pipeline / protocol / seed).

    python scripts/run_experiments.py --cache <cache folder>            # everything (about an hour on a GTX 1650)
    python scripts/run_experiments.py --cache <cache> --quick           # 1 seed, 8 epochs: smoke test

Grid
  protocol "word"  (main): speech, text, fusion {concat, gated, cross}      x 5 seeds
  protocol "random" (leaky, diagnostic): speech, text, fusion concat         x 3 seeds
  protocol "spk_OAF2YAF", "spk_YAF2OAF" (unseen speaker): same three models  x 3 seeds
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated subset of protocols to run, e.g. word,random")
    args = ap.parse_args()
    py = sys.executable
    jobs = [("word", ["speech", "text", "fusion:concat", "fusion:gated", "fusion:cross"], [0, 1, 2, 3, 4]),
            ("random", ["speech", "text", "fusion:concat"], [0, 1, 2]),
            ("spk_OAF2YAF", ["speech", "text", "fusion:concat"], [0, 1, 2]),
            ("spk_YAF2OAF", ["speech", "text", "fusion:concat"], [0, 1, 2])]
    keep = set(args.only.split(",")) if args.only else None
    for protocol, models, seeds in jobs:
        if keep and protocol not in keep:
            continue
        for m in models:
            name, _, mode = m.partition(":")
            extra = (["--fusion_mode", mode] if mode else [])
            sd = ["--seeds"] + [str(s) for s in (seeds[:1] if args.quick else seeds)]
            common = ["--protocol", protocol, "--cache", args.cache] + sd + extra
            ep = ["--epochs", "8"] if args.quick else ["--epochs", "30"]
            script = os.path.join("models", name + "_pipeline")
            run([py, os.path.join(script, "train.py")] + common + ep)
            run([py, os.path.join(script, "test.py")] + common)
    run([py, os.path.join("scripts", "make_tables.py")])


if __name__ == "__main__":
    main()
