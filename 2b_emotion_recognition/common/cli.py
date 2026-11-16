"""Command-line front ends shared by models/<pipeline>/train.py and test.py."""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EPOCHS = {"speech": 40, "text": 25, "fusion": 40}


def _parser(modality, what):
    ap = argparse.ArgumentParser(description="%s %s pipeline" % (what, modality))
    ap.add_argument("--protocol", default="word", choices=["word", "random", "spk_OAF2YAF", "spk_YAF2OAF"],
                    help="word: word-disjoint train/val/test (default) | random: leaky clip-level split | spk_*: unseen speaker")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"), help="folder written by scripts/prepare_data.py")
    ap.add_argument("--device", default="auto")
    if modality == "fusion":
        ap.add_argument("--fusion_mode", default="concat", choices=["concat", "gated", "cross"])
    return ap


def train_main(modality):
    from common.engine import train_one
    ap = _parser(modality, "Train the")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS[modality])
    ap.add_argument("--lr", type=float, default=None)
    if modality == "fusion":
        ap.add_argument("--init_unimodal", action="store_true", help="start the encoders from the trained speech/text models")
    a = ap.parse_args()
    mode = getattr(a, "fusion_mode", "concat")
    lr = a.lr or (2e-3 if modality == "text" else 1e-3)
    for seed in a.seeds:
        print("== %s | protocol %s | seed %d%s" % (modality, a.protocol, seed, (" | fusion " + mode) if modality == "fusion" else ""), flush=True)
        best = train_one(modality, a.protocol, seed, a.cache, epochs=a.epochs, lr=lr, fusion_mode=mode, device=a.device,
                         init_unimodal=getattr(a, "init_unimodal", False))
        print("   best validation accuracy: %.3f" % best, flush=True)


def test_main(modality):
    from common.data import EMOTIONS
    from common.engine import evaluate_run
    ap = _parser(modality, "Evaluate the")
    a = ap.parse_args()
    mode = getattr(a, "fusion_mode", "concat")
    res = [evaluate_run(modality, a.protocol, s, a.cache, mode, a.device) for s in a.seeds]
    acc = np.array([r["accuracy"] for r in res])
    f1 = np.array([r["macro_f1"] for r in res])
    tag = a.protocol + ("_" + mode if modality == "fusion" else "")
    out_dir = os.path.join(ROOT, "Results", "tables")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "%s_%s_per_seed.csv" % (modality, tag)), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modality", "protocol", "seed", "n_test", "accuracy", "macro_f1"] + ["recall_" + e for e in EMOTIONS])
        for r in res:
            w.writerow([modality, a.protocol, r["seed"], r["n"], "%.4f" % r["accuracy"], "%.4f" % r["macro_f1"]] + ["%.3f" % r["per_class"][e]["recall"] for e in EMOTIONS])
    print("%s | %s | %d seeds | test accuracy %.1f%% +- %.1f | macro-F1 %.3f +- %.3f | chance 14.3%%" % (modality, tag, len(res), 100 * acc.mean(), 100 * acc.std(), f1.mean(), f1.std()))
    for r in res:
        print("   seed %d: accuracy %.3f  macro-F1 %.3f  (best epoch %d)" % (r["seed"], r["accuracy"], r["macro_f1"], r["best_epoch"]))
    # mean confusion matrix over the seeds
    from common.plots import plot_confusion
    cm = np.mean([np.array(r["confusion"]) for r in res], axis=0)
    plot_confusion(cm, os.path.join(ROOT, "Results", "plots", "confusion_%s_%s.png" % (modality, tag)),
                   "%s (%s): mean over %d seeds, accuracy %.1f%%" % (modality, tag, len(res), 100 * acc.mean()))
