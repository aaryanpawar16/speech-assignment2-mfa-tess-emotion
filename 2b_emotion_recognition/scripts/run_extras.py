"""Extra experiments (beyond the main grid): speaker normalisation, ablations, self-supervised speech features,
and the synthetic-text sanity check of the fusion code.  Every variant is trained and tested on the two unseen-speaker
protocols (OAF->YAF, YAF->OAF), 3 seeds, and summarised in Results/extras/extras_summary.md.

    python scripts/run_extras.py --cache <cache> --variants spkn noaug meanpool cnn gru
    python scripts/run_extras.py --cache <cache> --variants w2v w2v_spkn          # needs scripts/extract_ssl.py first
    python scripts/run_extras.py --cache <cache> --variants synth_text synth_speech_fusion_concat ...   # needs scripts/make_synthetic_text.py
    python scripts/run_extras.py --summary                                         # only rebuild the tables

variant -> (modality, fusion block, options); options: spk_norm, feat, text, specaug, arch, pool  (see common/engine.py)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SYN = "synth_p060"
VARIANTS = {
    # --- speaker normalisation and ablations of the speech pipeline
    "spkn":     ("speech", "concat", {"spk_norm": True}),
    "noaug":    ("speech", "concat", {"specaug": False}),
    "meanpool": ("speech", "concat", {"pool": "mean"}),
    "cnn":      ("speech", "concat", {"arch": "cnn"}),
    "gru":      ("speech", "concat", {"arch": "gru"}),
    # --- frozen wav2vec2 features instead of log-mel
    "w2v":      ("speech", "concat", {"feat": "w2v"}),
    "w2v_spkn": ("speech", "concat", {"feat": "w2v", "spk_norm": True}),
    # --- fusion on top of the improved speech front-end
    "fusion_spkn":     ("fusion", "concat", {"spk_norm": True}),
    "fusion_w2v":      ("fusion", "concat", {"feat": "w2v"}),      # control for the synthetic-text rows: same model, uninformative (real) text
    "fusion_w2v_spkn": ("fusion", "concat", {"feat": "w2v", "spk_norm": True}),
    # --- synthetic text that DOES carry the emotion (cue correct for 60% of the clips): does the fusion code exploit it?
    "synth_text":          ("text", "concat", {"text": SYN}),
    "synth_fusion_concat": ("fusion", "concat", {"text": SYN}),
    "synth_fusion_gated":  ("fusion", "gated", {"text": SYN}),
    "synth_fusion_cross":  ("fusion", "cross", {"text": SYN}),
    # --- modality dropout (hide one modality at random during training) so that the fusion model cannot rely on the speech alone
    "synth_fusion_concat_md":     ("fusion", "concat", {"text": SYN, "moddrop": 0.3}),
    "synth_fusion_gated_md":      ("fusion", "gated", {"text": SYN, "moddrop": 0.3}),
    "synth_w2v_fusion_concat":    ("fusion", "concat", {"text": SYN, "feat": "w2v"}),
    "synth_w2v_fusion_concat_md": ("fusion", "concat", {"text": SYN, "feat": "w2v", "moddrop": 0.3}),
}
PROTOCOLS = ["spk_OAF2YAF", "spk_YAF2OAF"]
EPOCHS = {"speech": 30, "text": 30, "fusion": 30}
OUT = os.path.join(ROOT, "Results", "extras")


def base_dir(name, modality):
    pipe = {"speech": "speech_pipeline", "text": "text_pipeline", "fusion": "fusion_pipeline"}[modality]
    return os.path.join(ROOT, "models", pipe, "runs_extras", name)


def run_variant(name, protocols, seeds, cache, device):
    from common.engine import evaluate_run, train_one
    modality, mode, cfg = VARIANTS[name]
    lr = 2e-3 if modality == "text" else 1e-3
    os.makedirs(OUT, exist_ok=True)
    for prot in protocols:
        rows = []
        for seed in seeds:
            print("== %s | %s | seed %d" % (name, prot, seed), flush=True)
            train_one(modality, prot, seed, cache, epochs=EPOCHS[modality], lr=lr, fusion_mode=mode, device=device, out_base=base_dir(name, modality), cfg=cfg, verbose=False)
            r = evaluate_run(modality, prot, seed, cache, mode, device, out_base=base_dir(name, modality))
            print("   accuracy %.3f  macro-F1 %.3f  (best epoch %d)" % (r["accuracy"], r["macro_f1"], r["best_epoch"]), flush=True)
            rows.append({"seed": seed, "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "recall": {e: v["recall"] for e, v in r["per_class"].items()}, "confusion": r["confusion"]})
        json.dump({"variant": name, "protocol": prot, "modality": modality, "fusion": mode, "cfg": cfg, "runs": rows}, open(os.path.join(OUT, "%s__%s.json" % (name, prot)), "w"), indent=1)


def summary():
    import glob
    from common.data import EMOTIONS
    res = {}
    for f in glob.glob(os.path.join(OUT, "*__*.json")):
        d = json.load(open(f))
        res[(d["variant"], d["protocol"])] = d
    # the main experiments (already in Results/accuracy_summary.csv) as reference rows
    ref = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "Results", "accuracy_summary.csv"), encoding="utf-8")):
        ref[(r["modality"], r["protocol"])] = r
    lines = ["| Variant | OAF -> YAF acc. % | YAF -> OAF acc. % | mean of the two | macro-F1 (OAF->YAF / YAF->OAF) |", "|---|---|---|---|---|"]
    for name, label in [("base_speech", "speech, log-mel (main model)"), ("base_fusion", "fusion concat, log-mel (main model)")] + [(k, k) for k in VARIANTS]:
        cells, means, f1s = [], [], []
        for prot in PROTOCOLS:
            if name.startswith("base_"):
                r = ref.get((name[5:], prot if name == "base_speech" else prot + "_concat"))
                if r is None:
                    break
                a, s, f = float(r["accuracy_mean_%"]), float(r["accuracy_std_%"]), float(r["macro_f1_mean"])
            else:
                if (name, prot) not in res:
                    break
                v = np.array([x["accuracy"] for x in res[(name, prot)]["runs"]]) * 100
                a, s, f = v.mean(), v.std(), np.mean([x["macro_f1"] for x in res[(name, prot)]["runs"]])
            cells.append("%.1f +- %.1f" % (a, s))
            means.append(a)
            f1s.append("%.3f" % f)
        if len(cells) == 2:
            lines.append("| %s | %s | %s | %.1f | %s |" % (label, cells[0], cells[1], np.mean(means), " / ".join(f1s)))
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "extras_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # per-emotion recall of the extras (mean over seeds and the two protocols)
    out = ["| Variant | " + " | ".join(EMOTIONS) + " |", "|---|" + "---|" * len(EMOTIONS)]
    for name in VARIANTS:
        if all((name, p) in res for p in PROTOCOLS):
            rec = [np.mean([np.mean([x["recall"][e] for x in res[(name, p)]["runs"]]) for p in PROTOCOLS]) for e in EMOTIONS]
            out.append("| %s | %s |" % (name, " | ".join("%.2f" % x for x in rec)))
    open(os.path.join(OUT, "extras_per_emotion.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    for v in a.variants:
        run_variant(v, a.protocols, a.seeds, a.cache, a.device)
    summary()


if __name__ == "__main__":
    main()
