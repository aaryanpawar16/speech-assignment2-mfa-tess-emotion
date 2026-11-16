"""Collect the per-seed result files into the accuracy tables in Results/ (CSV + Markdown)."""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
ORDER = ["speech_word", "text_word", "fusion_word_concat", "fusion_word_gated", "fusion_word_cross"]


def main():
    rows = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(ROOT, "Results", "tables", "*_per_seed.csv"))):
        name = os.path.basename(p).replace("_per_seed.csv", "")
        for r in csv.DictReader(open(p, encoding="utf-8")):
            rows[name].append(r)
    table = []
    for name, rs in rows.items():
        acc = np.array([float(r["accuracy"]) for r in rs]) * 100
        f1 = np.array([float(r["macro_f1"]) for r in rs])
        rec = {e: np.mean([float(r["recall_" + e]) for r in rs]) for e in EMOS}
        modality, rest = name.split("_", 1)
        protocol = rest.rsplit("_", 1)[0] if modality == "fusion" else rest
        mode = rest.rsplit("_", 1)[1] if modality == "fusion" else "-"
        if modality == "fusion" and protocol in ("spk_OAF2YAF", "spk_YAF2OAF", "word", "random") is False:
            pass
        table.append(dict(modality=modality, protocol=rest if modality != "fusion" else rest, fusion=mode, seeds=len(rs), n_test=rs[0]["n_test"],
                          acc_mean=acc.mean(), acc_std=acc.std(), f1_mean=f1.mean(), f1_std=f1.std(), recall=rec))
    table.sort(key=lambda t: (t["protocol"].split("_")[0] if t["modality"] != "fusion" else t["protocol"].rsplit("_", 1)[0], ["speech", "text", "fusion"].index(t["modality"]), t["fusion"]))
    # CSV
    with open(os.path.join(ROOT, "Results", "accuracy_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modality", "protocol", "fusion_block", "seeds", "n_test", "accuracy_mean_%", "accuracy_std_%", "macro_f1_mean", "macro_f1_std"] + ["recall_" + e for e in EMOS])
        for t in table:
            w.writerow([t["modality"], t["protocol"], t["fusion"], t["seeds"], t["n_test"], "%.2f" % t["acc_mean"], "%.2f" % t["acc_std"], "%.4f" % t["f1_mean"], "%.4f" % t["f1_std"]] + ["%.3f" % t["recall"][e] for e in EMOS])
    # Markdown
    md = ["| Model | Protocol | Fusion block | Seeds | Test clips | Accuracy % (mean +- std) | Macro-F1 (mean +- std) |", "|---|---|---|---|---|---|---|"]
    for t in table:
        md.append("| %s | %s | %s | %d | %s | %.1f +- %.1f | %.3f +- %.3f |" % (t["modality"], t["protocol"], t["fusion"], t["seeds"], t["n_test"], t["acc_mean"], t["acc_std"], t["f1_mean"], t["f1_std"]))
    md += ["", "Per-emotion recall (mean over seeds):", "", "| Model | Protocol | " + " | ".join(EMOS) + " |", "|---|---|" + "---|" * len(EMOS)]
    for t in table:
        md.append("| %s%s | %s | " % (t["modality"], "" if t["fusion"] == "-" else " (%s)" % t["fusion"], t["protocol"]) + " | ".join("%.2f" % t["recall"][e] for e in EMOS) + " |")
    open(os.path.join(ROOT, "Results", "accuracy_summary.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md[:2 + len(table)]))


if __name__ == "__main__":
    main()
