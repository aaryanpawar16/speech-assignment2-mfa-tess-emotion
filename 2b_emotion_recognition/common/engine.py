"""Shared training / evaluation engine used by the train.py and test.py of the three pipelines."""
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.data import EMOTIONS, MAX_FRAMES, load_metadata, make_split
from common.models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = {"speech": "speech_pipeline", "text": "text_pipeline", "fusion": "fusion_pipeline"}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name="auto"):
    return torch.device("cuda" if (name == "auto" and torch.cuda.is_available()) else ("cpu" if name == "auto" else name))


class Bundle:
    """All clips as tensors on the device: speech log-mel, frozen text features, labels."""

    def __init__(self, cache, device, meta_path=None, cfg=None):
        """cfg (extras, all optional): feat = "logmel" (default) | "w2v" (frozen wav2vec2 states, scripts/extract_ssl.py);
        text = "distilbert" (default) | "synth_p060" (synthetic emotion cue, scripts/make_synthetic_text.py);
        spk_norm = standardise the features of each speaker with that speaker's own statistics (no labels used)."""
        cfg = cfg or {}
        self.device = device
        self.spk_norm = bool(cfg.get("spk_norm", False))
        self.meta = load_metadata(meta_path or os.path.join(ROOT, "data", "metadata.csv"))
        feat, text = cfg.get("feat", "logmel"), cfg.get("text", "distilbert")
        sp = np.load(os.path.join(cache, "speech_%s.npz" % feat))
        tx = np.load(os.path.join(cache, "text_%s.npz" % text))
        self.speech = torch.from_numpy(sp["X"]).to(device)                        # (N,300,80) float16
        self.lengths = torch.from_numpy(sp["lengths"]).to(device)
        self.max_frames, self.n_feat = self.speech.shape[1], self.speech.shape[2]
        self.text = torch.from_numpy(tx["X"]).to(device)                          # (N,10,768) float16
        self.text_mask = torch.from_numpy(tx["mask"]).to(device)
        self.y = torch.tensor([EMOTIONS.index(m["emotion"]) for m in self.meta], device=device)
        self.mean = torch.zeros(self.n_feat, device=device)
        self.std = torch.ones(self.n_feat, device=device)
        if self.spk_norm:
            names = sorted({m["speaker"] for m in self.meta})
            self.spk = torch.tensor([names.index(m["speaker"]) for m in self.meta], device=device)
            self.smean, self.sstd = [], []
            for k in range(len(names)):
                mu, sd = self._stats(torch.nonzero(self.spk == k).squeeze(1))
                self.smean.append(mu)
                self.sstd.append(sd)
            self.smean, self.sstd = torch.stack(self.smean), torch.stack(self.sstd)

    def _stats(self, idx):
        x = self.speech[idx].float()
        m = (torch.arange(self.max_frames, device=self.device)[None] < self.lengths[idx][:, None])
        sel = x[m]                                                               # valid frames only
        return sel.mean(0), sel.std(0) + 1e-5

    def fit_standardiser(self, train_idx):
        if not self.spk_norm:
            self.mean, self.std = self._stats(torch.tensor(train_idx, device=self.device))

    def batch(self, idx, augment=False, speech_override=None, specaug=True):
        idx = torch.as_tensor(idx, device=self.device)
        mask = torch.arange(self.max_frames, device=self.device)[None] < self.lengths[idx][:, None]
        x = (self.speech[idx].float() if speech_override is None else speech_override)
        if self.spk_norm:
            x = (x - self.smean[self.spk[idx]][:, None]) / self.sstd[self.spk[idx]][:, None]
        else:
            x = (x - self.mean) / self.std
        x = x * mask.unsqueeze(-1)
        if augment and specaug:
            x = spec_augment(x, self.lengths[idx])
        return {"speech": x, "speech_mask": mask, "text": self.text[idx].float(), "text_mask": self.text_mask[idx], "y": self.y[idx]}


def spec_augment(x, lengths, n_freq=2, max_f=12, n_time=2, max_t=30):
    """SpecAugment: random frequency and time masks (applied per example on the standardised log-mel)."""
    B, T, Fq = x.shape
    if Fq != 80:                                    # scale the masks to other feature sizes / frame rates
        max_f = int(0.15 * Fq)
    max_t = T // 10
    x = x.clone()
    for b in range(B):
        for _ in range(n_freq):
            w = random.randint(0, max_f)
            f0 = random.randint(0, Fq - w)
            x[b, :, f0:f0 + w] = 0
        for _ in range(n_time):
            w = random.randint(0, max_t)
            t0 = random.randint(0, max(0, int(lengths[b]) - w))
            x[b, t0:t0 + w, :] = 0
    return x


def run_dir(modality, protocol, seed, fusion_mode="concat", base=None):
    tag = protocol + ("_" + fusion_mode if modality == "fusion" and fusion_mode != "concat" else "")
    return os.path.join(base or os.path.join(ROOT, "models", PIPELINE_DIR[modality], "runs"), tag, "seed%d" % seed)


@torch.no_grad()
def predict(model, bundle, idx, bs=128, return_reps=False, speech_override=None):
    model.eval()
    logits, reps = [], {}
    for s in range(0, len(idx), bs):
        sel = idx[s:s + bs]
        ov = None if speech_override is None else speech_override[s:s + bs]
        out = model(bundle.batch(sel, speech_override=ov), return_reps=return_reps)
        if return_reps:
            out, r = out
            for k, v in r.items():
                reps.setdefault(k, []).append(v.cpu())
        logits.append(out.cpu())
    logits = torch.cat(logits)
    if return_reps:
        return logits, {k: torch.cat(v).numpy() for k, v in reps.items()}
    return logits


def metrics(y_true, logits):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    y_pred = logits.argmax(1).numpy() if torch.is_tensor(logits) else np.asarray(logits).argmax(1)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(EMOTIONS))), zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(EMOTIONS))), zero_division=0)),
            "per_class": {e: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, e in enumerate(EMOTIONS)},
            "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS)))).tolist()}


def model_cfg(cfg, bundle):
    """SpeechEncoder options from the experiment cfg (defaults = the main model)."""
    cfg = cfg or {}
    out = {"n_mels": bundle.n_feat, "arch": cfg.get("arch", "full"), "pool": cfg.get("pool", "attn")}
    if cfg.get("moddrop"):
        out["moddrop"] = cfg["moddrop"]
    return out


def train_one(modality, protocol, seed, cache, epochs=40, lr=1e-3, wd=1e-2, bs=32, fusion_mode="concat", patience=15,
              device="auto", out_base=None, init_unimodal=False, verbose=True, cfg=None):
    """Train one model on the train split, select the epoch with the best validation accuracy, save it."""
    dev = get_device(device)
    seed_everything(seed)
    cfg = cfg or {}
    bundle = Bundle(cache, dev, cfg=cfg)
    split = make_split(bundle.meta, protocol, seed)
    bundle.fit_standardiser(split["train"])
    model = build_model(modality, fusion_mode, model_cfg(cfg, bundle)).to(dev)
    if modality == "fusion" and init_unimodal:                    # start both encoders from the unimodal models
        for name, mod in (("speech", model.speech), ("text", model.text)):
            p = os.path.join(run_dir(name, protocol, seed, base=out_base), "best.pt")
            if os.path.exists(p):
                mod.load_state_dict({k[4:]: v for k, v in torch.load(p, map_location=dev)["model"].items() if k.startswith("enc.")})
                if verbose:
                    print("  initialised %s encoder from %s" % (name, p))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n_train = len(split["train"])
    steps_per_epoch = math.ceil(n_train / bs)
    total, warm = epochs * steps_per_epoch, 2 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total))) * 0.99 + 0.01))
    out = run_dir(modality, protocol, seed, fusion_mode, out_base)
    os.makedirs(out, exist_ok=True)
    hist, best, bad, t0 = [], (-1, 1e9), 0, time.time()
    y_val = bundle.y[split["val"]].cpu().numpy()
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(split["train"])
        tl = tc = 0
        for s in range(0, n_train, bs):
            sel = perm[s:s + bs]
            if len(sel) < 2:
                continue
            batch = bundle.batch(sel, augment=(modality != "text"), specaug=cfg.get("specaug", True))
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["y"], label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(sel)
            tc += (logits.argmax(1) == batch["y"]).sum().item()
        v = predict(model, bundle, split["val"])
        vl = F.cross_entropy(v, bundle.y[split["val"]].cpu()).item()
        va = float((v.argmax(1).numpy() == y_val).mean())
        hist.append({"epoch": ep, "train_loss": tl / n_train, "train_acc": tc / n_train, "val_loss": vl, "val_acc": va})
        if (va, -vl) > (best[0], -best[1]):
            best, bad = (va, vl), 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val_acc": va, "protocol": protocol, "seed": seed, "modality": modality,
                        "fusion_mode": fusion_mode, "mean": bundle.mean.cpu(), "std": bundle.std.cpu(), "cfg": cfg}, os.path.join(out, "best.pt"))
        else:
            bad += 1
        if verbose and (ep % 5 == 0 or ep == 1):
            print("  ep %2d | train loss %.3f acc %.3f | val loss %.3f acc %.3f | best val acc %.3f | %.0fs" % (ep, tl / n_train, tc / n_train, vl, va, best[0], time.time() - t0), flush=True)
        if bad >= patience:
            break
    json.dump({"history": hist, "best_val_acc": best[0], "epochs_run": len(hist), "n_train": n_train, "n_val": len(split["val"]), "protocol": protocol, "seed": seed,
               "seconds": time.time() - t0}, open(os.path.join(out, "history.json"), "w"), indent=1)
    return best[0]


def load_trained(modality, protocol, seed, cache, fusion_mode="concat", device="auto", out_base=None):
    """Load a saved model + bundle (standardiser restored from the checkpoint) and its split."""
    dev = get_device(device)
    ck = torch.load(os.path.join(run_dir(modality, protocol, seed, fusion_mode, out_base), "best.pt"), map_location=dev, weights_only=False)
    cfg = ck.get("cfg", {})                       # checkpoints of the main experiments have no cfg (= defaults)
    bundle = Bundle(cache, dev, cfg=cfg)
    bundle.mean, bundle.std = ck["mean"].to(dev), ck["std"].to(dev)
    model = build_model(modality, fusion_mode, model_cfg(cfg, bundle)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, bundle, make_split(bundle.meta, protocol, seed), ck


def evaluate_run(modality, protocol, seed, cache, fusion_mode="concat", device="auto", out_base=None, split_name="test"):
    model, bundle, split, ck = load_trained(modality, protocol, seed, cache, fusion_mode, device, out_base)
    idx = split[split_name]
    logits = predict(model, bundle, idx)
    y = bundle.y[idx].cpu().numpy()
    m = metrics(y, logits)
    m.update({"protocol": protocol, "seed": seed, "modality": modality, "n": len(idx), "best_epoch": ck["epoch"], "val_acc": ck["val_acc"]})
    out = run_dir(modality, protocol, seed, fusion_mode, out_base)
    json.dump(m, open(os.path.join(out, "%s_metrics.json" % split_name), "w"), indent=1)
    probs = torch.softmax(logits, 1).numpy()
    with open(os.path.join(out, "%s_predictions.csv" % split_name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "file", "speaker", "word", "true", "pred", "confidence"] + ["p_" + e for e in EMOTIONS])
        for k, i in enumerate(idx):
            mm = bundle.meta[i]
            w.writerow([i, mm["file"], mm["speaker"], mm["word"], EMOTIONS[y[k]], EMOTIONS[probs[k].argmax()], "%.3f" % probs[k].max()] + ["%.3f" % p for p in probs[k]])
    return m
