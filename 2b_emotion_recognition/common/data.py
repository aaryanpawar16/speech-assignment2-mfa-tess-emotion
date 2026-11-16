"""Data handling for the TESS multimodal emotion-recognition experiments.

TESS (Toronto Emotional Speech Set): two actresses (OAF: older, YAF: younger) each say the carrier
phrase "Say the word ____" for 200 target words in 7 emotions -> 2 x 200 x 7 = 2,800 clips.
The dataset has no transcript files; the text of a clip is its carrier phrase with the target word
(taken from the file name).  Every target word occurs equally often in every emotion, so the text
carries NO information about the emotion by construction (see the report).

Kaggle's copy contains every file twice (a nested duplicate folder) and two neutral files of OAF are
named "OA_*"; build_metadata() de-duplicates and repairs these.
"""
import csv
import glob
import os
import random

import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
EMO_FROM_NAME = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "neutral": "neutral",
                 "ps": "pleasant_surprise", "sad": "sad"}
SR = 16000
HOP = 160                      # 10 ms
N_MELS = 80
MAX_FRAMES = 300               # 3 s; the longest trimmed TESS clip is shorter


def carrier_text(word):
    return "Say the word %s." % word


def build_metadata(tess_root):
    """Scan the TESS folder -> list of dict(path, speaker, word, emotion, text), de-duplicated, sorted."""
    found = {}
    for p in sorted(glob.glob(os.path.join(tess_root, "**", "*.wav"), recursive=True)):
        stem = os.path.splitext(os.path.basename(p))[0]
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        spk, word, emo = parts[0].upper(), parts[1].lower(), parts[2].lower()
        spk = "OAF" if spk == "OA" else spk                                   # two mislabelled neutral files
        if emo not in EMO_FROM_NAME or spk not in ("OAF", "YAF"):
            continue
        key = (spk, word, EMO_FROM_NAME[emo])
        nested = p.count("TESS Toronto emotional speech set data") > 1       # prefer the top-level copy
        if key not in found or (found[key][1] and not nested):
            found[key] = (p, nested)
    rows = [dict(path=v[0], speaker=k[0], word=k[1], emotion=k[2], text=carrier_text(k[1])) for k, v in sorted(found.items())]
    return rows


def save_metadata(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "speaker", "word", "emotion", "text"])
        w.writeheader()
        w.writerows(rows)


def load_metadata(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


# ----------------------------------------------------------------------------------------- splits
def make_split(meta, protocol="word", seed=0):
    """Return dict(train=[idx], val=[idx], test=[idx]).

    word        : the 200 target words are divided 70/10/20 into train/val/test (the same word is never in two
                  sets, so a model cannot memorise word-specific acoustics); both speakers are in every set.
    spk_OAF2YAF : train + validate on OAF (validation = 10% of the words), test on all of YAF (unseen speaker).
    spk_YAF2OAF : the reverse.
    """
    rng = random.Random(seed)
    words = sorted({m["word"] for m in meta})
    rng.shuffle(words)
    idx = lambda f: [i for i, m in enumerate(meta) if f(m)]
    if protocol == "word":
        n = len(words)
        tr, va = set(words[: int(0.7 * n)]), set(words[int(0.7 * n): int(0.8 * n)])
        return dict(train=idx(lambda m: m["word"] in tr), val=idx(lambda m: m["word"] in va),
                    test=idx(lambda m: m["word"] not in tr and m["word"] not in va))
    if protocol == "random":        # clip-level stratified 70/10/20: the SAME words occur in train and test (leaky, diagnostic only)
        out = dict(train=[], val=[], test=[])
        groups = {}
        for i, m in enumerate(meta):
            groups.setdefault((m["speaker"], m["emotion"]), []).append(i)
        for k in sorted(groups):
            g = groups[k]
            rng.shuffle(g)
            n = len(g)
            out["train"] += g[: int(0.7 * n)]
            out["val"] += g[int(0.7 * n): int(0.8 * n)]
            out["test"] += g[int(0.8 * n):]
        return out
    if protocol.startswith("spk_"):
        src, dst = protocol[4:].split("2")
        va = set(words[: int(0.1 * len(words))])
        return dict(train=idx(lambda m: m["speaker"] == src and m["word"] not in va),
                    val=idx(lambda m: m["speaker"] == src and m["word"] in va),
                    test=idx(lambda m: m["speaker"] == dst))
    raise ValueError(protocol)


# ----------------------------------------------------------------------------------------- speech features
def load_audio(path, sr=SR):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)                                # resamples 24.4 kHz -> 16 kHz
    return y


def preprocess_audio(y, top_db=30):
    """Trim leading/trailing silence, peak-normalise."""
    import librosa
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) < 1600:
        yt = y
    return yt / (np.abs(yt).max() + 1e-8)


def logmel(y, sr=SR):
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=HOP, n_mels=N_MELS, fmin=20, fmax=8000)
    return np.log(m + 1e-6).T.astype(np.float32)                                # (time_steps, n_mels)


def speech_features(path):
    return logmel(preprocess_audio(load_audio(path)))


def pad_or_crop(x, T=MAX_FRAMES):
    """(t, f) -> (T, f) plus a boolean mask of valid frames."""
    t = min(len(x), T)
    out = np.zeros((T, x.shape[1]), np.float32)
    out[:t] = x[:t]
    mask = np.zeros(T, bool)
    mask[:t] = True
    return out, mask
