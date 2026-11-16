# Multimodal emotion recognition on TESS (speech, text, fusion)

Assignment 2b: three pipelines on the Toronto Emotional Speech Set (2 speakers x 7 emotions x 200 words = 2,800 clips),
each built from the same blocks: **preprocessing -> feature extraction -> temporal/contextual modelling -> fusion -> classifier**.

| Pipeline | Input | Temporal / contextual block | Fusion | 
|---|---|---|---|
| `models/speech_pipeline` | 80-bin log-mel | Conv1d x2 + 2-layer BiGRU + attention pooling | - |
| `models/text_pipeline` | frozen DistilBERT states of "Say the word ___" | Linear + BiGRU + attention pooling | - |
| `models/fusion_pipeline` | both | both encoders | `concat`, `gated` or `cross` (cross-attention) |

All pipelines end in the same MLP classifier (7 emotions). The report (`Report/Report_2b.pdf`) gives the architecture
decisions, the experiments and the analysis.

> **Important property of the data.** Every TESS clip is the carrier phrase "Say the word ___". Each sentence occurs
> once per emotion and speaker, so **the text contains no information about the emotion**: text-only is at chance
> (14.3%) by construction and fusion cannot gain from the *meaning* of the words. The results are reported as they are.

## Results (test accuracy %, mean +- std over seeds)

| Model | word-disjoint (5 seeds) | OAF -> YAF (3) | YAF -> OAF (3) | random split, leaky (3) |
|---|---|---|---|---|
| Speech only | 99.8 +- 0.2 | 48.6 +- 8.1 | 40.2 +- 5.7 | 99.8 +- 0.2 |
| Text only | 14.3 +- 0.0 | 14.3 +- 0.0 | 14.3 +- 0.0 | 13.8 +- 0.6 |
| Fusion (concat) | 99.9 +- 0.1 | 51.9 +- 5.0 | 44.0 +- 8.0 | 99.9 +- 0.1 |
| Fusion (gated) | 99.8 +- 0.2 | | | |
| Fusion (cross-attention) | 99.8 +- 0.3 | | | |

* **word-disjoint** (main protocol): the 200 words are split 70/10/20 into train/val/test, so no word is in two sets; both speakers are in every set.
* **OAF -> YAF / YAF -> OAF**: train on one speaker, test on all 1,400 clips of the other (unseen speaker).
* **random**: clip-level split where the same word is in train and test; shown only as a diagnostic of leakage.

Full tables (with macro-F1 and per-emotion recall): `Results/accuracy_summary.md` / `.csv`; per-seed tables: `Results/tables/`;
confusion matrices of every variant: `Results/plots/confusion_*.png`.

Other analyses (scripts below): noise robustness (`Results/plots/noise_robustness.png`, `Results/tables/noise_robustness.csv`),
t-SNE + silhouette/k-NN of the Temporal, Contextual and Fusion block outputs (`Results/representations/`),
failure cases (`Results/error_analysis_*.md`, `Results/plots/failure_*.png`), acoustic profile per speaker and emotion (`Results/tables/acoustic_profile.csv`).

## Extra experiments (unseen-speaker protocols, 3 seeds)

Mean of OAF -> YAF and YAF -> OAF accuracy (full table with std and macro-F1: `Results/extras/extras_summary.md`, per-emotion recall: `extras_per_emotion.md`):

| Variant | mean acc. % |
|---|---|
| Main speech model (log-mel, attention pooling) | 44.4 |
| no SpecAugment | 33.1 |
| mean pooling instead of attention | 56.7 (seen on the test speaker, so not adopted as the main model) |
| CNN only / BiGRU only | 42.6 / 42.2 |
| per-speaker standardisation | 44.4 (no gain) |
| **frozen wav2vec2-base features (layers 6-9)** | **66.6** (angry recall drops to 0.02) |
| synthetic text that states the emotion (60% correct): text only | 60.9 |
| ... fusion concat / gated / cross, log-mel speech | 47.1 / 41.5 / 42.8 (text not exploited) |
| ... fusion concat + modality dropout | 54.0 |
| ... fusion concat, wav2vec2 speech | 70.8 (vs 66.6 speech only) |

The synthetic-text experiment only checks the fusion code (it is not a TESS result). Section E of the report discusses all of it,
including the negative results.

```bash
python scripts/extract_ssl.py        --tess "<TESS folder>" --cache cache      # wav2vec2 features (cache/speech_w2v.npz, ~400 MB, not committed)
python scripts/make_synthetic_text.py --cache cache                            # synthetic informative text (data/synthetic_text_p060.csv)
python scripts/run_extras.py --cache cache --variants spkn noaug meanpool cnn gru w2v w2v_spkn synth_text synth_fusion_concat synth_fusion_gated synth_fusion_cross synth_fusion_concat_md synth_w2v_fusion_concat
python scripts/run_extras.py --summary                                         # only rebuild the tables from Results/extras/*.json
```
Extras checkpoints (`models/*/runs_extras/`) are not committed.

## Repository layout

```
2b_emotion_recognition/
├── models/
│   ├── speech_pipeline/   train.py  test.py  runs/<protocol>/seed<k>/{best.pt, history.json, test_metrics.json, test_predictions.csv}
│   ├── text_pipeline/     train.py  test.py  runs/...
│   └── fusion_pipeline/   train.py  test.py  runs/...      (--fusion_mode concat | gated | cross)
├── Results/               accuracy_summary.md/.csv, tables/, plots/, representations/, error_analysis_*.md, logs/
├── Report/                Report_2b.pdf (+ build_report.py)
├── common/                data.py (metadata, splits, preprocessing, log-mel), models.py (all blocks), engine.py (training / evaluation), cli.py, plots.py
├── scripts/               prepare_data.py  run_experiments.py  make_tables.py  representations.py  noise_robustness.py  error_analysis.py  acoustic_profile.py  run_extras.py  extract_ssl.py  make_synthetic_text.py
├── data/metadata.csv      speaker, word, emotion, text and file name of every clip (+ synthetic_text_p060.csv for the fusion sanity check)
├── README.md   requirements.txt
```

Only the seed-0 checkpoint of every configuration is committed (the other seeds are recreated by `train.py`; the per-seed result tables are included).

## 1. Install

Python 3.12, tested on Windows 11 with an NVIDIA GTX 1650 (4 GB). A GPU is optional (CPU training of the full grid takes several hours).

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # or the CPU wheel
pip install -r requirements.txt
```

## 2. Get the data and cache the features

```bash
python - <<'EOF'
import kagglehub
print(kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess"))   # prints the local folder, no login needed
EOF
python scripts/prepare_data.py --tess "<printed folder>" --cache cache
# -> data/metadata.csv, cache/speech_logmel.npz (2800 x 300 x 80), cache/text_distilbert.npz (frozen DistilBERT states)
```

## 3. Train and test the three pipelines

```bash
# speech-only / text-only / fusion; the default protocol is "word" and 5 seeds
python models/speech_pipeline/train.py --cache cache        && python models/speech_pipeline/test.py --cache cache
python models/text_pipeline/train.py   --cache cache        && python models/text_pipeline/test.py   --cache cache
python models/fusion_pipeline/train.py --cache cache --fusion_mode gated && python models/fusion_pipeline/test.py --cache cache --fusion_mode gated

# other protocols / seeds
python models/speech_pipeline/train.py --cache cache --protocol spk_OAF2YAF --seeds 0 1 2
python models/speech_pipeline/test.py  --cache cache --protocol spk_OAF2YAF --seeds 0 1 2
```

`test.py` loads the saved checkpoints, writes `Results/tables/<model>_<protocol>_per_seed.csv` and the confusion matrix
`Results/plots/confusion_*.png`. `--init_unimodal` (fusion) starts the encoders from the trained unimodal models.

Everything at once (about 1.5-2 h on a GTX 1650): `python scripts/run_experiments.py --cache cache`
(`--quick` = 1 seed, 8 epochs, smoke test; `--only word,random` selects protocols). It finishes by calling `scripts/make_tables.py`.

## 4. Analysis scripts

```bash
python scripts/representations.py   --cache cache [--protocol word|spk_OAF2YAF|spk_YAF2OAF]   # t-SNE + silhouette / 5-NN of the three blocks
python scripts/noise_robustness.py  --cache cache --tess "<TESS folder>"                       # speech degraded with white noise, 30..0 dB SNR
python scripts/error_analysis.py    --cache cache --tess "<TESS folder>" --protocol spk_OAF2YAF  # failure cases vs the same speaker + word in the true class
python scripts/acoustic_profile.py  --tess "<TESS folder>"                                      # duration / F0 per speaker x emotion
python Report/build_report.py --repo-url <url of this repository>                              # Report/Report_2b.pdf (needs Microsoft Edge)
```

## 5. Training details

AdamW (lr 1e-3, text 2e-3, weight decay 1e-2), cosine schedule with 2 warm-up epochs, label smoothing 0.05, batch 32,
30 epochs (what `run_experiments.py` uses; the `train.py` defaults are 40 speech/fusion, 25 text), early stopping patience 15, the epoch with the best validation accuracy is kept. SpecAugment (frequency + time masks) on the speech input.
Log-mel statistics for standardisation come from the training clips only. Seeds change the model initialisation **and**, for the
word-disjoint protocol, the word split.
