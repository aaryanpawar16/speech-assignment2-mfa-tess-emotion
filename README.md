# Assignment 2: forced alignment (2a) and multimodal emotion recognition (2b)

| Part | Folder | What it contains |
|---|---|---|
| **2a** | [`2a_forced_alignment/`](2a_forced_alignment) | Montreal Forced Aligner on six utterances (3 news reading, 3 non-native minimal-pair sentences): setup, data preparation, custom G2P dictionaries, nine alignment configurations (two acoustic models, fine-tuning, speaker adaptation, pronunciation variants, a model trained from scratch, a G2P model trained here), TextGrids, error analysis, figures, report |
| **2b** | [`2b_emotion_recognition/`](2b_emotion_recognition) | Speech-only, text-only and fusion emotion recognition on TESS: three pipelines, experiments on four evaluation protocols, noise robustness, t-SNE / separability of the Temporal, Contextual and Fusion blocks, failure analysis, extra experiments (ablations, wav2vec2 features, synthetic-text fusion check), report |

The two reports in one file (the form submission): [`Report_Assignment2_2a_2b.pdf`](Report_Assignment2_2a_2b.pdf).

Each folder has its own README with installation, preparation and run commands (copy-paste ready).

## 2a in short

```bash
bash 2a_forced_alignment/setup_mfa.sh            # MFA 3.4.3 via conda-forge (inside WSL on Windows)
bash 2a_forced_alignment/scripts/run_pipeline.sh # prepare, validate, G2P dictionary, 9 alignments, analysis, figures
```

TextGrids: `2a_forced_alignment/outputs/<configuration>/<speaker>/<utterance>.TextGrid`; ready-to-open Praat pairs in `outputs/praat_view/`;
report: `2a_forced_alignment/report/Report_2a.pdf`.

## 2b in short

```bash
cd 2b_emotion_recognition
pip install -r requirements.txt
python scripts/prepare_data.py --tess <TESS folder> --cache cache
python scripts/run_experiments.py --cache cache   # trains and tests every pipeline, writes Results/
```

Result summary (word-disjoint test split; unseen-speaker numbers in the folder's README): speech-only 99.8%, text-only 14.3% (chance, because every TESS
sentence is the same carrier phrase and the text carries no emotion), fusion 99.8-99.9%; across speakers speech-only 40-49% and fusion 44-52%; frozen wav2vec2 features lift the unseen-speaker accuracy to about 67% (extras in the folder's README).
Report: `2b_emotion_recognition/Report/Report_2b.pdf`.
