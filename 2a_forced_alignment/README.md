# Forced alignment with the Montreal Forced Aligner (MFA)

Assignment 2a: align six utterances (about 85 s of news reading and three short non-native minimal-pair
sentences) with their transcripts at word and phone level using MFA 3.x, inspect the TextGrids, and analyse the
errors. Everything is automated by one script.

| | |
|---|---|
| **Tool** | Montreal Forced Aligner 3.4.3 (Kaldi, HMM-GMM) |
| **Acoustic models** | `english_us_arpa` (main), `english_mfa` (second model), `english_us_arpa` adapted to these speakers, and one trained from scratch on the six clips (plus a **G2P model trained here** for the dictionary) |
| **Dictionaries** | stock `english_us_arpa` / `english_us_mfa`, and **custom dictionaries** = stock + G2P pronunciations for the out-of-vocabulary words |
| **Outputs** | TextGrid files for 9 configurations x 6 utterances (`outputs/`), CSV analysis (`outputs/analysis/`), figures (`outputs/figures/`) |
| **Report** | `report/Report_2a.pdf` (also sent by e-mail, as the assignment requires) |

## Repository layout

```
2a_forced_alignment/
├── data/                      the provided dataset: wav/ and transcripts/ (unchanged)
├── scripts/
│   ├── prepare_data.py        -> MFA corpus layout (speaker folders, .wav + .lab), two transcript variants
│   ├── run_pipeline.sh        -> validate, G2P dictionary, 9 alignments, adaptation, training from scratch, analysis, figures (the whole pipeline)
│   ├── analyze_alignments.py  -> statistics, system agreement, OOV handling, squeezed phones, pauses vs punctuation
│   ├── make_variant_dictionary.py -> dictionary with confusable vowel variants for the minimal-pair words (A7)
│   ├── extras_analysis.py     -> A7 vowel choice, boundary check against an energy-based reference, manual-check template/evaluation
│   ├── make_g2p_training_set.py -> 25k-word sample of the stock dictionary to train a G2P model (A9) + 500 held-out words
│   ├── g2p_eval.py            -> word / phone error rate of the trained G2P model; its OOV pronunciations vs the pretrained ones
│   ├── manual_check_figures.py -> zoomed spectrogram per boundary of the manual-check template (outputs/manual_check/)
│   ├── isle_vowels.py         -> F1/F2 of the minimal-pair vowels (Praat via parselmouth)
│   └── visualize.py           -> Praat-style figures + outputs/praat_view/
├── dictionaries/              oov_words_<dict>.txt (OOV lists), oov_<dict>.dict (G2P output), english_us_arpa_custom.dict, english_us_mfa_custom.dict
├── models/                    english_us_arpa_adapted.zip: acoustic model adapted with `mfa adapt` (92 MB, created by the pipeline, not committed)
├── outputs/
│   ├── A1_naive_stock_arpa/ ... A9_norm_trained_g2p/    TextGrids  <config>/<speaker>/<utterance>.TextGrid
│   ├── analysis/              CSV tables + summary.txt
│   ├── figures/               alignment figures
│   └── praat_view/            wav + TextGrid pairs of the main configuration, ready to open in Praat
├── report/                    Report_2a.pdf
├── setup_mfa.sh   requirements.txt   README.md
```

## 1. Install MFA

MFA (its Kaldi binaries) is distributed through conda-forge, so on **Windows it has to run inside WSL (Ubuntu)**;
on Linux/macOS it runs natively.

```bash
bash setup_mfa.sh          # installs micromamba/conda, creates the env "mfa", downloads the models used below
conda activate mfa         # or:  micromamba activate mfa
mfa version                # 3.x
pip install -r requirements.txt     # analysis / plotting packages (any Python, also on Windows)
```

Manual equivalent:

```bash
conda create -n mfa -c conda-forge python=3.11 montreal-forced-aligner
conda activate mfa
mfa model download acoustic   english_us_arpa ; mfa model download dictionary english_us_arpa ; mfa model download g2p english_us_arpa
mfa model download acoustic   english_mfa     ; mfa model download dictionary english_us_mfa  ; mfa model download g2p english_us_mfa
```

## 2. Prepare the data

MFA expects, for every utterance, an audio file and a transcript with the same name, and treats each folder as one
speaker. The provided files are 16 kHz mono PCM wavs (no conversion needed).

```bash
python scripts/prepare_data.py --data data --out .
# corpus_naive/ : transcripts exactly as provided (whitespace collapsed)
# corpus_norm/  : "spoken form": S.J.C.'s -> S J C's, WBUR's -> W B U R's, "seventy- six" -> "seventy six"
#   corpus_*/F2B/F2BJRLP1.wav + .lab ...   corpus_*/ISLE_SESS0131/ISLE_SESS0131_BLOCKD02_01_sprt1.wav + .lab ...
```

## 3. Run the alignment

The single command below runs every step (it skips the steps whose output already exists; `FORCE=1` redoes them):

```bash
bash scripts/run_pipeline.sh
```

What it does, with the underlying MFA commands (this is what you would type by hand):

```bash
# a) find the out-of-vocabulary words
mfa validate corpus_norm english_us_arpa --ignore_acoustics --output_directory outputs/validation/norm
# b) pronunciations for them with the G2P model (3 candidates each; the aligner chooses the best one)
mfa g2p dictionaries/oov_words_english_us_arpa.txt english_us_arpa dictionaries/oov_english_us_arpa.dict -n 3
cat <stock english_us_arpa.dict> dictionaries/oov_english_us_arpa.dict > dictionaries/english_us_arpa_custom.dict
# c) align
mfa align corpus_norm dictionaries/english_us_arpa_custom.dict english_us_arpa outputs/A3_norm_custom_arpa --clean
# d) other configurations: --fine_tune, an adapted model (mfa adapt), a second acoustic model (english_mfa)
mfa adapt corpus_norm dictionaries/english_us_arpa_custom.dict english_us_arpa models/english_us_arpa_adapted.zip
mfa align corpus_norm dictionaries/english_us_mfa_custom.dict english_mfa outputs/A6_norm_custom_mfa --clean
# e) pronunciation variants (A7), an acoustic model trained on the six clips only (A8), a G2P model trained here (A9)
python scripts/make_variant_dictionary.py
mfa align corpus_norm dictionaries/english_us_arpa_variants.dict english_us_arpa outputs/A7_norm_variants_arpa --clean
mfa train corpus_norm dictionaries/english_us_arpa_custom.dict models/trained_on_six_clips.zip --output_directory outputs/A8_trained_on_data_only --beam 100 --retry_beam 400 --single_speaker --clean
python scripts/make_g2p_training_set.py --dict <stock english_us_arpa.dict> --n 25000 --out dictionaries/g2p_train_subset.dict
mfa train_g2p dictionaries/g2p_train_subset.dict models/g2p_trained_subset.zip --clean -j 3 --overwrite      # ~10 min
mfa g2p dictionaries/oov_words_english_us_arpa.txt models/g2p_trained_subset.zip dictionaries/oov_trained_g2p.dict -n 3
cat <stock english_us_arpa.dict> dictionaries/oov_trained_g2p.dict > dictionaries/english_us_arpa_trainedg2p.dict
mfa align corpus_norm dictionaries/english_us_arpa_trainedg2p.dict english_us_arpa outputs/A9_norm_trained_g2p --clean
```

On Windows, start the pipeline from PowerShell/Git Bash with
`wsl -d Ubuntu -- bash /mnt/d/.../2a_forced_alignment/scripts/run_pipeline.sh`
(set `MFA="micromamba run -n mfa mfa"` if `mfa` is not on the PATH inside WSL). The analysis step needs the packages
of `requirements.txt` in the Python that runs it; if they are missing the script prints the two commands to run:

```bash
python scripts/analyze_alignments.py --root .     # -> outputs/analysis/*.csv, summary.txt
python scripts/isle_vowels.py --root .            # -> outputs/analysis/isle_vowel_formants.csv, figures/isle_vowel_chart.png
python scripts/visualize.py --root .              # -> outputs/figures/*.png, outputs/praat_view/
python scripts/extras_analysis.py --root .        # -> variant_choice_A7.csv, pause_boundary_vs_energy*.csv, manual_check_template.csv
python scripts/g2p_eval.py --root .               # -> g2p_trained_vs_pretrained.csv (held-out WER/PER of the trained G2P, OOV pronunciations)
python scripts/manual_check_figures.py --root .   # -> outputs/manual_check/*.png (one zoomed figure per boundary to check by hand)
```

### Manual check of 12 word starts

`outputs/analysis/manual_check_template.csv` lists 12 word starts (the ones where the two acoustic models disagree most, plus a spread of agreeing ones).
`python scripts/manual_check_figures.py --root . [--blind]` draws one zoomed waveform + spectrogram per boundary (`outputs/manual_check/` with the A3/A6 times,
`outputs/manual_check_blind/` without them, for unbiased reading). The reference in `outputs/analysis/manual_check_filled.csv` has a `source` per row: 9 starts were read **visually from the blind figures by the AI assistant that wrote this report (no audio, about +-10-30 ms)**,
1 start (*thirty*, 16.30 s) was set **by ear in Praat by the author**, and for 2 starts (*to*, *bad*) the author confirmed by ear in Praat that A3's boundary is correct
(this favours A3 by construction, so `manual_check_errors_independent10.csv` scores only the 10 other starts; `manual_check_errors_all12.csv` scores all 12). To use your own corrections, fill the `praat_time_s` column (open the wav + TextGrid from `outputs/praat_view/` in Praat),
save as a new CSV and run `python scripts/extras_analysis.py --root . --manual <csv>`: it prints the mean/maximum error of every configuration
and writes `outputs/analysis/manual_check_errors.csv`.

### Configurations

| Name | Transcripts | Dictionary | Acoustic model | Purpose |
|---|---|---|---|---|
| A1 | as provided | stock `english_us_arpa` | `english_us_arpa` | baseline |
| A2 | spoken form | stock | `english_us_arpa` | effect of text normalisation |
| **A3** | spoken form | **custom** (stock + G2P) | `english_us_arpa` | **main result** |
| A4 | spoken form | custom | `english_us_arpa` + `--fine_tune` | MFA's extra boundary fine-tuning |
| A5 | spoken form | custom | `english_us_arpa` adapted to the data (`mfa adapt`) | speaker/corpus adaptation |
| A6 | spoken form | custom (`english_us_mfa`) | `english_mfa` | a different pre-trained model (IPA phone set) |
| A7 | spoken form | custom + **pronunciation variants** of the minimal-pair words (`make_variant_dictionary.py`) | `english_us_arpa` | which vowel does the aligner pick when both are allowed? |
| A8 | spoken form | custom | **trained from scratch on the six clips** (`mfa train`, 97 s) | shows why pre-trained models are needed (degenerate result) |
| A9 | spoken form | stock + OOV pronunciations from a **G2P model trained here** (`mfa train_g2p`, 25k-word sample of the stock dictionary) | `english_us_arpa` | does a small trained G2P change the alignment? (held-out WER 50.8%, PER 12.8%; word boundaries identical to A3 within 10 ms) |

## 4. Inspecting the output in Praat

`outputs/praat_view/` holds the audio and the main-configuration TextGrid with the same base name. In Praat:
**Open > Read from file...** select a `.wav` and the `.TextGrid`, then **View & Edit**. The `words` tier holds the word
intervals, the `phones` tier the ARPAbet phones (digits are stress marks); empty intervals are silences.
`outputs/figures/` shows the same information drawn with matplotlib (waveform, spectrogram, both tiers).

## 5. Results and observations

See `report/Report_2a.pdf` for the discussion; the numbers are in `outputs/analysis/summary.txt` and the CSV files.
