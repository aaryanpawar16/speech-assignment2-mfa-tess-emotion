#!/usr/bin/env bash
# Full forced-alignment pipeline with the Montreal Forced Aligner (MFA 3.x).
#
#   bash scripts/run_pipeline.sh            # everything (skips stages whose output already exists)
#   FORCE=1 bash scripts/run_pipeline.sh    # redo everything
#
# Needs: the `mfa` conda environment (see setup_mfa.sh) and the corpora written by
#        `python scripts/prepare_data.py`.  Run it on Linux/macOS, or inside WSL on Windows.
# To use a non-default MFA launcher:  MFA="micromamba run -n mfa mfa" bash scripts/run_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MFA="${MFA:-mfa}"
JOBS="${JOBS:-2}"
FORCE="${FORCE:-0}"
OUT="$ROOT/outputs"
DICT_DIR="$ROOT/dictionaries"
MFA_HOME="${MFA_ROOT_DIR:-$HOME/Documents/MFA}"
PRE="$MFA_HOME/pretrained_models"
mkdir -p "$OUT" "$DICT_DIR" "$ROOT/models"

step() { echo; echo "==================== $* ===================="; }
need() { [ "$FORCE" = "1" ] || [ ! -e "$1" ]; }          # true when the output $1 must be (re)built

# ---------------------------------------------------------------------------------------------
step "0. Prepare corpora (audio + .lab per speaker folder)"
python "$ROOT/scripts/prepare_data.py" --data "$ROOT/data" --out "$ROOT" 2>/dev/null \
  || python3 "$ROOT/scripts/prepare_data.py" --data "$ROOT/data" --out "$ROOT"

# ---------------------------------------------------------------------------------------------
step "1. Validate the corpus against the stock dictionary (finds out-of-vocabulary words)"
for v in naive norm; do
  if need "$OUT/validation/$v/oovs_found_english_us_arpa.txt"; then
    $MFA validate "$ROOT/corpus_$v" english_us_arpa --ignore_acoustics --clean --quiet -j "$JOBS" \
         --output_directory "$OUT/validation/$v" || true     # validate exits non-zero when OOVs exist
  fi
  echo "[$v] OOV words:" $(cat "$OUT/validation/$v/oovs_found_english_us_arpa.txt" 2>/dev/null | tr '\n' ' ')
done
# every dictionary has its own gaps (english_us_mfa lacks e.g. "judgeships" and "r's"): validate against it as well
if need "$OUT/validation/norm_mfa/oovs_found_english_us_mfa.txt"; then
  $MFA validate "$ROOT/corpus_norm" english_us_mfa --ignore_acoustics --clean --quiet -j "$JOBS" \
       --output_directory "$OUT/validation/norm_mfa" || true
fi
echo "[norm, english_us_mfa dictionary] OOV words:" $(cat "$OUT/validation/norm_mfa/oovs_found_english_us_mfa.txt" 2>/dev/null | tr '\n' ' ')

# ---------------------------------------------------------------------------------------------
step "2. Custom pronunciation dictionaries = stock dictionary + G2P pronunciations for the OOV words"
for pair in "english_us_arpa:english_us_arpa:validation/norm" "english_us_mfa:english_us_mfa:validation/norm_mfa"; do
  dict="${pair%%:*}"; rest="${pair#*:}"; g2p="${rest%%:*}"; vdir="${rest##*:}"
  cp "$OUT/$vdir/oovs_found_$dict.txt" "$DICT_DIR/oov_words_$dict.txt"
  if need "$DICT_DIR/${dict}_custom.dict"; then
    $MFA model download g2p "$g2p" >/dev/null 2>&1 || true
    # three candidate pronunciations per OOV word; the aligner picks the best one during alignment
    $MFA g2p "$DICT_DIR/oov_words_$dict.txt" "$g2p" "$DICT_DIR/oov_${dict}.dict" -n 3 --clean --quiet -j 1
    cat "$PRE/dictionary/$dict.dict" "$DICT_DIR/oov_${dict}.dict" > "$DICT_DIR/${dict}_custom.dict"
  fi
  echo "[$dict] custom dictionary:" $(wc -l < "$DICT_DIR/${dict}_custom.dict") "entries,"  $(wc -l < "$DICT_DIR/oov_${dict}.dict") "from G2P"
done

# ---------------------------------------------------------------------------------------------
step "3. Align with several configurations"
align() {   # name corpus dict acoustic [extra mfa options...]
  local name="$1" corpus="$2" dict="$3" am="$4"; shift 4
  if need "$OUT/$name/alignment_analysis.csv"; then
    echo "--- $name"
    $MFA align "$ROOT/$corpus" "$dict" "$am" "$OUT/$name" --clean --quiet -j "$JOBS" \
         --output_format long_textgrid --overwrite "$@"
  fi
  echo "[$name] TextGrids:" $(find "$OUT/$name" -name '*.TextGrid' | wc -l)
}
# A1  as-provided transcripts, stock dictionary            (baseline)
align A1_naive_stock_arpa      corpus_naive english_us_arpa                        english_us_arpa
# A2  spoken-form transcripts, stock dictionary            (effect of text normalisation)
align A2_norm_stock_arpa       corpus_norm  english_us_arpa                        english_us_arpa
# A3  spoken-form transcripts + custom dictionary          (main result)
align A3_norm_custom_arpa      corpus_norm  "$DICT_DIR/english_us_arpa_custom.dict" english_us_arpa
# A4  A3 + MFA's extra fine-tuning of the boundaries
align A4_norm_custom_arpa_ft   corpus_norm  "$DICT_DIR/english_us_arpa_custom.dict" english_us_arpa --fine_tune
# A5  acoustic model adapted to these two speakers, then A3 repeated with it
if need "$ROOT/models/english_us_arpa_adapted.zip"; then
  $MFA adapt "$ROOT/corpus_norm" "$DICT_DIR/english_us_arpa_custom.dict" english_us_arpa \
       "$ROOT/models/english_us_arpa_adapted.zip" --clean --quiet -j "$JOBS" --output_directory "$OUT/_adapt_tmp"
fi
align A5_norm_custom_adapted   corpus_norm  "$DICT_DIR/english_us_arpa_custom.dict" "$ROOT/models/english_us_arpa_adapted.zip"
# A6  a different pre-trained acoustic model (english_mfa, IPA phone set) with its own dictionary
align A6_norm_custom_mfa       corpus_norm  "$DICT_DIR/english_us_mfa_custom.dict"  english_mfa

# A7  custom dictionary + pronunciation VARIANTS for the minimal-pair words (the aligner picks the vowel that fits the audio)
{ python "$ROOT/scripts/make_variant_dictionary.py" --root "$ROOT" 2>/dev/null || python3 "$ROOT/scripts/make_variant_dictionary.py" --root "$ROOT"; }
align A7_norm_variants_arpa    corpus_norm  "$DICT_DIR/english_us_arpa_variants.dict" english_us_arpa
# A8  an acoustic model trained FROM SCRATCH on the six clips only (97 s of audio): expected to be much worse than the pre-trained models
if need "$ROOT/models/trained_on_six_clips.zip"; then
  echo "--- A8_trained_on_data_only"
  $MFA train "$ROOT/corpus_norm" "$DICT_DIR/english_us_arpa_custom.dict" "$ROOT/models/trained_on_six_clips.zip" \
       --output_directory "$OUT/A8_trained_on_data_only" --clean --quiet -j "$JOBS" --output_format long_textgrid \
       --beam 100 --retry_beam 400 --single_speaker       # the default beam finds no alignment with so little data
fi
echo "[A8_trained_on_data_only] TextGrids:" $(find "$OUT/A8_trained_on_data_only" -name '*.TextGrid' | wc -l)

# ---------------------------------------------------------------------------------------------
step "4. Analyse the alignments and draw figures (plain Python, no MFA needed)"
PY=python; command -v python >/dev/null 2>&1 || PY=python3
if $PY -c "import numpy, scipy, matplotlib, soundfile, tgt" 2>/dev/null; then
  $PY "$ROOT/scripts/analyze_alignments.py" --root "$ROOT"
  $PY "$ROOT/scripts/extras_analysis.py" --root "$ROOT"
  $PY "$ROOT/scripts/visualize.py" --root "$ROOT"
else
  echo "numpy/scipy/matplotlib/soundfile/tgt not available in this environment: run"
  echo "  pip install -r requirements.txt && python scripts/analyze_alignments.py && python scripts/visualize.py"
fi
echo; echo "Done. TextGrids: $OUT/<configuration>/<speaker>/<utterance>.TextGrid ; analysis: $OUT/analysis ; figures: $OUT/figures"
