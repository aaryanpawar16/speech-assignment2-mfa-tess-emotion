#!/usr/bin/env bash
# Install the Montreal Forced Aligner (MFA 3.x) and download the models used in this project.
# Works on Linux, macOS and WSL (on Windows MFA must run inside WSL Ubuntu because its Kaldi binaries are conda/Linux only).
#
#   bash setup_mfa.sh
#   conda activate mfa        # (or: micromamba activate mfa)
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  CREATE="conda create -y -n mfa -c conda-forge python=3.11 montreal-forced-aligner"; RUN="conda run -n mfa"
elif command -v micromamba >/dev/null 2>&1; then
  CREATE="micromamba create -y -n mfa -c conda-forge python=3.11 montreal-forced-aligner"; RUN="micromamba run -n mfa"
else
  echo "No conda/micromamba found: installing micromamba into ~/mm"
  mkdir -p ~/mm/bin && curl -L -o ~/mm/bin/micromamba \
      https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 && chmod +x ~/mm/bin/micromamba
  export MAMBA_ROOT_PREFIX=~/mm/root
  CREATE="$HOME/mm/bin/micromamba create -y -n mfa -c conda-forge python=3.11 montreal-forced-aligner"; RUN="$HOME/mm/bin/micromamba run -n mfa"
fi

$CREATE
$RUN mfa version

# models used by scripts/run_pipeline.sh
$RUN mfa model download acoustic   english_us_arpa     # ARPAbet acoustic model (main)
$RUN mfa model download dictionary english_us_arpa
$RUN mfa model download g2p        english_us_arpa     # used to build the custom dictionary
$RUN mfa model download acoustic   english_mfa         # second acoustic model (IPA phone set), for comparison
$RUN mfa model download dictionary english_us_mfa
$RUN mfa model download g2p        english_us_mfa
echo "MFA ready."
