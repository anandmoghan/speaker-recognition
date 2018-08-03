#!/usr/bin/env bash

source /state/partition1/softwares/Miniconda3/etc/profile.d/conda.sh
conda activate tensorflow

chmod 775 -R ./kaldi

python -u run-model.py --stage 3 --batch-size 128 -sc
