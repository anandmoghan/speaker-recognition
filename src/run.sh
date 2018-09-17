#!/usr/bin/env bash

source /state/partition1/softwares/Miniconda3/etc/profile.d/conda.sh
conda activate tensorflow

python -u run-e2e.py --stage 3 -sc --model-tag ATTN_3 --gpu 0
