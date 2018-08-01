#!/usr/bin/env bash

source ./kaldi/path.sh
source /state/partition1/softwares/Miniconda3/etc/profile.d/conda.sh
conda activate tensorflow
python -u run-model.py --stage 1