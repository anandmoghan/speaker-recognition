#!/usr/bin/env bash

cd $PWD/kaldi
source ./path.sh

cmd=queue.pl
nj=4
vad_loc="./vad"
vad_config="./vad/vad.conf"

if [ $# -lt 2 ] || [ $# -gt 2 ]; then
   echo "Usage: $0 [options] <feats-scp> <params-file>";
   echo "e.g.: $0 feats.scp vad.params"
   exit 1;
fi

source $2
feats_scp=$1

log_loc="$vad_loc/log"
mkdir -p ${vad_loc} || exit 1;
mkdir -p ${log_loc}|| exit 1;

python split_scp.py --splits ${nj} --file ${feats_scp} --prefix feats --dest ${log_loc} || exit 1;

${cmd} JOB=1:${nj} ${log_loc}/vad.JOB.log \
  compute-vad --config=${vad_config} scp:${log_loc}/feats.JOB.scp \
  ark,scp:${vad_loc}/vad.JOB.ark,${vad_loc}/vad.JOB.scp || exit 1;

for ((n=1; n<=nj; n++)); do
  cat ${vad_loc}/vad.${n}.scp || exit 1;
done > ${vad_loc}/vad.scp