#!/usr/bin/env bash

cd $PWD/kaldi
source ./path.sh

cmd=queue.pl
nj=4
compress=true
mfcc_loc="./mfcc"
mfcc_config="./mfcc/mfcc.conf"

if [ $# -lt 2 ] || [ $# -gt 2 ]; then
   echo "Usage: $0 [options] <data-scp> <params-file>";
   echo "e.g.: $0 data.scp mfcc.params"
   exit 1;
fi

source $2
data_scp=$1

log_loc="$mfcc_loc/log"
mkdir -p ${mfcc_loc} || exit 1;
mkdir -p ${log_loc}|| exit 1;

if [ -f ${mfcc_loc}/feats.scp ]; then
  mkdir -p ${mfcc_loc}/.backup
  echo "$0: Moving $mfcc_loc/feats.scp to $mfcc_loc/.backup"
  mv ${mfcc_loc}/feats.scp ${mfcc_loc}/.backup
fi

python split_scp.py --splits ${nj} --file ${data_scp} --prefix wav --dest ${log_loc} || exit 1;

${cmd} JOB=1:${nj} ${log_loc}/mfcc.JOB.log \
    compute-mfcc-feats --verbose=2 --config=${mfcc_config} scp,p:${log_loc}/wav.JOB.scp ark:- \| \
    copy-feats "--write-num-frames=ark,t:$log_loc/utt2num_frames.JOB" --compress=${compress} ark:- \
    ark,scp:${mfcc_loc}/mfcc.JOB.ark,${mfcc_loc}/mfcc.JOB.scp || exit 1;

if [ -f ${log_loc}/.error ]; then
  echo "Error producing mfcc features: "
  tail ${log_loc}/mfcc.1.log
  exit 1;
fi

for n in $(seq ${nj}); do
  cat ${mfcc_loc}/mfcc.${n}.scp || exit 1;
done > ${mfcc_loc}/feats.scp || exit 1