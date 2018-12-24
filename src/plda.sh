#!/usr/bin/env bash

source ./kaldi/path.sh

score_set='dev'
model_tag='HGRU'
lda_dim=150

train_cmd='perl ./kaldi/queue.pl'
save_dir='/home/anandm/workspace/speaker-recognition/save'
data_dir=${save_dir}/data
work_dir=${save_dir}/plda/${model_tag}
sre_trials=${data_dir}/sre_${score_set}_test/trials
mkdir -p ${work_dir}

stage=1


if [ ${stage} -le 0 ]; then
    echo "$0: Computing mean"
    ${train_cmd} ${work_dir}/log/compute_mean.log \
        ivector-mean scp:${data_dir}/sre_unlabelled/embeddings.${model_tag}.scp \
        ${work_dir}/mean.vec || exit 1;

    echo "$0: Training LDA"
    ${train_cmd} ${work_dir}/log/lda.log \
        ivector-compute-lda --total-covariance-factor=0.0 --dim=${lda_dim} \
        "ark:ivector-subtract-global-mean scp:${data_dir}/train_data/embeddings.${model_tag}.scp ark:- |" \
        ark:${data_dir}/train_data/utt2spk ${work_dir}/transform.mat || exit 1;

    echo "$0: Training PLDA"
    ${train_cmd} ${work_dir}/log/plda.log \
        ivector-compute-plda ark:${data_dir}/train_data/spk2utt \
        "ark:ivector-subtract-global-mean scp:${data_dir}/train_data/embeddings.${model_tag}.scp ark:- | transform-vec ${work_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
        ${work_dir}/plda || exit 1;

    echo "$0: Adapting PLDA"
    ${train_cmd} ${work_dir}/log/plda_adapt.log \
        ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
        ${work_dir}/plda \
        "ark:ivector-subtract-global-mean scp:${data_dir}/sre_unlabelled/embeddings.${model_tag}.scp ark:- | transform-vec ${work_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        ${work_dir}/plda_adapt || exit 1;
fi


if [ ${stage} -le 1 ]; then
    echo "$0: Scoring with unadapted PLDA"
    ${train_cmd} ${work_dir}/log/sre_${score_set}_scoring.log \
        ivector-plda-scoring --normalize-length=true --num-utts=ark:${data_dir}/sre_${score_set}_enroll/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 ${work_dir}/plda - |" \
        "ark:ivector-mean ark:${data_dir}/sre_${score_set}_enroll/spk2utt scp:${data_dir}/sre_${score_set}_enroll/embeddings.${model_tag}.scp ark:- | ivector-subtract-global-mean ${work_dir}/mean.vec ark:- ark:- | transform-vec ${work_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean ${work_dir}/mean.vec scp:${data_dir}/sre_${score_set}_test/embeddings.${model_tag}.scp ark:- | transform-vec ${work_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$sre_trials' | cut -d\  --fields=1,2 |" ${work_dir}/sre_${score_set}_scores || exit 1;

    pooled_eer=$(paste ${sre_trials} ${work_dir}/sre_${score_set}_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "Using PLDA, EER: Pooled ${pooled_eer}%"
fi


if [ ${stage} -le 2 ]; then
    echo "$0: Scoring with adapted PLDA"
    ${train_cmd} ${work_dir}/log/sre_${score_set}_scoring_adapt.log \
        ivector-plda-scoring --normalize-length=true --num-utts=ark:${data_dir}/sre_${score_set}_enroll/num_utts.ark \
        "ivector-copy-plda --smoothing=0.0 ${work_dir}/plda_adapt - |" \
        "ark:ivector-mean ark:${data_dir}/sre_${score_set}_enroll/spk2utt scp:${data_dir}/sre_${score_set}_enroll/embeddings.${model_tag}.scp ark:- | ivector-subtract-global-mean ${work_dir}/mean.vec ark:- ark:- | transform-vec ${work_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "ark:ivector-subtract-global-mean ${work_dir}/mean.vec scp:${data_dir}/sre_${score_set}_test/embeddings.${model_tag}.scp ark:- | transform-vec ${work_dir}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
        "cat '$sre_trials' | cut -d\  --fields=1,2 |" ${work_dir}/sre_${score_set}_scores_adapt || exit 1;

    pooled_eer=$(paste ${sre_trials} ${work_dir}/sre_${score_set}_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
    echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%"
fi
