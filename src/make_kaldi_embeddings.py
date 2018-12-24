from os.path import join as join_path, abspath

from constants.app_constants import DATA_DIR, EMB_DIR
from services.common import run_parallel, load_array
from services.kaldi import write_vector, write_num_utterance


def write_embeddings(args):
    utt, loc, ark_file = args[0], args[1], args[2]
    emb = load_array(loc)
    return write_vector(emb, utt, ark_file)


def convert_embeddings(split, model_tag, save_loc, n_workers=5, p_bar=False):
    data_loc = join_path(save_loc, '{}/{}'.format(DATA_DIR, split))
    embedding_loc = join_path(save_loc, '{}/{}'.format(EMB_DIR, model_tag))
    utt_to_spk = join_path(data_loc, 'utt2spk')

    args_list = []
    with open(utt_to_spk) as f:
        for line in f.readlines():
            utt = line.strip().split()[0]
            loc = join_path(embedding_loc, '{}.npy'.format(utt))
            ark_file = join_path(embedding_loc, '{}.ark'.format(utt))
            args_list.append((utt, loc, ark_file))

    scp_list = run_parallel(write_embeddings, args_list, n_workers, p_bar=p_bar)

    with open(join_path(data_loc, 'embeddings.{}.scp'.format(model_tag)), 'w') as f:
        for scp in scp_list:
            f.write(scp)


def make_num_utterances(split, save_loc):
    data_loc = join_path(save_loc, '{}/{}'.format(DATA_DIR, split))
    utt_to_spk = join_path(data_loc, 'utt2spk')
    num_utterances_file = join_path(data_loc, 'num_utts.ark')

    speakers = []
    with open(utt_to_spk) as f:
        for line in f.readlines():
            spk = line.strip().split()[1]
            speakers.append(spk)

    write_num_utterance(speakers, num_utterances_file)


if __name__ == '__main__':
    for split_ in ['train_data', 'sre_unlabelled', 'sre_dev_enroll', 'sre_dev_test']:
        print('Converting to kaldi embeddings - {}'.format(split_))
        convert_embeddings(split_, 'HGRU3', abspath('../save'), n_workers=32, p_bar=True)

    make_num_utterances('sre_dev_enroll', abspath('../save'))
