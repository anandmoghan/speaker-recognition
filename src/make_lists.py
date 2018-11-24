from os.path import abspath, join as join_path

from constants.app_constants import DATA_DIR
from services.common import save_object, make_directory
from services.kaldi import make_kaldi_data_dir
from services.sre_data import get_train_data, make_sre18_dev_data, make_sre18_eval_data


def make_sre_data(data_config, save_loc):
    save_loc = abspath(save_loc)
    data_loc = join_path(save_loc, DATA_DIR)
    make_directory(data_loc)
    train_data = get_train_data(data_config)
    make_kaldi_data_dir(train_data, join_path(data_loc, 'train_data'))
    print('Made {:d} files for training.'.format(train_data.shape[0]))
    sre_dev_enroll, sre_dev_test, sre_unlabelled = make_sre18_dev_data(data_config)
    print('Made {:d} enroll, {:d} test and {:d} unlabeled from sre2018 dev files.'.format(sre_dev_enroll.shape[0], sre_dev_test.shape[0], sre_unlabelled.shape[0]))
    make_kaldi_data_dir(sre_dev_enroll, join_path(data_loc, 'sre_dev_enroll'))
    make_kaldi_data_dir(sre_dev_test, join_path(data_loc, 'sre_dev_test'))
    make_kaldi_data_dir(sre_unlabelled, join_path(data_loc, 'sre_unlabelled'))
    sre_eval_enroll, sre_eval_test = make_sre18_eval_data(data_config)
    make_kaldi_data_dir(sre_eval_enroll, join_path(data_loc, 'sre_eval_enroll'))
    make_kaldi_data_dir(sre_eval_test, join_path(data_loc, 'sre_eval_test'))
    print('Made {:d} enroll and {:d} test from sre2018 eval files.'.format(sre_eval_enroll.shape[0], sre_eval_test.shape[0]))
    print('Saving data lists..')
    save_object(join_path(data_loc, 'train_data.pkl'), train_data)
    save_object(join_path(data_loc, 'sre_unlabelled.pkl'), sre_unlabelled)
    save_object(join_path(data_loc, 'sre_dev.pkl'), (sre_dev_enroll, sre_dev_test))
    save_object(join_path(data_loc, 'sre_eval.pkl'), (sre_eval_enroll, sre_eval_test))
    print('Data lists saved at: {}'.format(data_loc))
    # print('Stage 0: Making trials file..')
    # trials_file = join_path(args.save, TRIALS_FILE)
    # make_sre16_trials_file(DATA_CONFIG, trials_file)
    return train_data, sre_unlabelled, sre_dev_enroll, sre_dev_test, sre_eval_enroll, sre_eval_test


if __name__ == '__main__':
    make_sre_data(data_config='../configs/sre_data.json', save_loc='../save')
