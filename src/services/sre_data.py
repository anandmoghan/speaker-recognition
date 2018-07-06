from json import loads as load_json
from os.path import join as join_path

import numpy as np
import re

from services.common import get_file_list, get_file_list_as_dict, sort_by_index


def get_file_name(x):
    return x.split('/')[-1].split('.sph')[0]


def get_sre_swbd_data(sre_config):
    with open(sre_config, 'r') as f:
        sre_data = load_json(f.read())
    data_root = sre_data['ROOT']
    data_loc = sre_data['LOCATION']
    speaker_key = sre_data['SPEAKER_KEY']

    sre04 = make_old_sre_data(data_root, data_loc['SRE04'], 2004, speaker_key)
    sre05_train = make_old_sre_data(data_root, data_loc['SRE05_TRAIN'], 2005, speaker_key)
    sre05_test = make_old_sre_data(data_root, data_loc['SRE05_TEST'], 2005, speaker_key)
    sre06 = make_old_sre_data(data_root, data_loc['SRE06'], 2006, speaker_key)
    sre08 = make_sre08_data(data_root, data_loc['SRE08_TRAIN'], data_loc['SRE08_TEST'])
    sre10 = make_sre10_data(data_root, data_loc['SRE10'])
    swbd_c1 = make_swbd_cellular(data_root, data_loc['SWBD_C1'], 1)
    swbd_c2 = make_swbd_cellular(data_root, data_loc['SWBD_C2'], 2)
    print('Sorting sre_swbd data...')
    return sort_by_index(np.hstack([sre04, sre05_train, sre05_test, sre06, sre08, sre10, swbd_c1, swbd_c2]).T)


def make_old_sre_data(data_root, data_loc, sre_year, speaker_key):
    print('Making sre{} lists...'.format(sre_year))
    sre_loc = join_path(data_root, data_loc)
    sre_year = 'sre' + str(sre_year)
    bad_audio = ['jagi', 'jaly', 'jbrg', 'jcli', 'jfmx']
    file_list = get_file_list_as_dict(sre_loc)

    for ba in bad_audio:
        try:
            del file_list[ba]
        except KeyError:
            pass

    index_list = []
    location_list = []
    speaker_list = []
    channel_list = []
    with open(speaker_key, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            speaker_id = tokens[0]
            file_name = tokens[3]
            channel = tokens[4]
            if sre_year == tokens[2]:
                try:
                    file_loc = file_list[file_name]
                    index_list.append(sre_year + '_' + file_name)
                    location_list.append(file_loc)
                    speaker_list.append(sre_year + '_' + speaker_id)
                    channel_list.append(1 if channel == 'A' else 2)
                    del file_list[file_name]
                except KeyError:
                    pass

    return np.vstack([index_list, location_list, channel_list, speaker_list])


def make_sre08_data(data_root, data_train_loc, data_test_loc):
    print('Making sre2008 lists...')
    train_loc = join_path(data_root, data_train_loc)
    test_loc = join_path(data_root, data_test_loc)

    model_key = join_path(test_loc, 'data/keys/NIST_SRE08_KEYS.v0.1/model-keys/NIST_SRE08_short2.model.key')
    trials_key = join_path(test_loc, 'data/keys/NIST_SRE08_KEYS.v0.1/trial-keys/NIST_SRE08_short2-short3.trial.key')

    train_file_list = get_file_list_as_dict(train_loc)
    test_file_list = get_file_list_as_dict(test_loc)
    file_list = {**train_file_list, **test_file_list}

    index_list = []
    location_list = []
    speaker_list = []
    channel_list = []
    model_to_speaker = dict()
    with open(model_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[,:]+', line.strip())
            model_id = tokens[0]
            file_name = tokens[2]
            channel = tokens[3]
            speaker_id = tokens[4]
            model_to_speaker[model_id] = speaker_id
            try:
                file_loc = file_list[file_name]
                index_list.append('sre2008_' + file_name)
                location_list.append(file_loc)
                channel_list.append(1 if channel == 'a' else 2)
                speaker_list.append('sre2008_' + speaker_id)
                del file_list[file_name]
            except KeyError:
                pass

    with open(trials_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[,]+', line.strip())
            model_id = tokens[0]
            file_name = tokens[1]
            channel = tokens[2]
            target_type = tokens[3]
            try:
                file_loc = file_list[file_name]
                speaker_id = model_to_speaker[model_id]
                if target_type == 'target':
                    index_list.append('sre2008_' + file_name)
                    location_list.append(file_loc)
                    channel_list.append(1 if channel == 'a' else 2)
                    speaker_list.append('sre2008_' + speaker_id)
                    del file_list[file_name]
            except KeyError:
                pass

    return np.vstack([index_list, location_list, channel_list, speaker_list])


def make_sre10_data(data_root, data_loc):
    print('Making sre2010 lists...')
    sre_loc = join_path(data_root, data_loc)

    model_key = join_path(sre_loc, 'keys/coreext.modelkey.csv')
    train_key = join_path(sre_loc, 'train/coreext.trn')
    trials_key = join_path(sre_loc, 'keys/coreext-coreext.trialkey.csv')

    file_list = get_file_list_as_dict(join_path(sre_loc, 'data'))

    index_list = []
    location_list = []
    speaker_list = []
    channel_list = []
    model_to_speaker = dict()
    with open(model_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[,]+', line.strip())
            model_id = tokens[0]
            speaker_id = tokens[1]
            if not speaker_id == 'NOT_SCORED':
                model_to_speaker[model_id] = speaker_id

    with open(train_key, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s:]+', line.strip())
            model_id = tokens[0]
            file_name = tokens[2].split('/')[2].split('.sph')[0]
            channel = tokens[3]
            try:
                file_loc = file_list[file_name]
                speaker_id = model_to_speaker[model_id]
                index_list.append('sre2010_' + file_name)
                location_list.append(file_loc)
                speaker_list.append('sre2010_' + speaker_id)
                channel_list.append(1 if channel == 'A' else 2)
                del file_list[file_name]
            except KeyError:
                pass

    with open(trials_key, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[,]+', line.strip())
            model_id = tokens[0]
            file_name = tokens[1]
            channel = tokens[2]
            target_type = tokens[3]
            try:
                speaker_id = model_to_speaker[model_id]
                file_loc = file_list[file_name]
                if target_type == 'target':
                    index_list.append('sre2010_' + file_name)
                    location_list.append(file_loc)
                    speaker_list.append('sre2010_' + speaker_id)
                    channel_list.append(1 if channel == 'A' else 2)
                    del file_list[file_name]
            except KeyError:
                pass

    return np.vstack([index_list, location_list, channel_list, speaker_list])


def make_swbd_cellular(data_root, data_loc, swbd_type=1):
    print('Making swbd cellular {} lists...'.format(swbd_type))
    swbd_loc = join_path(data_root, data_loc)

    bad_audio = [40019, 45024, 40022]
    stats_key = join_path(swbd_loc, 'doc{}/swb_callstats.tbl'.format('' if swbd_type == 1 else 's'))
    swbd_type = 'swbd_c{:d}_'.format(swbd_type)

    file_list = get_file_list_as_dict(swbd_loc)

    for ba in bad_audio:
        try:
            del file_list['sw_' + str(ba)]
        except KeyError:
            pass

    index_list = []
    location_list = []
    channel_list = []
    speaker_list = []
    with open(stats_key, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[,]+', line.strip())
            file_name = tokens[0]
            speaker_id1 = tokens[1]
            speaker_id2 = tokens[2]
            try:
                file_loc = file_list['sw_' + str(file_name)]
                index_list.append(swbd_type + file_name + '_ch1')
                location_list.append(file_loc)
                channel_list.append(1)
                speaker_list.append('sw_' + speaker_id1)
                index_list.append(swbd_type + file_name + '_ch2')
                location_list.append(file_loc)
                channel_list.append(2)
                speaker_list.append('sw_' + speaker_id2)
                del file_list['sw_' + str(file_name)]
            except KeyError:
                pass

    return np.vstack([index_list, location_list, channel_list, speaker_list])


def make_sre16_eval_data(sre_config):
    print('Making sre2016 eval lists...')
    with open(sre_config, 'r') as f:
        sre_data = load_json(f.read())
    data_root = sre_data['ROOT']
    data_loc = sre_data['LOCATION']['SRE16_EVAL']
    sre_loc = join_path(data_root, data_loc)

    file_list = get_file_list_as_dict(join_path(sre_loc, 'data/enrollment'))

    meta_key = join_path(sre_loc, 'docs/sre16_eval_enrollment.tsv')

    index_list = []
    location_list = []
    speaker_list = []
    channel_list = []
    with open(meta_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[\s]+', line.strip())
            speaker_id = tokens[0]
            file_name = tokens[1]
            try:
                file_loc = file_list[file_name]
                index_list.append('sre16_eval_enroll_' + file_name)
                location_list.append(file_loc)
                speaker_list.append('sre16_eval_enroll_' + speaker_id)
                channel_list.append(1)
                del file_list[file_name]
            except KeyError:
                pass

    enrollment_data = np.vstack([index_list, location_list, channel_list, speaker_list]).T

    file_list = get_file_list_as_dict(join_path(sre_loc, 'data/test'))

    segment_key = join_path(sre_loc, 'docs/sre16_eval_segment_key.tsv')
    language_key = join_path(sre_loc, 'metadata/calls.tsv')
    trial_key = join_path(sre_loc, 'docs/sre16_eval_trial_key.tsv')

    utt_to_call = dict()
    with open(segment_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[\s]+', line.strip())
            utt_to_call[tokens[0]] = tokens[1]

    call_to_language = dict()
    with open(language_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[\s]+', line.strip())
            call_to_language[tokens[0]] = tokens[1]

    index_list = []
    location_list = []
    speaker_list = []
    channel_list = []
    language_list = []
    target_list = []
    with open(trial_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[\s]+', line.strip())
            speaker_id = tokens[0]
            file_name = tokens[1]
            target_type = tokens[3]
            call_id = utt_to_call[file_name]
            try:
                file_loc = file_list[file_name]
                index_list.append('sre16_eval_test_' + file_name)
                location_list.append(file_loc)
                speaker_list.append('sre16_eval_test_' + speaker_id)
                channel_list.append(1)
                language_list.append(call_to_language[call_id])
                target_list.append(target_type)
                del file_list[file_name]
            except KeyError:
                pass

    test_data = np.vstack([index_list, location_list, channel_list, speaker_list, target_list, language_list]).T

    return enrollment_data, test_data


def make_sre16_unlabeled_data(sre_config):
    with open(sre_config, 'r') as f:
        sre_data = load_json(f.read())
    data_root = sre_data['ROOT']
    data_loc = sre_data['LOCATION']['SRE16_UNLABELED']
    sre_loc = join_path(data_root, data_loc)

    file_list = get_file_list(join_path(sre_loc, 'data/unlabeled/major'))
    index_list = []
    location_list = []
    speaker_list = []
    channel_list = []

    return np.vstack([index_list, location_list, channel_list, speaker_list])
