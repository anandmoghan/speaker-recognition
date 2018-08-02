from json import loads as load_json
from os.path import join as join_path

import numpy as np
import re

from services.common import get_file_list_as_dict, remove_duplicates, sort_by_index


def get_train_data(data_config):
    with open(data_config, 'r') as f:
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
    swbd_p1 = make_swbd_phase(data_root, data_loc['SWBD_P1'], 1)
    swbd_p2 = make_swbd_phase(data_root, data_loc['SWBD_P2'], 2)
    swbd_p3 = make_swbd_phase(data_root, data_loc['SWBD_P3'], 3)
    mx6_calls = make_mixer6_calls(data_root, data_loc['MX6'])
    mx6_mic = make_mixer6_mic(data_root, data_loc['MX6'])
    train_data = np.hstack([sre04, sre05_train, sre05_test, sre06, sre08, sre10, swbd_c1, swbd_c2, swbd_p1,
                            swbd_p2, swbd_p3, mx6_calls, mx6_mic]).T
    print('Removing Duplicates...')
    train_data, n_dup = remove_duplicates(train_data)
    print('Removed {} duplicates.'.format(n_dup))
    print('Sorting train data by index...')
    return sort_by_index(train_data)


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
    read_list = []
    with open(speaker_key, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            speaker_id = tokens[0]
            file_name = tokens[3]
            channel = 1 if tokens[4] == 'A' else 2
            if sre_year == tokens[2]:
                try:
                    file_loc = file_list[file_name]
                    index_list.append('{}_{}_ch{}'.format(sre_year, file_name, channel))
                    location_list.append(file_loc)
                    speaker_list.append(sre_year + '_' + speaker_id)
                    channel_list.append(channel)
                    read_list.append('sph2pipe -f wav -p -c {} {}'.format(channel, file_loc))
                except KeyError:
                    pass

    print('Made {:d} files from {}.'.format(len(index_list), sre_year))
    return np.vstack([index_list, location_list, channel_list, speaker_list, read_list])


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
    read_list = []
    model_to_speaker = dict()
    with open(model_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[,:]+', line.strip())
            model_id = tokens[0]
            file_name = tokens[2]
            channel = 1 if tokens[3] == 'a' else 2
            speaker_id = tokens[4]
            model_to_speaker[model_id] = speaker_id
            try:
                file_loc = file_list[file_name]
                index_list.append('sre2008_{}_ch{}'.format(file_name, channel))
                location_list.append(file_loc)
                channel_list.append(channel)
                speaker_list.append('sre2008_' + speaker_id)
                read_list.append('sph2pipe -f wav -p -c {} {}'.format(channel, file_loc))
            except KeyError:
                pass

    with open(trials_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[,]+', line.strip())
            model_id = tokens[0]
            file_name = tokens[1]
            channel = 1 if tokens[2] == 'a' else 2
            target_type = tokens[3]
            try:
                file_loc = file_list[file_name]
                speaker_id = model_to_speaker[model_id]
                if target_type == 'target':
                    index_list.append('sre2008_{}_ch{}'.format(file_name, channel))
                    location_list.append(file_loc)
                    channel_list.append(channel)
                    speaker_list.append('sre2008_' + speaker_id)
                    read_list.append('sph2pipe -f wav -p -c {} {}'.format(channel, file_loc))
                    del file_list[file_name]
            except KeyError:
                pass

    print('Made {:d} files from sre2008.'.format(len(index_list)))
    return np.vstack([index_list, location_list, channel_list, speaker_list, read_list])


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
    read_list = []
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
            channel = 1 if tokens[3] == 'A' else 2
            try:
                file_loc = file_list[file_name]
                speaker_id = model_to_speaker[model_id]
                index_list.append('sre2010_{}_ch{}'.format(file_name, channel))
                location_list.append(file_loc)
                speaker_list.append('sre2010_' + speaker_id)
                channel_list.append(channel)
                read_list.append('sph2pipe -f wav -p -c {} {}'.format(channel, file_loc))
            except KeyError:
                pass

    with open(trials_key, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[,]+', line.strip())
            model_id = tokens[0]
            file_name = tokens[1]
            channel = 1 if tokens[2] == 'A' else 2
            target_type = tokens[3]
            try:
                speaker_id = model_to_speaker[model_id]
                file_loc = file_list[file_name]
                if target_type == 'target':
                    index_list.append('sre2010_{}_ch{}'.format(file_name, channel))
                    location_list.append(file_loc)
                    speaker_list.append('sre2010_' + speaker_id)
                    channel_list.append(channel)
                    read_list.append('sph2pipe -f wav -p -c {} {}'.format(channel, file_loc))
                    del file_list[file_name]
            except KeyError:
                pass

    print('Made {:d} files from sre2010.'.format(len(index_list)))
    return np.vstack([index_list, location_list, channel_list, speaker_list, read_list])


def make_swbd_cellular(data_root, data_loc, cellular=1):
    print('Making swbd cellular {} lists...'.format(cellular))
    swbd_loc = join_path(data_root, data_loc)

    bad_audio = [40019, 45024, 40022]
    stats_key = join_path(swbd_loc, 'doc{}/swb_callstats.tbl'.format('' if cellular == 1 else 's'))
    swbd_type = 'swbd_c{:d}_'.format(cellular)

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
    read_list = []
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
                read_list.append('sph2pipe -f wav -p -c 1 {}'.format(file_loc))
                index_list.append(swbd_type + file_name + '_ch2')
                location_list.append(file_loc)
                channel_list.append(2)
                speaker_list.append('sw_' + speaker_id2)
                read_list.append('sph2pipe -f wav -p -c 2 {}'.format(file_loc))
                del file_list['sw_' + str(file_name)]
            except KeyError:
                pass

    print('Made {:d} files swbd cellular {}.'.format(len(index_list), cellular))
    return np.vstack([index_list, location_list, channel_list, speaker_list, read_list])


def make_swbd_phase(data_root, data_loc, phase=1):
    print('Making swbd phase {} lists...'.format(phase))
    swbd_loc = join_path(data_root, data_loc)

    stats_key = join_path(swbd_loc, 'docs/callinfo.tbl')
    swbd_type = 'swbd_p{:d}_'.format(phase)

    file_list = get_file_list_as_dict(swbd_loc)

    index_list = []
    location_list = []
    channel_list = []
    speaker_list = []
    read_list = []
    with open(stats_key, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[,]+', line.strip())
            file_name = ('sw_' + tokens[0]) if phase == 3 else ('' + tokens[0].split('.')[0])
            speaker_id = str(tokens[2])
            channel = 1 if tokens[3] == 'A' else 2
            try:
                file_loc = file_list[file_name]
                index_list.append(swbd_type + file_name + '_ch{:d}'.format(channel))
                location_list.append(file_loc)
                channel_list.append(channel)
                speaker_list.append('sw_' + speaker_id)
                read_list.append('sph2pipe -f wav -p -c {} {}'.format(channel, file_loc))
            except KeyError:
                pass

    print('Made {:d} files swbd phase {}.'.format(len(index_list), phase))
    return np.vstack([index_list, location_list, channel_list, speaker_list, read_list])


def make_mixer6_calls(data_root, data_loc):
    print('Making mixer6 calls lists...')
    mx6_loc = join_path(data_root, data_loc)
    mx6_calls_loc = join_path(mx6_loc, 'data/ulaw_sphere')

    stats_key = join_path(mx6_loc, 'docs/mx6_calls.csv')
    file_list = get_file_list_as_dict(mx6_calls_loc)

    call_to_file = dict()
    for key in file_list.keys():
        call_id = re.split('[_]+', key)[2]
        call_to_file[call_id] = key

    index_list = []
    location_list = []
    channel_list = []
    speaker_list = []
    read_list = []
    with open(stats_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[,]+', line.strip())
            call_id = tokens[0]
            speaker_id1 = tokens[4]
            speaker_id2 = tokens[12]
            try:
                file_name = call_to_file[call_id]
                file_loc = file_list[file_name]
                index_list.append('MX6_CALLS_{}_ch1'.format(file_name))
                location_list.append(file_loc)
                channel_list.append(1)
                speaker_list.append('MX6_{}'.format(speaker_id1))
                read_list.append('sph2pipe -f wav -p -c 1 {}'.format(file_loc))
                index_list.append('MX6_CALLS_{}_ch2'.format(file_name))
                location_list.append(file_loc)
                channel_list.append(2)
                speaker_list.append('MX6_{}'.format(speaker_id2))
                read_list.append('sph2pipe -f wav -p -c 2 {}'.format(file_loc))
            except KeyError:
                pass
    print('Made {:d} files from mixer6 calls.'.format(len(index_list)))
    return np.vstack([index_list, location_list, channel_list, speaker_list, read_list])


def make_mixer6_mic(data_root, data_loc):
    print('Making mixer6 mic lists...')
    mx6_loc = join_path(data_root, data_loc)
    mx6_mic_loc = join_path(mx6_loc, 'data/pcm_flac')

    bad_audio = ['20091208_091618_HRM_120831']

    stats_key = join_path(mx6_loc, 'docs/mx6_ivcomponents.csv')
    file_list = dict()
    mic_idx = ['02', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']  # Omitting 01, 03 and 14
    for idx in mic_idx:
        mic_loc = join_path(mx6_mic_loc, 'CH' + idx)
        mic_file_list = get_file_list_as_dict(mic_loc, pattern='*.flac')
        file_list = {**mic_file_list, **file_list}

    index_list = []
    location_list = []
    channel_list = []
    speaker_list = []
    read_list = []
    with open(stats_key, 'r') as f:
        for line in f.readlines()[1:]:
            tokens = re.split('[,]+', line.strip())
            session_id = tokens[0]
            speaker_id = re.split('[_]+', session_id)[3]
            start_time = tokens[7]
            end_time = tokens[8]
            if session_id not in bad_audio:
                for idx in mic_idx:
                    file_name = '{}_CH{}'.format(session_id, idx)
                    try:
                        file_loc = file_list[file_name]
                        index_list.append('MX6_MIC_{}'.format(file_name))
                        location_list.append(file_loc)
                        channel_list.append(1)
                        speaker_list.append('MX6_{}'.format(speaker_id))
                        read_list.append('sox -t flac {} -r 8k -t wav -V0 - trim {} {}'
                                         .format(file_loc, start_time, float(end_time) - float(start_time)))
                    except KeyError:
                        pass

    print('Made {:d} files from mixer6 mic.'.format(len(index_list)))
    return np.vstack([index_list, location_list, channel_list, speaker_list, read_list])


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
    read_list = []
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
                read_list.append('sph2pipe -f wav -p -c 1 {}'.format(file_loc))
                del file_list[file_name]
            except KeyError:
                pass

    print('Made {:d} enrollment files.'.format(len(index_list)))
    enrollment_data = np.vstack([index_list, location_list, channel_list, speaker_list, read_list]).T

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
    read_list = []
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
                read_list.append('sph2pipe -f wav -p -c 1 {}'.format(file_loc))
                language_list.append(call_to_language[call_id])
                target_list.append(target_type)
                del file_list[file_name]
            except KeyError:
                pass

    print('Made {:d} test files.'.format(len(index_list)))
    test_data = np.vstack([index_list, location_list, channel_list, speaker_list, read_list, target_list,
                           language_list]).T

    return enrollment_data, test_data
