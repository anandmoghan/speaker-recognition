from os.path import join as join_path
from subprocess import Popen, PIPE
from fnmatch import fnmatch
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import pickle
import time
import os

from constants.app_constants import DATA_DIR, EMB_DIR, LOGS_DIR, MFCC_DIR, MODELS_DIR, PLDA_DIR, VAD_DIR, TMP_DIR


def append_cwd_to_python_path(cmd):
    return 'export PYTHONPATH=$PYTHONPATH:$CWD && {}'.format(cmd)


def arrange_data(data, window, hop):
    n_frames = data.shape[2]
    start_frame = 0
    end_frame = start_frame + window
    new_data = []
    while end_frame <= n_frames:
        new_data.append(data[:, :, start_frame:end_frame])
        start_frame += hop
        end_frame = start_frame + window

    return np.concatenate(new_data, axis=2)


def create_directories(save_loc):
    make_directory(save_loc)
    make_directory(join_path(save_loc, DATA_DIR))
    make_directory(join_path(save_loc, LOGS_DIR))
    make_directory(join_path(save_loc, VAD_DIR))
    make_directory(join_path(save_loc, MFCC_DIR))
    make_directory(join_path(save_loc, MODELS_DIR))
    make_directory(join_path(save_loc, PLDA_DIR))
    make_directory(join_path(save_loc, EMB_DIR))
    make_directory(join_path(save_loc, TMP_DIR))


def create_wav(save_loc, args):
    make_directory(save_loc)
    wav_file_loc = [join_path(save_loc, '{}.wav'.format(a)) for a in args[:, 0]]
    print('Checking if files exist...')
    result = run_parallel(os.path.exists, wav_file_loc, n_workers=20)
    present = np.sum(np.array(result, dtype=int))
    print('{} files present.'.format(present))
    absent = args.shape[0] - present
    if absent > 0:
        result = np.invert(result)
        convert_cmd = np.array(['{} > {}'.format(cmd, wav_file_loc[i]) for i, cmd in enumerate(args[:, 4])])
        convert_cmd = convert_cmd[result]
        print('Converting...')
        result = run_parallel(run_command, convert_cmd, n_workers=20)
        converted = np.sum(np.array(result, dtype=int))
        print('Converted {} files to wav.'.format(converted))
    args[:, 1] = wav_file_loc
    args[:, 4] = ['cat {}'.format(l) for l in wav_file_loc]
    return args


def delete_directory(path):
    try:
        for item in os.listdir(path):
            item = join_path(path, item)
            if os.path.isdir(item):
                delete_directory(item)
            else:
                os.remove(item)
        os.removedirs(path)
    except FileNotFoundError:
        pass


def get_file_list(location, pattern='*.sph'):
    file_list = []
    for path, _, files in os.walk(location):
        for name in files:
            if fnmatch(name, pattern):
                file_list = np.append(file_list, os.path.join(path, name))
    if len(file_list) == 0:
        raise Warning('No {} files in {}'.format(pattern, location))
    return file_list


def get_file_list_as_dict(location, pattern='*.sph', ext=False):
    file_list = dict()
    for path, _, files in os.walk(location):
        for name in files:
            if fnmatch(name, pattern):
                key = name
                if not ext:
                    key = key[:-len(pattern)+1]
                file_list[key] = os.path.join(path, name)
    if len(file_list) == 0:
        raise Warning('No {} files in {}'.format(pattern, location))
    return file_list


def get_free_gpu():
    output, _ = run_command('nvidia-smi')
    output = output.split('\n')
    start_idx = 7
    n_gpu = 0
    all_gpu_break = 0
    for i, line in enumerate(output[start_idx:]):
        line = line.strip()
        if line == '':
            all_gpu_break = start_idx + i
            break
        elif line[0] == '+':
            n_gpu += 1

    start_idx = all_gpu_break + 5
    all_gpu_dict = dict(zip(range(n_gpu), [True] * n_gpu))
    for line in output[start_idx:-2]:
        try:
            gpu = int(line.split()[1])
            del all_gpu_dict[gpu]
        except ValueError:
            break
        except KeyError:
            pass
    return list(all_gpu_dict.keys())


def get_index_array(max_len, shuffle=False):
    if shuffle:
        return np.random.permutation(max_len)
    else:
        return np.linspace(0, max_len - 1, max_len, dtype=int)


def get_time_stamp():
    return time.strftime('%b %d, %Y %l:%M:%S%p')


def load_array(file_name):
    return np.load(file_name)


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def make_dict(key_list, value_list):
    return dict([(key, value) for key, value in zip(key_list, value_list)])


def print_script_args(args):
    print(' '.join(args))
    print()


def print_time_stamp():
    print(time.strftime(' %b %d, %Y %l:%M:%S%p'))


def put_time_stamp(text):
    return time.strftime(' %b %d, %Y %l:%M:%S%p - ') + text


def remove_duplicates(args):
    _, unique_idx = np.unique(args[:, 0], return_index=True)
    return args[unique_idx, :], args.shape[0] - len(unique_idx)


def run_command(cmd):
    output, error = Popen(cmd, stdout=PIPE, shell=True).communicate()
    return output.decode("utf-8"), error.decode("utf-8") if error is not None else error


def run_parallel(func, args_list, n_workers=10, p_bar=True):
    pool = mp.Pool(n_workers)
    if p_bar:
        if type(args_list) is list:
            total_len = len(args_list)
        else:
            total_len = args_list.shape[0]
        out = tqdm(pool.imap(func, args_list), total=total_len)
    else:
        out = pool.map(func, args_list)
    pool.close()
    # pool.join()
    if out is not None:
        return list(out)


def save_array(file_name, obj):
    np.save(file_name, obj)


def save_batch_array(location, args_idx, obj, ext='.npy'):
    for i, arg in enumerate(args_idx):
        save_loc = os.path.join(location, arg + ext)
        save_array(save_loc, obj[i])


def save_object(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def split_dict(old_dict, keys):
    new_dict = dict()
    for key in keys:
        try:
            new_dict[key] = old_dict[key]
        except KeyError:
            raise Exception('{}: {} not found.'.format('SPLIT_DICT KeyError', key))
    return new_dict


def sort_by_index(args_list):
    return args_list[args_list[:, 0].argsort()]


def split_args_list(args_list, split=0.1, shuffle=True):
    idx = get_index_array(args_list.shape[0], shuffle)
    split = int(split * args_list.shape[0])
    return args_list[idx[:split], :], args_list[idx[split:], :]


def tensorflow_debug(debug=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' if debug is False else '1'


def use_gpu(gpu_id):
    if gpu_id < 0:
        available = get_free_gpu()
        if len(available) == 0:
            raise Exception('No free gpu devices available.')
        else:
            gpu_id = available[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
