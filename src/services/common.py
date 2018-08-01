from fnmatch import fnmatch
from subprocess import Popen, PIPE

from tqdm import tqdm
from os.path import join as join_path

import multiprocessing as mp
import numpy as np
import os
import pickle
import time

from constants.app_constants import DATA_DIR, EMB_DIR, MFCC_DIR, MODELS_DIR, VAD_DIR


def create_directories(save_loc):
    make_directory(save_loc)
    make_directory(join_path(save_loc, DATA_DIR))
    make_directory(join_path(save_loc, VAD_DIR))
    make_directory(join_path(save_loc, MFCC_DIR))
    make_directory(join_path(save_loc, MODELS_DIR))
    make_directory(join_path(save_loc, EMB_DIR))


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


def get_file_list(location, pattern='*.sph'):
    file_list = []
    for path, _, files in os.walk(location):
        for name in files:
            if fnmatch(name, pattern):
                file_list = np.append(file_list, os.path.join(path, name))
    if len(file_list) == 0:
        raise Warning('No {} files in {}'.format(pattern, location))
    return file_list


def get_file_list_as_dict(location, pattern='*.sph'):
    file_list = dict()
    for path, _, files in os.walk(location):
        for name in files:
            if fnmatch(name, pattern):
                file_list[name[:-len(pattern)+1]] = os.path.join(path, name)
    if len(file_list) == 0:
        raise Warning('No {} files in {}'.format(pattern, location))
    return file_list


def load_array(file_name):
    return np.load(file_name)


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def make_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def sort_by_index(args_list):
    return args_list[args_list[:, 0].argsort()]


def tensorflow_debug(debug=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' if debug is False else '1'


def use_gpu(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
