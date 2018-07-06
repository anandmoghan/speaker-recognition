from fnmatch import fnmatch
from tqdm import tqdm

import multiprocessing as mp
import numpy as np
import pickle
import time
import os


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


# def run_parallel(func, args_list, batch_size=64, n_workers=10, p_bar=True):
#     n_parts = len(args_list) / batch_size
#     args_split = np.array_split(args_list, n_parts)
#     pool = mp.Pool(n_workers)
#     if p_bar:
#         res = tqdm(pool.imap(func, args_split), total=int(n_parts))
#     else:
#         res = pool.map(func, args_split)
#     pool.close()
#     if res is not None:
#         return list(chain.from_iterable(res))


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


def save_object(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def sort_by_index(args_list):
    return args_list[args_list[:, 0].argsort()]


def tensorflow_debug(debug=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' if debug is False else '1'

