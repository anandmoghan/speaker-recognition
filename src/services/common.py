import multiprocessing as mp
import time
from tqdm import tqdm

import numpy as np
import pickle
import os

from fnmatch import fnmatch
from itertools import chain


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


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def put_time_stamp(text):
    return time.strftime(' %b %d, %Y %l:%M%p - ') + text


def run_parallel(func, args_list, batch_size=64, n_workers=10, p_bar=True):
    n_parts = len(args_list) / batch_size
    args_split = np.array_split(args_list, n_parts)
    pool = mp.Pool(n_workers)
    if p_bar:
        res = tqdm(pool.imap(func, args_split), total=int(n_parts))
    else:
        res = list(pool.map(func, args_split))
    pool.close()
    if res is not None:
        return list(chain.from_iterable(res))


def save_object(file_name, obj):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
