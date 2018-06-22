import multiprocessing as mp
import numpy as np
import os

from fnmatch import fnmatch
from itertools import chain


def get_file_list(location, pattern='*.sph'):
    file_list = []
    for path, _, files in os.walk(location):
        for name in files:
            if fnmatch(name, pattern):
                file_list = np.append(file_list, os.path.join(path, name))
    return file_list


def run_parallel(func, args_list, batch_size=64, n_workers=10):
    n_parts = len(args_list) / batch_size
    args_split = np.array_split(args_list, n_parts)
    pool = mp.Pool(n_workers)
    res = pool.map(func, args_split)
    pool.close()
    if res is not None:
        return list(chain.from_iterable(res))
