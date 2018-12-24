from os.path import join as join_path

import argparse as ap
import numpy as np


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--splits', type=int, default=20, help='No of Splits')
    parser.add_argument('--file', type=str, help='SCP File')
    parser.add_argument('--prefix', type=str, help='Prefix for split files.')
    parser.add_argument('--dest', type=str, help='Destination')
    parser.add_argument('--ext', type=str, default='scp', help='Destination')
    return parser.parse_args()


def split_scp(file_path, num_splits, dest_loc, prefix, ext='scp'):
    with open(file_path, 'r') as f:
        lines = np.array(f.readlines())

    idx = np.array_split(np.arange(0, len(lines), dtype=int), num_splits)
    for i in range(num_splits):
        file_name = join_path(dest_loc, '{}.{}.{}'.format(prefix, i+1, ext))
        with open(file_name, 'w') as f:
            f.writelines(lines[idx[i]])

if __name__ == '__main__':
    args = parse_args()
    split_scp(args.file, args.splits, args.dest, args.prefix, args.ext)
