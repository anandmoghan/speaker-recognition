from os.path import join as join_path

import argparse as ap
import numpy as np

parser = ap.ArgumentParser()
parser.add_argument('--splits', type=int, default=20, help='No of Splits')
parser.add_argument('--file', type=str, help='SCP File')
parser.add_argument('--prefix', type=str, help='Prefix for split files.')
parser.add_argument('--dest', type=str, help='Destination')
args = parser.parse_args()

with open(args.file, 'r') as f:
    lines = np.array(f.readlines())

idx = np.array_split(np.arange(0, len(lines), dtype=int), args.splits)
for i in range(args.splits):
    file_name = join_path(args.dest, '{}.{}.scp'.format(args.prefix, i+1))
    with open(file_name, 'w') as f:
        f.writelines(lines[idx[i]])
