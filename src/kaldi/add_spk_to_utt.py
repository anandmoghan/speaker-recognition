from os.path import join as join_path

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--data', type=str, help='Data Dir.')
parser.add_argument('--spk-present', type=bool, default=True, help='Data Dir.')
parser.add_argument('--file', type=str, help='File to append')
args = parser.parse_args()


def make_spaced_dict(file, spk_present=False):
    file_dict = dict()
    with open(file) as f:
        for line in f.readlines():
            tokens = line.strip().split()
            start_pt = len(tokens[1]) + 1 if spk_present else 0
            file_dict[tokens[0][start_pt:]] = tokens[1]
    return file_dict


utt2spk = join_path(args.data, 'utt2spk')
new_file = join_path(args.data, args.file)

utt2spk_dict = make_spaced_dict(utt2spk, spk_present=args.spk_present)
new_file_dict = make_spaced_dict(new_file)

with open(new_file, 'w') as f:
    for utt, item in new_file_dict.items():
        try:
            spk = utt2spk_dict[utt]
            utt = '{}-{}'.format(spk, utt)
        except KeyError:
            pass
        f.write('{} {}\n'.format(utt, item))
