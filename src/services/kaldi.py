import re


def scp_to_dict(scp_file):
    scp_dict = dict()
    with open(scp_file, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            scp_dict[tokens[0]] = tokens[1]
    return scp_dict
