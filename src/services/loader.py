from os.path import join as join_path

import numpy as np

from constants.app_constants import FEATS_SCP_FILE, TMP_SCP_FILE
from services.common import get_index_array
from services.kaldi import spaced_file_to_dict, read_feats


class BatchLoader:
    def __init__(self, args_list, n_features, batch_size, model_tag, multiple=1, shuffle=True, save_loc='../save'):
        self.batch_size = batch_size
        self.n_features = n_features
        self.save_loc = save_loc
        self.multiple = multiple
        self.shuffle = shuffle

        idx = np.argsort(args_list[:, -1])
        self.index_list = np.array(args_list[idx, 0])
        self.frames = np.array(args_list[idx, -1], dtype=int)
        self.labels = np.array(args_list[idx, 3])

        self.feats_dict = spaced_file_to_dict(join_path(save_loc, FEATS_SCP_FILE))
        self.tmp_scp_file = '{}_{}'.format(join_path(save_loc, TMP_SCP_FILE), model_tag)

        self.batch_pointer = 0
        self.n_batches = int(self.index_list.shape[0] / batch_size) + (0 if args_list.shape[0] % batch_size == 0 else 1)
        idx = get_index_array(self.index_list.shape[0], shuffle)
        self.batch_splits = np.array_split(idx, self.n_batches)

    def get_batch_size(self):
        return self.batch_size

    def get_current_batch(self):
        return self.batch_pointer

    def increment_pointer(self):
        self.batch_pointer = self.batch_pointer + 1
        if self.batch_pointer == self.n_batches:
            self.reset()

    def next(self):
        current_batch_idx = self.batch_splits[self.batch_pointer]
        self.batch_size = len(current_batch_idx)
        self.increment_pointer()

        frames = self.frames[current_batch_idx]
        labels = np.array(self.labels[current_batch_idx])
        max_len = int(frames[0] / self.multiple) * self.multiple

        with open(self.tmp_scp_file, 'w') as f:
            for i, f_len in enumerate(frames):
                if f_len > max_len and self.shuffle:
                    idx = np.random.choice(f_len - max_len, 1)[0]
                else:
                    idx = 0
                utt = self.index_list[current_batch_idx[i]]
                f.write('{} {}[{}:{}]\n'.format(utt, self.feats_dict[utt], idx, idx + max_len - 1))

        _, np_features = read_feats(self.tmp_scp_file, self.n_features)
        return np.array(np_features), labels

    def reset(self):
        idx = get_index_array(self.index_list.shape[0], self.shuffle)
        self.batch_pointer = 0
        self.batch_splits = np.array_split(idx, self.n_batches)

    def set_current_batch(self, value):
        self.batch_pointer = value

    def set_multiple(self, value):
        self.multiple = value

    def total_batches(self):
        return self.n_batches


class FixedBatchLoader(BatchLoader):
    def __init__(self, args_list, n_features, batch_size, max_frames, model_tag, multiple=1, shuffle=True, save_loc='../save'):
        super().__init__(args_list, n_features, batch_size, model_tag, multiple, shuffle, save_loc)
        self.max_frames = max_frames
        self.multiple = multiple

    def next(self):
        current_batch_idx = self.batch_splits[self.batch_pointer]
        self.batch_size = len(current_batch_idx)
        self.increment_pointer()

        frames = self.frames[current_batch_idx]
        labels = np.array(self.labels[current_batch_idx])
        max_len = np.min(frames) if self.max_frames > np.min(frames) else self.max_frames
        max_len = int(max_len / self.multiple) * self.multiple

        with open(self.tmp_scp_file, 'w') as f:
            for i, f_len in enumerate(frames):
                if f_len > max_len and self.shuffle:
                    idx = np.random.choice(f_len - max_len, 1)[0]
                else:
                    idx = 0
                utt = self.index_list[current_batch_idx[i]]
                f.write('{} {}[{}:{}]\n'.format(utt, self.feats_dict[utt], idx, idx + max_len - 1))

        _, np_features = read_feats(self.tmp_scp_file, self.n_features)
        return np.array(np_features), labels


class SplitBatchLoader:
    def __init__(self, args_list, n_features, batch_size, splits, model_tag, multiple=1, shuffle=True, save_loc='../save'):
        idx = np.argsort(args_list[:, -1])
        frames = np.array(args_list[idx, -1], dtype=int)

        start = 0
        free_start = np.where(frames > splits[len(splits) - 1])[0][0]
        self.batch_loaders = []
        for s in splits[1:]:
            end = np.where(frames > s)[0][0]
            split_idx = np.hstack([idx[start: end], idx[free_start:]])
            self.batch_loaders.append(FixedBatchLoader(args_list[split_idx, :], n_features, batch_size, s, model_tag, multiple, shuffle, save_loc))
            start = end

        self.splits = splits
        self.current_split = 0
        self.current_batch_loader = self.batch_loaders[self.current_split]

    def get_batch_size(self):
        return self.current_batch_loader.get_batch_size()

    def get_current_batch(self):
        return self.current_batch_loader.get_current_batch()

    def get_splits(self):
        return list(range(0, len(self.splits) - 1))

    def next(self):
        return self.current_batch_loader.next()

    def reset(self):
        self.current_batch_loader.reset()

    def set_split(self, value):
        self.current_split = value
        self.current_batch_loader = self.batch_loaders[self.current_split]

    def total_batches(self):
        return self.current_batch_loader.total_batches()
