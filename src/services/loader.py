from os.path import join as join_path

import numpy as np

from constants.app_constants import MFCC_DIR
from services.common import run_parallel, load_array
from services.feature import load_feature


class SREBatchLoader:
    def __init__(self, location, args, n_features, speaker_dict, batch_size):
        self.location = join_path(location, MFCC_DIR)
        self.classes = np.array([speaker_dict[k] for k in args[:, 3]])
        self.feature_idx = np.array(args[:, 0])
        self.n_features = n_features
        self.frames = np.array(args[:, -1], dtype=int)
        self.data_len = self.feature_idx.shape[0]
        if not self.data_len == self.classes.shape[0]:
            raise RuntimeError('Length of input and classes does not match.')

        self.batch_size = batch_size
        self.batch_pointer = 0
        self.n_batches = int(self.data_len / batch_size)
        self.max_size = self.n_batches * batch_size
        self.permutation_idx = np.random.permutation(self.data_len)
        self.batch_splits = np.array_split(self.permutation_idx[:self.max_size], self.n_batches)

    def get_batch_size(self):
        return self.batch_size

    def load_feature(self, idx):
        return load_array(join_path(self.location, '{}.npy'.format(self.feature_idx[idx])))

    def next(self):
        current_split = self.batch_splits[self.batch_pointer]
        self.batch_pointer = self.batch_pointer + 1
        if self.batch_pointer == self.n_batches:
            self.reset()
        frames = self.frames[current_split]
        max_len = max(frames)
        if sum(frames/max_len < 0.95) < (0.05 * self.batch_size):
            print('Batch formation issue.')
            return self.next()
        np_features = np.zeros([self.batch_size, self.n_features, max_len])
        features = run_parallel(self.load_feature, current_split, n_workers=6, p_bar=False)
        for i, f in enumerate(features):
            np_features[i, :, max_len - frames[i]:] = f
        return np_features, self.classes[current_split]

    def reset(self):
        self.batch_pointer = 0
        self.permutation_idx = np.random.permutation(self.data_len)
        self.batch_splits = np.array_split(self.permutation_idx[:self.max_size], self.n_batches)

    def total_batches(self):
        return self.n_batches


class SRESplitBatchLoader:
    def __init__(self, location, args, n_features, splits, batch_size):
        location = join_path(location, MFCC_DIR)
        self.frames = np.array(args[:, -1], dtype=int)
        self.data_len = self.frames.shape[0]
        self.file_loc = np.array([join_path(location, x + '.npy') for x in args[:, 0]])
        self.speakers = args[:, 3]
        self.n_features = n_features
        self.batch_size = batch_size if type(batch_size) is list else [batch_size] * (len(splits) - 1)
        self.current_split = 0
        self.splits = splits

        index = np.arange(0, self.data_len, dtype=int)
        start = 0
        free_start = np.where(self.frames > splits[len(splits) - 1])[0][0]
        self.splits_index = []
        for s in splits[1:]:
            end = np.where(self.frames > s)[0][0]
            self.splits_index.append(np.hstack([index[start: end], index[free_start:]]))
            start = end

        self.batch_pointer = 0
        self.current_split_index = self.splits_index[self.current_split]
        self.n_batches = int(len(self.current_split_index) / self.batch_size[self.current_split])
        self.permutation_idx = np.random.permutation(len(self.current_split_index))
        self.batch_splits = np.array_split(self.permutation_idx[:self.n_batches * self.batch_size[self.current_split]],
                                           self.n_batches)

    def get_batch_size(self):
        return self.batch_size[self.current_split]

    def next(self):
        current_batch_idx = self.current_split_index[self.batch_splits[self.batch_pointer]]
        self.batch_pointer = self.batch_pointer + 1
        if self.batch_pointer == self.n_batches:
            self.reset()

        frame_len = self.splits[self.current_split]
        frames = self.frames[current_batch_idx]
        file_loc = self.file_loc[current_batch_idx]
        np_features = np.zeros([self.batch_size[self.current_split], self.n_features, frame_len])
        features = run_parallel(load_feature, file_loc, n_workers=10, p_bar=False)
        for i, f in enumerate(features):
            f_len = frames[i]
            if f_len > frame_len:
                idx = np.random.choice(f_len - frame_len, 1)[0]
            else:
                idx = 0
            np_features[i, :, :] = f[:, idx:(idx + frame_len)]
        return np_features, self.speakers[current_batch_idx]

    def reset(self):
        self.batch_pointer = 0
        self.permutation_idx = np.random.permutation(len(self.current_split_index))
        self.batch_splits = np.array_split(self.permutation_idx[:self.n_batches * self.batch_size[self.current_split]],
                                           self.n_batches)

    def set_split(self, split):
        self.current_split = split
        self.current_split_index = self.splits_index[split]
        self.n_batches = int(len(self.current_split_index) / self.batch_size[self.current_split])
        self.reset()

    def total_batches(self):
        return self.n_batches


class SRETestLoader:
    def __init__(self, location, args, n_features, batch_size):
        location = join_path(location, MFCC_DIR)
        self.frames = np.array(args[:, -1], dtype=int)
        idx = np.argsort(self.frames)
        args = args[idx]
        self.frames = self.frames[idx]
        self.args_idx = args[:, 0]
        self.file_loc = np.array([join_path(location, x + '.npy') for x in args[:, 0]])
        self.n_features = n_features
        self.batch_size = batch_size

        self.batch_pointer = 0
        self.n_batches = int(self.frames.shape[0] / batch_size)
        data_len = self.n_batches * batch_size
        self.batch_splits = np.array_split(np.linspace(0, data_len - 1, data_len, dtype=int), self.n_batches)

    def get_batch_size(self):
        return self.batch_size

    def get_last_idx(self):
        return self.n_batches * self.batch_size

    def next(self):
        if self.batch_pointer == self.n_batches:
            raise Exception('No batches left.')

        current_batch_idx = self.batch_splits[self.batch_pointer]
        self.batch_pointer = self.batch_pointer + 1

        frames = self.frames[current_batch_idx]
        frame_len = frames[0]
        file_loc = self.file_loc[current_batch_idx]
        np_features = np.zeros([self.batch_size, self.n_features, frame_len])
        features = run_parallel(load_feature, file_loc, n_workers=6, p_bar=False)
        for i, f in enumerate(features):
            f_len = frames[i]
            if f_len > frame_len:
                idx = np.random.choice(f_len - frame_len, 1)[0]
            else:
                idx = 0
            np_features[i, :, :] = f[:, idx:(idx + frame_len)]
        return np_features, self.args_idx[current_batch_idx]

    def total_batches(self):
        return self.n_batches


class OnlineBatchLoader:
    def __init__(self, args_list, feature_extractor, batch_size):
        self.args_list = args_list
        self.location_list = args_list[:, 0]
        self.file_idx = args_list[:, 2]
        self.speakers = args_list[:, 3]
        self.feature_extractor = feature_extractor
        self.data_len = args_list.shape[0]

        self.batch_size = batch_size
        self.batch_pointer = 0
        self.n_batches = int(self.data_len / batch_size)
        self.max_size = self.n_batches * batch_size
        self.permutation_idx = np.random.permutation(self.data_len)
        self.batch_splits = np.array_split(self.permutation_idx[:self.max_size], self.n_batches)

    def get_batch_size(self):
        return self.batch_size

    def next(self):
        current_split = self.batch_splits[self.batch_pointer]
        features = run_parallel(self.feature_extractor.extract_piped_file_with_sad_and_cmvn,
                                args_list=self.args_list[current_split], n_workers=10, p_bar=False)
        feature_len = features[0].shape[0]
        frames_list = [f.shape[1] for f in features]
        max_frames = max(frames_list)
        np_features = np.zeros([self.batch_size, feature_len, max_frames])
        for i, f in enumerate(features):
            np_features[i, :, max_frames - frames_list[i]:] = f
        self.batch_pointer = self.batch_pointer + 1
        if self.batch_pointer == self.n_batches:
            self.reset()
        return np_features

    def reset(self):
        self.batch_pointer = 0
        self.permutation_idx = np.random.permutation(self.data_len)
        self.batch_splits = np.array_split(self.permutation_idx[:self.max_size], self.n_batches)
