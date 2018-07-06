from os.path import join as join_path

import numpy as np

from constants.app_constants import MFCC_DIR
from services.common import run_parallel, load_array


class SREBatchLoader:
    def __init__(self, location, args, n_features, speaker_dict, batch_size):
        self.location = join_path(location, MFCC_DIR)
        self.classes = np.array([speaker_dict[k] for k in args[:, 3]])
        self.feature_idx = np.array(args[:, 0])
        self.n_features = n_features
        self.frames = np.array(args[:, 4], dtype=int)
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


class SRESplitBatchLoader(SREBatchLoader):
    def __init__(self, location, args, n_features, speaker_dict, batch_size):
        super().__init__(location, args, n_features, speaker_dict, batch_size)

    def next(self):
        print('Here')
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

    def next(self):
        current_split = self.batch_splits[self.batch_pointer]
        features = run_parallel(self.feature_extractor.extract_sph_file_with_sad_and_cmvn,
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
