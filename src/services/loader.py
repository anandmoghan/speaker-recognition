from collections import Counter
from os.path import join as join_path

import numpy as np

from constants.app_constants import FEATS_SCP_FILE, TMP_SCP_FILE, SPK_EMB_SCP_FILE, \
    ENROLL_SPK_EMB_SCP_FILE
from services.common import get_index_array, split_dict, make_dict
from services.kaldi import spaced_file_to_dict, read_feats, make_labels_to_index_dict, read_vector
from services.sre_data import split_trials_file


class BatchLoader:
    def __init__(self, args_list, n_features, batch_size, model_tag, multiple=1, shuffle=True, save_loc='../save'):
        self.batch_size = batch_size
        self.n_features = n_features
        self.save_loc = save_loc
        self.multiple = multiple
        self.shuffle = shuffle

        idx = np.argsort(np.array(args_list[:, -1], dtype=int))
        self.index_list = np.array(args_list[idx, 0])
        self.frames = np.array(args_list[idx, -1], dtype=int)
        self.labels = np.array(args_list[idx, 3])

        self.feats_dict = spaced_file_to_dict(join_path(save_loc, FEATS_SCP_FILE))
        self.tmp_scp_file = join_path(save_loc, TMP_SCP_FILE.format(model_tag))

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
    def __init__(self, args_list, n_features, batch_size, max_frames, model_tag, multiple=1, shuffle=True,
                 save_loc='../save'):
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
    def __init__(self, args_list, n_features, batch_size, splits, model_tag, multiple=1, shuffle=True,
                 save_loc='../save'):
        idx = np.argsort(np.array(args_list[:, -1], dtype=int))
        frames = np.array(args_list[idx, -1], dtype=int)

        start = 0
        free_start = np.where(frames > splits[len(splits) - 1])[0][0]
        self.batch_loaders = []
        for s in splits[1:]:
            end = np.where(frames > s)[0][0]
            split_idx = np.hstack([idx[start: end], idx[free_start:]])
            self.batch_loaders.append(FixedBatchLoader(args_list[split_idx, :], n_features, batch_size, s, model_tag,
                                                       multiple, shuffle, save_loc))
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


class ExtractLoader(FixedBatchLoader):
    def __init__(self, args_list, n_features, batch_size, max_frames, model_tag, multiple=1, shuffle=True,
                 save_loc='../save'):
        super().__init__(args_list, n_features, batch_size, max_frames, model_tag, multiple, shuffle, save_loc)

    def next(self):
        current_batch_idx = self.batch_splits[self.batch_pointer]
        self.batch_size = len(current_batch_idx)
        self.increment_pointer()

        frames = self.frames[current_batch_idx]
        labels = np.array(self.index_list[current_batch_idx])
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


class LabelBatchLoader:
    def __init__(self, args_list, label_to_index_list, n_features, batch_size, min_frames, max_frames,
                 model_tag, multiple=1, shuffle=True, save_loc='../save'):
        self.index_list = np.array(args_list[:, 0])
        self.duration_list = np.array(args_list[:, 1], dtype=int)
        self.labels = np.array(args_list[:, 2])
        self.n_features = n_features
        self.batch_size = batch_size
        self.half_batch_size = int(batch_size / 2)
        self.multiple = multiple
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.model_tag = model_tag
        self.label_to_index_list = label_to_index_list

        labels = np.array(list(label_to_index_list.keys()))
        counts = np.array([len(label_to_index_list[key]) for key in labels])
        self.main_labels = labels[counts >= self.half_batch_size]
        if shuffle:
            np.random.shuffle(self.main_labels)

        self.feats_dict = spaced_file_to_dict(join_path(save_loc, FEATS_SCP_FILE))
        self.spk_emb_dict = spaced_file_to_dict(join_path(save_loc, SPK_EMB_SCP_FILE))
        self.tmp_scp_file = join_path(save_loc, TMP_SCP_FILE.format(model_tag))
        self.n_batches = len(self.main_labels)

        self.batch_pointer = 0
        self.frame_len = [self.get_frame_len() for _ in range(self.n_batches)]
        self.batch_splits = self.make_batches()

    def get_batch_size(self):
        return self.batch_size

    def get_current_batch(self):
        return self.batch_pointer

    def get_frame_len(self):
        frame_len = self.min_frames + np.random.choice(self.max_frames - self.min_frames, 1)[0]
        return int(frame_len / self.multiple) * self.multiple

    def increment_pointer(self):
        self.batch_pointer = self.batch_pointer + 1
        if self.batch_pointer == self.n_batches:
            self.reset()

    def make_batches(self):
        print('{}: Making batches...'.format(self.model_tag))
        batch_splits = []
        for label in self.main_labels:
            idx = np.array(self.label_to_index_list[label])
            idx = idx[np.random.choice(len(idx), self.half_batch_size, replace=False)]
            count = 0
            while count < self.half_batch_size:
                other_label = self.main_labels[np.random.randint(self.n_batches)]
                if other_label != label:
                    other_idx = np.array(self.label_to_index_list[other_label])
                    other_idx = other_idx[np.random.randint(len(other_idx))]
                    idx = np.append(idx, other_idx)
                    count = count + 1
            batch_splits.append(idx)
        return batch_splits

    def next(self):
        current_batch_idx = self.batch_splits[self.batch_pointer]
        index_list = self.index_list[current_batch_idx]
        duration_list = self.duration_list[current_batch_idx]
        labels = self.labels[current_batch_idx]
        main_label = self.main_labels[self.batch_pointer]
        frame_len = self.frame_len[self.batch_pointer]
        self.increment_pointer()

        with open(self.tmp_scp_file, 'w') as f:
            for utt, start in zip(index_list, duration_list):
                f.write('{} {}[{}:{}]\n'.format(utt, self.feats_dict[utt], start, start + frame_len - 1))
        _, np_features = read_feats(self.tmp_scp_file, self.n_features)

        with open(self.tmp_scp_file, 'w') as f:
            f.write('{} {}\n'.format(main_label, self.spk_emb_dict[main_label]))
        _, emb_vector = read_vector(self.tmp_scp_file, float)

        out_labels = np.array([1 if label == main_label else 0 for label in labels])
        return np.array(np_features), out_labels, np.array(emb_vector).reshape([1, -1])

    def reset(self):
        self.batch_pointer = 0
        if self.shuffle:
            np.random.shuffle(self.main_labels)
            self.frame_len = [self.get_frame_len() for _ in range(self.n_batches)]
            self.batch_splits = self.make_batches()

    def set_current_batch(self, value):
        self.batch_pointer = value

    def total_batches(self):
        return self.n_batches


class LabelExtractLoader:
    def __init__(self, trials_file, test_list, n_features, max_batch_size, model_tag, multiple=1, save_loc='../save'):
        index_list, label_list, target_list = split_trials_file(trials_file)
        self.index_list = np.array(index_list)
        self.target_labels = np.array([1 if t == 'target' else 0 for t in target_list])
        self.n_features = n_features
        self.batch_size = max_batch_size
        self.multiple = multiple
        self.frames_dict = make_dict(test_list[:, 0], np.array(test_list[:, -1], dtype=int))

        number_list = list(range(len(label_list)))
        labels_to_index_dict = make_labels_to_index_dict(number_list, label_list)

        self.batch_pointer = 0
        self.main_labels = []
        self.batch_splits = []
        for label, values in labels_to_index_dict.items():
            n_batches = int(len(values) / self.batch_size) + (0 if len(values) % self.batch_size == 0 else 1)
            self.main_labels = self.main_labels + [label] * n_batches
            frames = np.array([self.frames_dict[self.index_list[val]] for val in values], dtype=int)
            idx = np.argsort(frames)
            self.batch_splits = self.batch_splits + np.array_split(np.array(values)[idx], n_batches)
        self.n_batches = len(self.main_labels)

        self.feats_dict = spaced_file_to_dict(join_path(save_loc, FEATS_SCP_FILE))
        self.spk_emb_dict = spaced_file_to_dict(join_path(save_loc, ENROLL_SPK_EMB_SCP_FILE))
        self.tmp_scp_file = join_path(save_loc, TMP_SCP_FILE.format(model_tag))
        self.scores = np.zeros([1, len(label_list)])

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
        index_list = np.array(self.index_list[current_batch_idx])
        main_label = self.main_labels[self.batch_pointer]
        frames = [self.frames_dict[key] for key in index_list]
        self.batch_size = len(current_batch_idx)
        self.increment_pointer()

        max_len = int(min(frames) / self.multiple) * self.multiple
        with open(self.tmp_scp_file, 'w') as f:
            for utt in index_list:
                f.write('{} {}[0:{}]\n'.format(utt, self.feats_dict[utt], max_len - 1))

        _, np_features = read_feats(self.tmp_scp_file, self.n_features)

        with open(self.tmp_scp_file, 'w') as f:
            f.write('{} {}\n'.format(main_label, self.spk_emb_dict[main_label]))

        _, emb_vector = read_vector(self.tmp_scp_file, float)

        out_labels = np.array(self.target_labels[current_batch_idx])
        return np.array(np_features), out_labels, np.array(emb_vector).reshape([1, -1])

    def reset(self):
        self.batch_pointer = 0

    def set_current_batch(self, value):
        self.batch_pointer = value

    def total_batches(self):
        return self.n_batches

    def total_trials(self):
        return len(self.index_list)


class AttentionBatchLoader:
    def __init__(self, args_list, n_features, batch_size, min_frames, max_frames, model_tag,
                 num_repeats=10, multiple=1, shuffle=True, save_loc='../save'):
        self.n_features = n_features
        self.min_frames = min_frames
        self.max_frames = max_frames

        index_list = args_list[:, 0]
        frames = np.array(args_list[:, -1], dtype=int)
        labels = args_list[:, 3]

        half_batch_size = int(batch_size / 2)

        print('{}: Processing train data...'.format(model_tag))
        split_index_list = []
        split_duration_list = []
        split_labels = []
        for (i, f, l) in zip(index_list, frames, labels):
            n = int(f / max_frames)
            n = n if n < num_repeats else num_repeats
            for k in range(n):
                if f > max_frames:
                    idx = np.random.choice(f - max_frames, 1)[0]
                else:
                    idx = 0
                split_index_list.append(i)
                split_duration_list.append(idx)
                split_labels.append(l)

        split_number_list = list(range(len(split_index_list)))

        print('{}: Splitting data into train and dev sets...'.format(model_tag))
        labels_to_index_list = make_labels_to_index_dict(split_number_list, split_labels)

        counter = Counter(split_labels)
        labels = []
        other_labels = []
        for key, value in counter.most_common():
            if value >= half_batch_size:
                labels.append(key)
            else:
                other_labels.append(key)

        labels = np.array(labels)
        other_labels = np.array(other_labels)
        train_labels = np.hstack([labels[:-half_batch_size], other_labels])
        dev_labels = labels[-half_batch_size:]

        train_label_index_dict = split_dict(labels_to_index_list, train_labels)
        dev_label_index_dict = split_dict(labels_to_index_list, dev_labels)

        split_args_list = np.vstack([split_index_list, split_duration_list, split_labels]).T

        print('{}: Preparing Train Batch Loader...'.format(model_tag))
        self.train_loader = LabelBatchLoader(split_args_list, train_label_index_dict, n_features, batch_size, min_frames,
                                             max_frames, model_tag, multiple, shuffle, save_loc)
        print('{}: Preparing Dev Batch Loader...'.format(model_tag))
        self.dev_loader = LabelBatchLoader(split_args_list, dev_label_index_dict, n_features, batch_size, min_frames,
                                           max_frames, model_tag, multiple, False, save_loc)

    def get_batch_size(self):
        return self.train_loader.get_batch_size()

    def get_current_batch(self):
        return self.train_loader.get_current_batch()

    def get_dev_loader(self):
        return self.dev_loader

    def next(self):
        return self.train_loader.next()

    def reset(self):
        self.train_loader.reset()

    def set_current_batch(self, value):
        self.train_loader.set_current_batch(value)

    def total_batches(self):
        return self.train_loader.total_batches()
