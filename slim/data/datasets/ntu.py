# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import os
from enum import Enum
import logging
import random
from torch.utils.data import Dataset

logger = logging.getLogger("slim")


class NTU(Dataset):
    """Unlabelled NTU clips for pre-training.

    Expects a .npz with `x_train` / `x_test` of shape (N, 300, 150) — 300 frames of
    2 people x 25 joints x 3 coordinates — and one-hot `y_train` / `y_test`.
    Pre-training uses `split=TRAIN` of the .npz of the protocol being targeted, so the
    protocol's test clips are never seen; labels are loaded but unused.
    """

    class Split(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self,
                 root,
                 split,
                 transform=None,
                 min_buffer=150,
                 **kwargs):

        self.root = root
        self.split = split
        self.transform = transform
        self.min_buffer = min_buffer
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.root):
            raise RuntimeError(f"Dataset not found: {self.root}")

        logger.info(f"Loading skeleton data from {self.root} ...")
        npz_data = np.load(self.root, mmap_mode='r')

        if self.split == NTU.Split.TRAIN:
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
        elif self.split == NTU.Split.TEST:
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)  # N, C, T, V, M
        logger.info(f"Loaded {len(self.data)} samples. Shape: {self.data.shape}")

    def _prepare_buffer(self, data_numpy):
        valid = (data_numpy.sum(axis=(0, 2, 3)) != 0)
        if valid.sum() > 0:
            seq = data_numpy[:, valid, :, :]
        else:
            seq = data_numpy

        C, T, V, M = seq.shape
        if T < self.min_buffer:
            blocks = [seq]
            curr = T
            while curr < self.min_buffer:
                to_cat = seq[:, ::-1, :, :]
                blocks.append(to_cat)
                curr += to_cat.shape[1]
                seq = to_cat
            seq = np.concatenate(blocks, axis=1)

        return seq, valid.sum()

    def _select_random_person(self, seq):
        """
        Input: (C, T, V, M)
        Output: (C, T, V) - Randomly selected single person
        """
        C, T, V, M = seq.shape
        valid_indices = []
        for m in range(M):
            if np.sum(np.abs(seq[..., m])) > 1e-5:
                valid_indices.append(m)

        if len(valid_indices) == 0:
            person_idx = 0
        elif len(valid_indices) == 1:
            person_idx = valid_indices[0]
        else:
            person_idx = random.choice(valid_indices)

        return seq[..., [person_idx]]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        raw_data = np.array(self.data[index])
        buffer_seq, valid_frame_num = self._prepare_buffer(raw_data)  # (C, T, V, M)
        single_seq = self._select_random_person(buffer_seq)
        origin = single_seq[:, 0, 1, 0]
        single_seq = single_seq - origin[:, None, None, None]

        if self.transform is not None:
            output = self.transform(single_seq, valid_frame_num)
        else:
            tensor_seq = torch.from_numpy(single_seq).float()
            output = {
                "global_crops": [tensor_seq, tensor_seq],
                "local_crops": []
            }

        target = self.label[index]
        output["target"] = target
        return output, target


class NTU_Classification(Dataset):
    """Labelled NTU split used by linear probing, fine-tuning and k-NN retrieval.

    `sample_ratio < 1.0` keeps a class-balanced random subset of the split, which is
    how the semi-supervised (1 % / 10 %) protocol is built. Selection is seeded, so
    the subset is identical across ranks and runs.
    """

    class Split(Enum):
        TRAIN = "train"
        TEST = "test"

    def __init__(self,
                 root_path,
                 split='train',
                 num_classes=60,
                 benchmark='xsub',
                 min_buffer=150,
                 transform=None,
                 sample_ratio=1.0,
                 seed=42):

        self.root = root_path
        self.split = split
        self.num_classes = num_classes
        self.benchmark = benchmark
        self.min_buffer = min_buffer
        self.transform = transform
        self.sample_ratio = sample_ratio
        self.seed = seed

        if num_classes == 60:
            if benchmark == 'xsub':
                self.file_path = os.path.join(self.root, "NTU60_XSub.npz")
            elif benchmark == 'xview':
                self.file_path = os.path.join(self.root, "NTU60_XView.npz")
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")
        else:
            if benchmark == 'xsub':
                self.file_path = os.path.join(self.root, "NTU120_CS.npz")
            elif benchmark == 'xview':
                self.file_path = os.path.join(self.root, "NTU120_CV.npz")
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")

        self.load_data()

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise RuntimeError(f"Dataset not found: {self.file_path}")

        logger.info(f"Loading skeleton data from {self.file_path} ...")
        npz_data = np.load(self.file_path, mmap_mode='r')

        if self.split == 'train' or self.split == self.Split.TRAIN:
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
        elif self.split == 'test' or self.split == self.Split.TEST:
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        logger.info(f"Loaded {len(self.data)} samples for {self.split}. Shape: {self.data.shape}")

        # Class-balanced subsampling for the semi-supervised protocol.
        if self.sample_ratio < 1.0:
            np.random.seed(self.seed)
            selected_indices = []
            for lbl in np.unique(self.label):
                lbl_indices = np.where(self.label == lbl)[0]
                np.random.shuffle(lbl_indices)
                num_keep = max(1, int(len(lbl_indices) * self.sample_ratio))  # keep >= 1 per class
                selected_indices.extend(lbl_indices[:num_keep])

            selected_indices = np.array(selected_indices)
            np.random.shuffle(selected_indices)

            self.data = self.data[selected_indices]
            self.label = self.label[selected_indices]
            logger.info(f"Subsampled to {self.sample_ratio * 100:g}% per class -> {len(self.label)} samples.")

    def _prepare_buffer(self, data_numpy):
        valid = (data_numpy.sum(axis=(0, 2, 3)) != 0)
        if valid.sum() > 0:
            seq = data_numpy[:, valid, :, :]
        else:
            seq = data_numpy

        C, T, V, M = seq.shape
        if T < self.min_buffer:
            blocks = [seq]
            curr = T
            while curr < self.min_buffer:
                to_cat = seq[:, ::-1, :, :]
                blocks.append(to_cat)
                curr += to_cat.shape[1]
                seq = to_cat
            seq = np.concatenate(blocks, axis=1)

        return seq, valid.sum()

    def __len__(self):
        return len(self.label)

    def get_targets(self):
        return self.label

    def __getitem__(self, index):
        raw_data = np.array(self.data[index])
        buffer_seq, valid_frame_num = self._prepare_buffer(raw_data)
        body_sum = np.sum(np.abs(buffer_seq), axis=(0, 1, 2))
        valid_person_mask = (body_sum > 1e-5)

        if self.transform is not None:
            output = self.transform(buffer_seq, valid_frame_num)
        else:
            output = torch.from_numpy(buffer_seq[:, :self.min_buffer, :, :]).float()

        target = self.label[index]
        return output, torch.from_numpy(valid_person_mask), target