# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

import random
import math
import numpy as np
import torch
import torch.nn.functional as F


logger = logging.getLogger("slim")


ntu_pairs = [
    (5, 9), (6, 10), (7, 11), (8, 12),  # Arms
    (13, 17), (14, 18), (15, 19), (16, 20),  # Legs
    (22, 24), (23, 25)  # Hands (Tip/Thumb)
]


ntu_transform_order = np.arange(25)
for i, j in ntu_pairs:
    ntu_transform_order[i - 1] = j - 1
    ntu_transform_order[j - 1] = i - 1


def spatial_flip(data_numpy, p=0.5):
    if random.random() < p:
        data_numpy = data_numpy[:, :, ntu_transform_order, :]
        data_numpy[0] = -data_numpy[0]
    return data_numpy


def random_rot(data_numpy, theta_tilt=0.3, theta_vertical=math.pi, p=0.5):
    if random.random() < p:
        alpha = random.uniform(-theta_tilt, theta_tilt)
        gamma = random.uniform(-theta_tilt, theta_tilt)

        beta = random.uniform(-theta_vertical, theta_vertical)

        Rx = np.array([[1, 0, 0],
                       [0, math.cos(alpha), -math.sin(alpha)],
                       [0, math.sin(alpha), math.cos(alpha)]])

        Ry = np.array([[math.cos(beta), 0, math.sin(beta)],
                       [0, 1, 0],
                       [-math.sin(beta), 0, math.cos(beta)]])

        Rz = np.array([[math.cos(gamma), -math.sin(gamma), 0],
                       [math.sin(gamma), math.cos(gamma), 0],
                       [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))
        data_numpy = data_numpy.transpose(1, 2, 3, 0)  # (T, V, M, 3)
        data_numpy = np.dot(data_numpy, R.T)  # (T, V, M, 3)
        data_numpy = data_numpy.transpose(3, 0, 1, 2)  # (3, T, V, M)
    return data_numpy


def axis_drop(data_numpy, p=0.05):
    if random.random() < p:
        axis = random.randint(0, 2)
        data_numpy[axis, :, :, :] = 0
    return data_numpy


def gaussian_noise(data_numpy, std=0.01, p=0.5):
    if random.random() < p:
        noise = np.random.normal(0, std, data_numpy.shape)
        return data_numpy + noise
    return data_numpy


def random_scale(data_numpy, scale_range=(0.85, 1.15), p=0.5):
    """
    Random Limb Scaling for Skeleton Data
    - data_numpy: (C, T, V, M)
    - scale_range: Scaling factor range (e.g., 0.85 ~ 1.15)
    - p: Probability of augmentation
    """
    if random.random() < p:
        C, T, V, M = data_numpy.shape

        # 1. Define Skeleton Structure (NTU RGB+D 25 joints)
        # 0-based index. -1 indicates no parent (root).
        parents = np.array([-1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11])

        # Define Limb Groups (Bone indices where the key is the child joint)
        limbs = {
            'torso': [1, 20, 2, 3],
            'arm': [4, 5, 6, 7, 21, 22, 8, 9, 10, 11, 23, 24],
            'leg': [12, 13, 14, 15, 16, 17, 18, 19]
        }

        # 2. Joint to Bone
        # data_numpy shape: (C, T, V, M)
        # Calculate bone vectors relative to parents
        bones = np.zeros_like(data_numpy)
        for v in range(1, V):
            bones[:, :, v, :] = data_numpy[:, :, v, :] - data_numpy[:, :, parents[v], :]

        # 3. Generate Random Scales per Limb Group
        # scale_factors shape: (1, 1, V, M) to broadcast over C and T
        scale_factors = np.ones((1, 1, V, M), dtype=data_numpy.dtype)

        for m in range(M):
            # Generate independent scales for diversity (X-Sub key factor)
            s_torso = random.uniform(*scale_range)
            s_arm = random.uniform(*scale_range)
            s_leg = random.uniform(*scale_range)

            # Apply scales to corresponding joints
            scale_factors[0, 0, limbs['torso'], m] = s_torso
            scale_factors[0, 0, limbs['arm'], m] = s_arm
            scale_factors[0, 0, limbs['leg'], m] = s_leg

        # 4. Apply Scaling
        scaled_bones = bones * scale_factors

        # 5. Bone to Joint (Reconstruction)
        # Preserve original root trajectory
        data_scaled = np.zeros_like(data_numpy)
        root_idx = 0
        data_scaled[:, :, root_idx, :] = data_numpy[:, :, root_idx, :]  # Keep Root Pos

        # Forward Kinematics reconstruction
        # Iterate in topological order (safe for NTU)
        for v in range(1, V):
            p_idx = parents[v]
            data_scaled[:, :, v, :] = data_scaled[:, :, p_idx, :] + scaled_bones[:, :, v, :]

        return data_scaled

    return data_numpy.copy()


class DataAugmentationDINO(object):
    def __init__(
            self,
            local_crops_number,
            global_crops_size=64,
            local_crops_size=16,
            split='train',
    ):
        self.global_crops_size = global_crops_size
        self.split = split

        self.global_p_interval = [0.5, 1.0]

        self.local_crops_config = [
            {'size': 32, 'count': 1, 'p': [0.35, 0.7]},  # 1
            {'size': 16, 'count': 2, 'p': [0.15, 0.4]},   # 2
            {'size': 8, 'count': 1, 'p': [0.05, 0.2]},  # 1
        ]

    def _interpolated_crop(self, data_numpy, window, p_interval, valid_length=None, constraint_indices=None):
        C, T, V, M = data_numpy.shape
        thres = max(8, min(16, window))

        if constraint_indices is not None:
            valid_start, valid_end = constraint_indices
            valid_size = valid_end - valid_start
            base_offset = valid_start
        else:
            if valid_length is not None and valid_length > 0:
                if valid_length < thres:
                    valid_size = T
                else:
                    valid_size = valid_length
            else:
                valid_size = T
            base_offset = 0

        if valid_size < thres:
            valid_size = T
            base_offset = 0

        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
        bias = np.random.randint(0, valid_size - cropped_length + 1)

        if cropped_length < window:
            inds = np.arange(cropped_length)
        elif window <= cropped_length < 2 * window:
            basic = np.arange(window)
            inds = np.random.choice(window + 1, cropped_length - window, replace=False)
            offset = np.zeros(window + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array([i * cropped_length // window for i in range(window + 1)])
            bsize = np.diff(bids)
            bst = bids[:window]
            offset = np.random.randint(bsize)
            inds = bst + offset

        final_inds = inds + bias + base_offset
        final_inds = np.clip(final_inds, 0, T - 1)

        data = data_numpy[:, final_inds, :, :]
        data = torch.tensor(data, dtype=torch.float)
        data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, len(final_inds))

        if len(final_inds) != window:
            data = F.interpolate(data, size=window, mode='linear', align_corners=False)

        data = data.view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()
        range_info = (final_inds[0], final_inds[-1] + 1)
        return data, range_info

    def _apply_geometric(self, crop):
        if self.split == 'train':
            crop = spatial_flip(crop, p=0.5)
            crop = random_scale(crop, scale_range=(0.85, 1.15), p=0.5)
            crop = random_rot(crop, theta_tilt=0.3, theta_vertical=0.3, p=0.5)  # math.pi
        return crop

    def _apply_global_1_extra(self, crop):
        if self.split == 'train':
            crop = gaussian_noise(crop, std=0.01, p=1.0)
        return crop

    def _apply_global_2_extra(self, crop):
        if self.split == 'train':
            crop = gaussian_noise(crop, std=0.05, p=0.1)
            crop = axis_drop(crop, p=0.2)
        return crop

    def _apply_local_extra(self, crop):
        if self.split == 'train':
            crop = gaussian_noise(crop, std=0.01, p=0.5)
        return crop

    def __call__(self, buffer_seq, valid_frame_num=None):
        output = {}

        global_crops = []
        global_ranges = []

        # Crop 1
        c1, r1 = self._interpolated_crop(buffer_seq, self.global_crops_size, self.global_p_interval, valid_frame_num)
        global_crops.append(torch.from_numpy(self._apply_global_1_extra(self._apply_geometric(c1))).float())
        global_ranges.append(r1)

        # Crop 2
        c2, r2 = self._interpolated_crop(buffer_seq, self.global_crops_size, self.global_p_interval, valid_frame_num)
        global_crops.append(torch.from_numpy(self._apply_global_2_extra(self._apply_geometric(c2))).float())
        global_ranges.append(r2)

        output["global_crops"] = global_crops
        output["global_crops_teacher"] = global_crops

        for config in self.local_crops_config:
            size = config['size']
            count = config['count']
            p_interval = config['p']

            key_name = f"local_crops_{size}"  # e.g., local_crops_32
            output[key_name] = []

            for g_idx in range(2):
                ref_range = global_ranges[g_idx]

                for _ in range(count):
                    crop_base, _ = self._interpolated_crop(
                        buffer_seq,
                        window=size,
                        p_interval=p_interval,
                        constraint_indices=ref_range
                    )

                    crop_geo = self._apply_geometric(crop_base.copy())
                    crop_final = self._apply_local_extra(crop_geo)
                    output[key_name].append(torch.from_numpy(crop_final).float())

        return output


class DownstreamAugmentation(object):
    def __init__(
            self,
            out_size=64,  #TODO: back to 64,
            split='train',
    ):
        self.out_size = out_size
        self.split = split
        logger.info(f"Initialized DownstreamAugmentation (Split: {split}, Size: {out_size})")

    def _train_sampling(self, seq, valid_frame_num=None):
        p_interval = [0.5, 1.0]
        thres = 16
        data, _ = self._valid_crop_resize(seq, valid_frame_num, p_interval, self.out_size, thres)
        return data

    def _test_sampling(self, seq, valid_frame_num=None):
        p_interval = [0.95]
        thres = 64
        data, _ = self._valid_crop_resize(seq, valid_frame_num, p_interval, self.out_size, thres)
        return data

    def _valid_crop_resize(self, data_numpy, valid_frame_num, p_interval, window, thres):
        C, T, V, M = data_numpy.shape

        if valid_frame_num is None or valid_frame_num == 0:
            valid_frame_num = T

        begin = 0
        end = valid_frame_num
        valid_size = end - begin

        if len(p_interval) == 1:
            p = p_interval[0]
            cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
            bias = int((1 - p) * valid_size / 2)

            if cropped_length < window:
                inds = np.arange(cropped_length)
            else:
                bids = np.array([i * cropped_length // window for i in range(window + 1)])
                bst = bids[:window]
                inds = bst

            inds = inds + bias
            data = data_numpy[:, inds, :, :]

        else:
            p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
            cropped_length = np.minimum(np.maximum(int(np.floor(valid_size * p)), thres), valid_size)
            bias = np.random.randint(0, valid_size - cropped_length + 1)

            if cropped_length < window:
                inds = np.arange(cropped_length)
            elif window <= cropped_length < 2 * window:
                basic = np.arange(window)
                inds = np.random.choice(window + 1, cropped_length - window, replace=False)
                offset = np.zeros(window + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * cropped_length // window for i in range(window + 1)])
                bsize = np.diff(bids)
                bst = bids[:window]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + bias
            data = data_numpy[:, inds, :, :]

        data = torch.tensor(data, dtype=torch.float)
        data = data.permute(2, 3, 0, 1).contiguous().view(V * M, C, len(inds))
        index_t = torch.tensor(inds, dtype=torch.float)

        # Interpolate if length mismatch
        if len(inds) != window:
            data = F.interpolate(data, size=window, mode='linear', align_corners=False)
            index_t = F.interpolate(index_t[None, None, :], size=window, mode='linear', align_corners=False).squeeze()

        # Reshape back to (C, T, V, M) -> numpy
        data = data.view(V, M, C, window).permute(2, 3, 0, 1).contiguous().numpy()

        # Index normalization (Optional)
        index_t = 2 * index_t / valid_size - 1

        return data, index_t.numpy()

    def __call__(self, buffer_seq, valid_frame_num=None):
        """
        buffer_seq: Numpy array (C, T, V, M)
        """

        if self.split == 'train':
            crop = self._train_sampling(buffer_seq, valid_frame_num)
        else:
            crop = self._test_sampling(buffer_seq, valid_frame_num)

        if self.split == 'train':
            crop = spatial_flip(crop, p=0.5)
            crop = random_scale(crop, scale_range=(0.85, 1.15), p=0.5)
            crop = random_rot(crop, theta_tilt=0.3, theta_vertical=math.pi, p=0.5)
            crop = gaussian_noise(crop, std=0.01, p=0.5)

        return torch.from_numpy(crop.copy()).float()