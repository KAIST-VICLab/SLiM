# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import random
import numpy as np

# Semantic joint groups of the Kinect-v2 25-joint skeleton (0-based indices):
# trunk, left arm, right arm, left leg, right leg.
kinect_v2_joint_groups = [
    [0, 1, 20, 2, 3],
    [4, 5, 6, 7, 21, 22],
    [8, 9, 10, 11, 23, 24],
    [12, 13, 14, 15],
    [16, 17, 18, 19]
]


class MaskingGenerator:
    """Semantic Tube Masking (STM).

    Samples spatio-temporal "tubes" over the compact token grid (N_T x N_J).
    Each tube covers a contiguous subset of joints taken from one anatomical
    group and a contiguous span of temporal tokens, so masked regions stay
    anatomically coherent instead of being scattered over independent joints.
    """

    def __init__(
        self,
        input_size,
        joint_groups=kinect_v2_joint_groups,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
    ):
        self.joint_groups = joint_groups

        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

    def __repr__(self):
        return "MaskingGenerator(%d, %d -> [%d ~ %d], max = %s)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
        )

    def get_shape(self):
        return self.height, self.width

    def _mask_semantic(self, mask, max_mask_patches):
        """Propose and apply a single tube. Returns the number of newly masked cells."""
        delta = 0
        for _ in range(10):
            # 1. Pick one anatomical group, then a contiguous subset of its joints.
            group_idx = random.randint(0, len(self.joint_groups) - 1)
            full_group_joints = self.joint_groups[group_idx]
            group_len = len(full_group_joints)
            if group_len > 1:
                subset_len = random.randint(1, group_len)
                start_idx = random.randint(0, group_len - subset_len)
                selected_joints = full_group_joints[start_idx: start_idx + subset_len]
            else:
                selected_joints = full_group_joints

            # 2. The joint subset size determines the tube width.
            w = len(selected_joints)
            if w == 0:
                continue

            # 3. Sample a target area and derive the temporal duration from it.
            target_area = random.uniform(self.min_num_patches, max_mask_patches)

            h = int(round(target_area / w))
            h = max(1, min(h, self.height))

            # 4. Uniform temporal start.
            if self.height - h >= 0:
                top = random.randint(0, self.height - h)

                # 5. Accept only if the tube adds new cells without overshooting the budget.
                current_patch_mask = mask[top: top + h, selected_joints]
                num_masked = current_patch_mask.sum()

                if 0 < h * w - num_masked <= max_mask_patches:
                    mask[top: top + h, selected_joints] = 1
                    delta = (h * w) - num_masked

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask_semantic(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
