# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(
        samples_list,
        mask_ratio_tuple,        # STM ratio range for global views, e.g. (0.5, 0.9)
        local_mask_ratio_tuple,  # STM ratio range for local views
        local_masking,           # whether local views are masked at all
        mask_probability,        # fraction of global views that get masked
        dtype,
        n_tokens=None,
        n_people=None,
        mask_generator=None,        # STM generator for the global grid
        local_mask_generator=None,  # per-resolution STM generators, {'local_crops_32': gen, ...}
):
    """Stack crops and generate Semantic Tube Masks.

    Global views: a `mask_probability` fraction is masked; only those contribute to MFP.
    Local views: all of them are masked when `local_masking` is set.
    """

    collated_output = {}

    # 1. Global crops.
    n_global_crops = len(samples_list[0][0]["global_crops"])

    collated_global_crops = torch.stack([
        s[0]["global_crops"][i]
        for i in range(n_global_crops)
        for s in samples_list
    ])

    collated_output["collated_global_crops"] = collated_global_crops.to(dtype)

    # 2. Global masks. Ratios are drawn from a batch-stratified partition of
    #    `mask_ratio_tuple`, so the batch covers the range roughly uniformly.
    B_global = len(collated_global_crops)
    N_global = n_tokens
    n_samples_masked = int(B_global * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []

    for i in range(0, n_samples_masked):  # masked views
        prob_min, prob_max = probs[i], probs[i + 1]
        n_masking = int(N_global * random.uniform(prob_min, prob_max))
        masks_list.append(torch.BoolTensor(mask_generator(n_masking)))
        upperbound += int(N_global * prob_max)

    for i in range(n_samples_masked, B_global):  # unmasked views
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)  # decouple mask assignment from view order
    collated_masks = torch.stack(masks_list).flatten(1)

    # Metadata consumed by the MFP loss.
    collated_output["collated_masks"] = collated_masks
    collated_output["mask_indices_list"] = collated_masks.flatten().nonzero().flatten()
    collated_output["masks_weight"] = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    collated_output["n_masked_patches"] = torch.full((1,), fill_value=collated_output["mask_indices_list"].shape[0], dtype=torch.long)
    collated_output["upperbound"] = upperbound

    # 3. Local crops and their masks, one group per temporal resolution
    #    ('local_crops_32' -> 'local_masks_32', ...).
    sample_keys = samples_list[0][0].keys()
    local_crop_keys = [k for k in sample_keys if "local_crops_" in k]

    for key_crop in local_crop_keys:
        key_mask = key_crop.replace("crops", "masks")

        n_crops = len(samples_list[0][0][key_crop])
        crops_stack = torch.stack([
            s[0][key_crop][i]
            for i in range(n_crops)
            for s in samples_list
        ])
        collated_output[key_crop] = crops_stack.to(dtype)

        if local_masking:
            if local_mask_generator is None or key_crop not in local_mask_generator:
                raise ValueError(f"Generator for {key_crop} is missing in local_mask_generator dict")

            current_generator = local_mask_generator[key_crop]
            local_H, local_W = current_generator.get_shape()
            N_local = local_H * local_W

            local_masks_list = []
            total_local_samples = len(crops_stack)  # B * n_crops

            probs = torch.linspace(local_mask_ratio_tuple[0], local_mask_ratio_tuple[1], total_local_samples + 1)
            for i in range(total_local_samples):
                prob_min, prob_max = probs[i], probs[i + 1]
                n_masking = int(N_local * random.uniform(prob_min, prob_max))
                mask = current_generator(n_masking)
                local_masks_list.append(torch.BoolTensor(mask))

            masks_stack = torch.stack(local_masks_list).flatten(1)
            collated_output[key_mask] = masks_stack

        else:
            N_local = crops_stack.shape[2] * crops_stack.shape[3]  # T * V
            total_local_samples = len(crops_stack)
            masks_stack = torch.zeros((total_local_samples, N_local), dtype=torch.bool)
            collated_output[key_mask] = masks_stack

    return collated_output
