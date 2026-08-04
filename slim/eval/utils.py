# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection

from slim.logging import MetricLogger


logger = logging.getLogger("slim")


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


@torch.inference_mode()
def evaluate(
        model: nn.Module,
        data_loader,
        postprocessors: Dict[str, nn.Module],
        metrics: Dict[str, MetricCollection],
        device: torch.device,
        criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    # Loader yields (data, valid-person mask, label).
    for samples, masks, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)


        B, C, T, V, M = samples.shape
        features_list = model(samples)
        aggregated_features = []

        # Aggregate over people: sum over valid persons, padded persons masked out.
        valid_mask_expanded = masks.float().view(B, M, 1)

        for (patch_tokens, cls_token) in features_list:
            D_cls = cls_token.shape[-1]
            cls_token = cls_token.view(B, M, D_cls)
            masked_cls = cls_token * valid_mask_expanded
            cls_sum = torch.sum(masked_cls, dim=1)  # (B, D)

            cls_token_agg = cls_sum

            # --- Patch Token Aggregation (Optional) ---
            if patch_tokens is not None:
                N_patch, D_patch = patch_tokens.shape[1], patch_tokens.shape[2]
                patch_tokens = patch_tokens.view(B, M, N_patch, D_patch)
                patch_masked = patch_tokens * valid_mask_expanded.unsqueeze(2)
                patch_sum = torch.sum(patch_masked, dim=1)
                patch_tokens_agg = patch_sum
            else:
                patch_tokens_agg = None

            aggregated_features.append((patch_tokens_agg, cls_token_agg))

        outputs = tuple(aggregated_features)

        first_key = list(postprocessors.keys())[0]
        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)
            if criterion is not None and k == first_key:
                preds = metric_inputs['preds']  # Logits
                loss = criterion(preds, targets)
                metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats
