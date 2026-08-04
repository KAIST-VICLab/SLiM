# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import torch
from torch.nn.functional import one_hot, softmax
import torch.distributed as dist
from tqdm import tqdm

import slim.distributed as distributed
from slim.data import SamplerType, make_data_loader
from slim.eval.setup import get_args_parser as get_setup_args_parser
from slim.eval.setup import setup_and_build_model
from slim.eval.utils import ModelWithIntermediateLayers

# NTU Dataset Imports
from slim.data.datasets.ntu import NTU_Classification
from slim.data.augmentations import DownstreamAugmentation

logger = logging.getLogger("slim")


def get_args_parser(
        description: Optional[str] = None,
        parents: Optional[List[argparse.ArgumentParser]] = None,
        add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    # NTU Dataset Arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/ntu",
        help="Path to NTU dataset (.npz files)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=60,
        help="Number of classes",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="xsub",
        choices=["xsub", "xview"],
        help="NTU Benchmark (xsub or xview)",
    )

    # k-NN Arguments
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. Include 1 for standard benchmarking.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower but useful to avoid OOM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )

    # Kept for CLI compatibility.
    parser.add_argument("--train-dataset", dest="train_dataset_str", type=str, default="NTU:split=TRAIN")
    parser.add_argument("--val-dataset", dest="val_dataset_str", type=str, default="NTU:split=VAL")
    parser.add_argument("--n-per-class-list", nargs="+", type=int, help="Number to take per class")
    parser.add_argument("--n-tries", type=int, help="Number of tries")

    parser.set_defaults(
        nb_knn=[1, 10, 20, 100, 200],  # k=1 is the retrieval protocol
        temperature=0.07,
        batch_size=256,
        n_per_class_list=[-1],
        n_tries=1,
    )
    return parser


# --- Distributed Helper ---
def gather_and_concat(tensor):
    if not distributed.is_enabled():
        return tensor

    world_size = distributed.get_global_size()
    # Gather tensors from every rank.
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    # Merge.
    output = torch.cat(tensors_gather, dim=0)
    return output


# --- Feature Extraction for NTU (CLS + Patch Token) ---
@torch.no_grad()
def extract_features_ntu(model, data_loader, gather_on_cpu=False):
    model.eval()
    features_list = []
    labels_list = []

    # Use only the last block's output.
    n_last_blocks = 1
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float16)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)

    for data, masks, labels in data_loader:
        data = data.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # Forward pass
        # outputs[-1] returns (patch_tokens, cls_token)
        outputs = feature_model(data)
        patch_tokens, cls_token = outputs[-1]

        # --- Reference Code Logic Implementation ---
        B, C, T, V, M = data.shape
        D_cls = cls_token.shape[-1]

        # 1. Mask Handling
        valid_mask_expanded = masks.float().view(B, M, 1)  # (B, M, 1)

        # 2. CLS Token Processing (Sum over M actors)
        cls_token = cls_token.view(B, M, D_cls)
        masked_cls = cls_token * valid_mask_expanded
        #cls_token_agg = torch.sum(masked_cls, dim=1)  # (B, D_cls)
        cls_token_agg = masked_cls.view(B, -1)

        # 3. Patch Token Processing (Average Pool -> Concat Actors)
        if patch_tokens is not None:
            N_patch, D_patch = patch_tokens.shape[1], patch_tokens.shape[2]
            patch_tokens = patch_tokens.view(B, M, N_patch, D_patch)

            # Mask broadcasting: (B, M, 1, 1) to multiply (B, M, N_patch, D_patch)
            patch_mask = valid_mask_expanded.unsqueeze(2)
            patch_masked = patch_tokens * patch_mask  # (B, M, N_patch, D_patch)

            # Average-pool patch tokens -> (B, M, D_patch).
            patch_avg = torch.mean(patch_masked, dim=2)

            # Flatten the person axis -> (B, M * D_patch).
            patch_tokens_agg = patch_avg.view(B, -1)

        else:
            patch_tokens_agg = None

        # 4. Combine Features (CLS + Patch)
        # Both aggregates are (B, M * D), so they concatenate directly

        features_to_cat = [cls_token_agg]  # (B, M * D_cls)

        if patch_tokens_agg is not None:
            features_to_cat.append(patch_tokens_agg)  # (B, M * D_patch)

        # -> (B, M * (D_cls + D_patch)).
        final_feature = torch.cat(features_to_cat, dim=-1)

        # 5. Normalize
        final_feature = torch.nn.functional.normalize(final_feature, dim=1, p=2)

        if gather_on_cpu:
            features_list.append(final_feature.cpu())
            labels_list.append(labels.cpu())
        else:
            features_list.append(final_feature)
            labels_list.append(labels)

    # Concatenate & Gather
    if gather_on_cpu:
        all_features = torch.cat(features_list, dim=0)
        all_labels = torch.cat(labels_list, dim=0)
    else:
        local_features = torch.cat(features_list, dim=0)
        local_labels = torch.cat(labels_list, dim=0)
        all_features = gather_and_concat(local_features)
        all_labels = gather_and_concat(local_labels)

    return all_features, all_labels


# --- k-NN Module ---
class KnnModule(torch.nn.Module):
    def __init__(self, train_features, train_labels, nb_knn, T, device, num_classes=1000):
        super().__init__()
        self.global_rank = distributed.get_global_rank()
        self.global_size = distributed.get_global_size()
        self.device = device
        self.train_features_rank_T = train_features.chunk(self.global_size)[self.global_rank].T.to(self.device)
        self.candidates = train_labels.chunk(self.global_size)[self.global_rank].view(1, -1).to(self.device)
        self.nb_knn = nb_knn
        self.max_k = max(self.nb_knn)
        self.T = T
        self.num_classes = num_classes

    def _get_knn_sims_and_labels(self, similarity, train_labels):
        topk_sims, indices = similarity.topk(self.max_k, largest=True, sorted=True)
        neighbors_labels = torch.gather(train_labels, 1, indices)
        return topk_sims, neighbors_labels

    def _similarity_for_rank(self, features_rank, source_rank):
        broadcast_shape = torch.tensor(features_rank.shape).to(self.device)
        torch.distributed.broadcast(broadcast_shape, source_rank)
        broadcasted = features_rank
        if self.global_rank != source_rank:
            broadcasted = torch.zeros(*broadcast_shape, dtype=features_rank.dtype, device=self.device)
        torch.distributed.broadcast(broadcasted, source_rank)
        similarity_rank = torch.mm(broadcasted, self.train_features_rank_T)
        candidate_labels = self.candidates.expand(len(similarity_rank), -1)
        return self._get_knn_sims_and_labels(similarity_rank, candidate_labels)

    def _gather_all_knn_for_rank(self, topk_sims, neighbors_labels, target_rank):
        topk_sims_rank = retrieved_rank = None
        if self.global_rank == target_rank:
            topk_sims_rank = [torch.zeros_like(topk_sims) for _ in range(self.global_size)]
            retrieved_rank = [torch.zeros_like(neighbors_labels) for _ in range(self.global_size)]
        torch.distributed.gather(topk_sims, topk_sims_rank, dst=target_rank)
        torch.distributed.gather(neighbors_labels, retrieved_rank, dst=target_rank)
        if self.global_rank == target_rank:
            topk_sims_rank = torch.cat(topk_sims_rank, dim=1)
            retrieved_rank = torch.cat(retrieved_rank, dim=1)
            results = self._get_knn_sims_and_labels(topk_sims_rank, retrieved_rank)
            return results
        return None

    def compute_neighbors(self, features_rank):
        for rank in range(self.global_size):
            topk_sims, neighbors_labels = self._similarity_for_rank(features_rank, rank)
            results = self._gather_all_knn_for_rank(topk_sims, neighbors_labels, rank)
            if results is not None:
                topk_sims_rank, neighbors_labels_rank = results
        return topk_sims_rank, neighbors_labels_rank

    def forward(self, features_rank):
        assert all(k <= self.max_k for k in self.nb_knn)
        topk_sims, neighbors_labels = self.compute_neighbors(features_rank)
        batch_size = neighbors_labels.shape[0]
        topk_sims_transform = softmax(topk_sims / self.T, 1)
        matmul = torch.mul(
            one_hot(neighbors_labels.long(), num_classes=self.num_classes),
            topk_sims_transform.view(batch_size, -1, 1),
        )
        probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in self.nb_knn}
        return probas_for_k


# --- Main Evaluation Function ---
def eval_knn_ntu(
        model,
        output_dir,
        data_path,
        num_classes,
        benchmark,
        nb_knn,
        temperature,
        batch_size,
        num_workers,
        gather_on_cpu,
        n_per_class_list=[-1],
        n_tries=1,
):
    # Retrieval is non-parametric: both the gallery and the queries use the
    # deterministic test-time transform, so no random augmentation enters the database.
    train_transform = DownstreamAugmentation(out_size=64, split='test')
    train_dataset = NTU_Classification(
        root_path=data_path,
        split='train',
        num_classes=num_classes,
        benchmark=benchmark,
        min_buffer=150,
        transform=train_transform
    )

    test_transform = DownstreamAugmentation(out_size=64, split='test')
    val_dataset = NTU_Classification(
        root_path=data_path,
        split='test',
        num_classes=num_classes,
        benchmark=benchmark,
        min_buffer=150,
        transform=test_transform
    )

    # DataLoaders
    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
    )

    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
    )

    # 1. Extract Features (Backbone)
    # Retrieval features concatenate the CLS token and the pooled patch tokens.
    logger.info("Extracting features (CLS + Patch) for train set...")
    train_features, train_labels = extract_features_ntu(model, train_loader, gather_on_cpu)
    logger.info(f"Train features shape: {train_features.shape}")

    logger.info("Extracting features (CLS + Patch) for val set...")
    val_features, val_labels = extract_features_ntu(model, val_loader, gather_on_cpu)

    device = torch.cuda.current_device()
    train_labels = train_labels.long()
    val_labels = val_labels.long()

    # 2. k-NN Execution
    logger.info("Using Distributed k-NN (Multi GPU Mode)")
    knn_module = KnnModule(train_features, train_labels, nb_knn, temperature, device, num_classes)

    logger.info("Running k-NN inference...")
    results = {}
    for k in nb_knn:
        results[f"{k}-NN"] = {"correct": 0, "total": 0}

    # Validation Loop (Batch processing)
    num_val = val_features.shape[0]
    val_batch_size = batch_size

    for i in tqdm(range(0, num_val, val_batch_size)):
        end = min(i + val_batch_size, num_val)
        batch_feats = val_features[i:end].to(device)
        batch_labels = val_labels[i:end].to(device)

        probas = knn_module(batch_feats)  # {k: scores}

        for k, scores in probas.items():
            preds = scores.argmax(dim=1)
            correct = (preds == batch_labels).sum().item()
            results[f"{k}-NN"]["correct"] += correct
            results[f"{k}-NN"]["total"] += (end - i)

    # Aggregate & Log
    final_results = {}
    for k_name, res in results.items():
        total_correct = torch.tensor(res["correct"]).cuda()
        total_count = torch.tensor(res["total"]).cuda()

        if distributed.is_enabled():
            dist.all_reduce(total_correct)
            dist.all_reduce(total_count)

        acc = (total_correct / total_count).item() * 100.0
        final_results[k_name] = acc
        logger.info(f"{k_name} Accuracy: {acc:.2f}%")

    if distributed.is_main_process():
        with open(os.path.join(output_dir, "results_eval_knn.json"), "a") as f:
            f.write(json.dumps(final_results) + "\n")

    return final_results


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    eval_knn_ntu(
        model=model,
        output_dir=args.output_dir,
        data_path=args.data_path,
        num_classes=args.num_classes,
        benchmark=args.benchmark,
        nb_knn=args.nb_knn,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_workers=4,  # args.num_workers,
        gather_on_cpu=args.gather_on_cpu,
        n_per_class_list=args.n_per_class_list,
        n_tries=args.n_tries,
    )
    return 0


if __name__ == "__main__":
    description = "SLiM k-NN retrieval evaluation"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))