"""Sanity-check released SLiM checkpoints.

For every encoder it builds the model from the evaluation config, loads the weights,
and runs one forward pass on a dummy clip; it fails if any encoder weight is missing.
Trained linear probes under checkpoints/linear/ are checked for structural consistency
with that encoder. Reports sha256 and parameter counts.

Usage:
    python tools/verify_checkpoints.py                     # all of checkpoints/*.pth
    python tools/verify_checkpoints.py checkpoints/x.pth   # a specific file
"""

import argparse
import glob
import hashlib
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slim.configs import slim_default_config           # noqa: E402
from slim.models import build_model_from_cfg           # noqa: E402
from slim.utils.utils import load_pretrained_weights   # noqa: E402

EVAL_CFG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "slim", "configs", "eval", "slim_eval.yaml")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path, cfg, device):
    model, embed_dim = build_model_from_cfg(cfg, only_teacher=True)
    load_pretrained_weights(model, path, "teacher")   # raises if encoder weights are missing
    model.eval().to(device)

    T, V = cfg.crops.global_crops_size
    x = torch.zeros(2, cfg.student.in_chans, T, V, 1, device=device)  # (B, C, T, V, M)
    with torch.no_grad():
        feats = model.get_intermediate_layers(x, 4, return_class_token=True)

    n_params = sum(p.numel() for p in model.parameters())
    patch_tokens, cls_token = feats[-1]
    n_t = T // cfg.student.patch_size[0]
    n_v = V // cfg.student.patch_size[1]
    assert patch_tokens.shape[1] == n_t * n_v, (patch_tokens.shape, n_t * n_v)
    assert cls_token.shape[-1] == embed_dim
    assert torch.isfinite(cls_token).all() and torch.isfinite(patch_tokens).all()

    print(f"  OK   {os.path.basename(path):<26} {n_params / 1e6:.2f}M params  "
          f"tokens {n_t}x{n_v}={patch_tokens.shape[1]}  dim {embed_dim}")
    print(f"       sha256 {sha256(path)}")
    ckpt = torch.load(path, map_location="cpu")
    if "note" in ckpt:
        print(f"       note   {ckpt['note']}")


def verify_classifier(path, embed_dim):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    names = sorted({k.split(".")[1] for k in state})
    if not names:
        raise SystemExit(f"{path}: no classifiers found")

    for name in names:  # feature width must match the encoder these probes belong to
        fields = name.split("_")
        n_blocks, avgpool = int(fields[1]), fields[4] == "True"
        expected = embed_dim * (n_blocks + (1 if avgpool else 0))  # CLS per block (+ pooled patches)
        got = state[f"classifiers_dict.{name}.linear.weight"].shape[1]
        assert got == expected, f"{path}: {name} expects {expected} features, checkpoint has {got}"

    n_classes = state[f"classifiers_dict.{names[0]}.linear.weight"].shape[0]
    print(f"  OK   {os.path.basename(path):<26} {len(names)} classifiers  {n_classes} classes")
    print(f"       sha256 {sha256(path)}")
    if "note" in ckpt:
        print(f"       note   {ckpt['note']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="*", help="checkpoint paths (default: checkpoints/*.pth)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = a.checkpoints or sorted(glob.glob(os.path.join(root, "checkpoints", "*.pth")))
    if not paths:
        raise SystemExit("no checkpoints found")

    cfg = OmegaConf.merge(OmegaConf.create(slim_default_config), OmegaConf.load(EVAL_CFG))
    print(f"encoder config: {EVAL_CFG}")
    for path in paths:
        verify(path, cfg, a.device)

    classifiers = sorted(glob.glob(os.path.join(root, "checkpoints", "linear", "*.pth")))
    if classifiers and not a.checkpoints:
        print("\ntrained linear probes:")
        for path in classifiers:
            verify_classifier(path, embed_dim=256)
    print(f"\n{len(paths) + len(classifiers)} checkpoint(s) verified.")


if __name__ == "__main__":
    main()
