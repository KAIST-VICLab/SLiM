"""Export the encoder from a SLiM pre-training checkpoint.

A training checkpoint (`<output_dir>/eval/training_<iter>/teacher_checkpoint.pth`)
carries the EMA teacher encoder plus the two 65,536-prototype heads. Only the
encoder survives transfer, so the released weights keep the backbone alone
(6.37 M parameters, ~25 MB instead of ~200 MB).

The exported file keeps the `{"teacher": {"backbone.*": ...}}` layout, so it is a
drop-in replacement wherever a full teacher checkpoint is accepted.

Usage:
    python tools/export_encoder.py <teacher_checkpoint.pth> <out.pth> [--note "..."]
"""

import argparse
import hashlib
import json

import torch


def export(src, dst, note=""):
    ckpt = torch.load(src, map_location="cpu")
    state = ckpt["teacher"] if "teacher" in ckpt else ckpt

    backbone = {k: v for k, v in state.items() if k.startswith("backbone.")}
    if not backbone:
        raise SystemExit(f"no 'backbone.*' weights found in {src}")

    dropped = sorted({k.split(".")[0] for k in state if not k.startswith("backbone.")})
    n_params = sum(v.numel() for v in backbone.values())

    payload = {"teacher": backbone}
    if note:
        payload["note"] = note
    torch.save(payload, dst)

    sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    print(json.dumps({
        "src": src,
        "dst": dst,
        "tensors": len(backbone),
        "parameters": n_params,
        "dropped_modules": dropped,
        "sha256": sha,
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="path to teacher_checkpoint.pth")
    p.add_argument("dst", help="output path for the encoder-only checkpoint")
    p.add_argument("--note", default="", help="free-text provenance note stored in the file")
    a = p.parse_args()
    export(a.src, a.dst, a.note)
