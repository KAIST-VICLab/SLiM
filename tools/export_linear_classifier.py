"""Export a trained linear-probe classifier grid for release.

`<linear_eval_dir>/model_final.pth` stores the classifiers together with the SGD
optimizer and LR-scheduler state. Only the classifier weights are needed to
reproduce a reported accuracy, so this strips the rest (roughly halving the file).

Usage:
    python tools/export_linear_classifier.py <model_final.pth> <out.pth> [--note "..."]
"""

import argparse
import hashlib
import json

import torch


def export(src, dst, note=""):
    ckpt = torch.load(src, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    names = sorted({k.split(".")[1] for k in state})
    if not names:
        raise SystemExit(f"no classifiers found in {src}")

    payload = {"model": state, "iteration": ckpt.get("iteration", 0)}
    if note:
        payload["note"] = note
    torch.save(payload, dst)

    sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    print(json.dumps({
        "src": src,
        "dst": dst,
        "classifiers": len(names),
        "parameters": sum(v.numel() for v in state.values()),
        "dropped": [k for k in ckpt if k not in ("model", "iteration")],
        "sha256": sha,
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", help="path to a linear-eval model_final.pth")
    p.add_argument("dst", help="output path for the classifier-only checkpoint")
    p.add_argument("--note", default="", help="free-text provenance note stored in the file")
    a = p.parse_args()
    export(a.src, a.dst, a.note)
