# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""HuggingFace Hub integration for the SLiM encoder.

    from slim.hub import SLiMEncoder
    model = SLiMEncoder.from_pretrained("JeonghyeokDo/SLiM", subfolder="ntu60_xsub").eval().cuda()

`from_pretrained` takes a local directory holding `config.json` and `model.safetensors`, or a
HuggingFace Hub repository id. Requires `pip install huggingface_hub safetensors`; the rest of
the repository does not.
"""

import json
import os

import torch

from slim.models.vision_transformer import DinoVisionTransformer, vit_tiny

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"

# Architecture of the released SLiM encoder. Mirrors slim/configs/eval/slim_eval.yaml.
DEFAULT_CONFIG = {
    "arch": "vit_tiny",
    "img_size": [64, 25],
    "patch_size": [8, 1],
    "in_chans": 3,
    "num_register_tokens": 4,
    "ffn_layer": "swiglufused",
    "block_chunks": 0,
    "qkv_bias": True,
    "proj_bias": True,
    "ffn_bias": True,
    "init_values": 0.0,
}


class SLiMEncoder(DinoVisionTransformer):
    """SLiM encoder with `from_pretrained` / `save_pretrained`.

    Input is a batch of skeleton clips shaped `(B, 3, T, V, M)` — coordinates, frames,
    joints, people. Use `get_intermediate_layers(x, n, return_class_token=True)` for
    downstream features, the same call the evaluation code uses.
    """

    @classmethod
    def from_config(cls, config):
        cfg = {**DEFAULT_CONFIG, **config}
        if cfg.pop("arch", "vit_tiny") != "vit_tiny":
            raise ValueError("only the vit_tiny SLiM encoder is released")
        cfg["img_size"] = tuple(cfg["img_size"])
        cfg["patch_size"] = tuple(cfg["patch_size"])
        model = vit_tiny(**{k: v for k, v in cfg.items() if k != "img_size"}, img_size=cfg["img_size"])
        model.__class__ = cls
        return model

    @classmethod
    def from_pretrained(cls, repo_id_or_path, subfolder=None, revision=None, cache_dir=None, token=None):
        """The repo root holds the released encoder; `subfolder` selects a nested one."""
        if os.path.isdir(repo_id_or_path):
            base = os.path.join(repo_id_or_path, subfolder) if subfolder else repo_id_or_path
            config_path = os.path.join(base, CONFIG_NAME)
            weights_path = os.path.join(base, WEIGHTS_NAME)
        else:
            from huggingface_hub import hf_hub_download

            kwargs = dict(repo_id=repo_id_or_path, subfolder=subfolder, revision=revision,
                          cache_dir=cache_dir, token=token)
            config_path = hf_hub_download(filename=CONFIG_NAME, **kwargs)
            weights_path = hf_hub_download(filename=WEIGHTS_NAME, **kwargs)

        with open(config_path) as f:
            config = json.load(f)
        model = cls.from_config(config)

        from safetensors.torch import load_file

        missing, unexpected = model.load_state_dict(load_file(weights_path), strict=False)
        if missing:
            raise RuntimeError(f"{repo_id_or_path} is missing encoder weights: {missing[:5]}")
        if unexpected:
            raise RuntimeError(f"{repo_id_or_path} has unexpected weights: {unexpected[:5]}")
        return model

    def save_pretrained(self, save_directory, config=None):
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, CONFIG_NAME), "w") as f:
            json.dump({**DEFAULT_CONFIG, **(config or {})}, f, indent=2)
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, os.path.join(save_directory, WEIGHTS_NAME))


def load_encoder_from_pth(path):
    """Build the encoder from a `.pth` checkpoint in the `tools/export_encoder.py` format."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["teacher"] if "teacher" in ckpt else ckpt
    state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
    model = SLiMEncoder.from_config({})
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{path} is missing encoder weights: {missing[:5]}")
    return model
