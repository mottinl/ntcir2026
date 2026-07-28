#!/usr/bin/env python3
# One-time offline conversion of SpursgoZmy/table-llava-v1.5-7b (original
# LLaVA repo format, config.json says architectures=["LlavaLlamaForCausalLM"])
# into a HF-native LlavaForConditionalGeneration checkpoint that
# advisor_table_llava.py can load directly.
#
# Adapted from transformers' own
# src/transformers/models/llava/convert_llava_weights_to_hf.py (fetched and
# read in full on 2026-07-19), with changes forced by inspecting the actual
# SpursgoZmy/table-llava-v1.5-7b repo contents AND this venv's transformers
# version (5.13.1) directly (not assumed):
#   1. That repo ships NEITHER `model_state_dict.bin` NOR `*.safetensors` --
#      only legacy sharded `pytorch_model-0000X-of-0000Y.bin` +
#      `pytorch_model.bin.index.json`. `load_sharded_bin_state_dict()` below
#      handles this format (download the index + every shard it references,
#      merge).
#   2. The stock script's `KEYS_TO_MODIFY_MAPPING` (a sequence of blind
#      substring replaces) targets an older transformers layout where
#      LlavaForConditionalGeneration exposed `.language_model`/`.vision_tower`
#      as flat top-level attributes. transformers 5.13.1 nests these under a
#      shared `.model` (confirmed by constructing an empty meta-device model
#      and reading `model.state_dict().keys()` directly): vision tower keys
#      are `model.vision_tower.*`, language model keys are
#      `model.language_model.*` (no extra `.model.` in between -- NOT
#      `model.language_model.model.*`), projector keys are
#      `model.multi_modal_projector.linear_{1,2}.*`, and `lm_head.weight`
#      stays top-level (sibling of `.model`, not under `.language_model`).
#      The raw checkpoint's own vision tower keys are also double-nested
#      (`model.vision_tower.vision_tower.vision_model.*`, confirmed by
#      inspecting the actual downloaded shards) and need de-duplicating.
#      `convert_state_dict_to_hf()` below implements this exact mapping,
#      verified to produce a byte-exact key-set match against the target
#      model's `state_dict()` (0 missing, 0 extra, 0 unmapped keys) before
#      ever calling `load_state_dict`.
#   3. The stock script only supports `model.push_to_hub(...)` (no local-save
#      path). Replaced with `save_pretrained(output_dir)` so this runs fully
#      offline, no Hub write token needed.
#
# LlavaConfig construction and the image-token embedding resize are
# unchanged from the stock script.
#
# Usage:
#   python convert_table_llava.py --text_model_id lmsys/vicuna-7b-v1.5 \
#       --vision_model_id openai/clip-vit-large-patch14-336 \
#       --old_state_dict_id SpursgoZmy/table-llava-v1.5-7b \
#       --output_dir /data/models/table-llava-v1.5-7b-hf

import argparse
import glob
import json
import os

os.environ.setdefault("HF_HOME", "/data/models")

import torch
from huggingface_hub import file_exists, hf_hub_download, snapshot_download
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
)

_MM_PROJECTOR_KEYS = {
    "model.mm_projector.0.weight": "model.multi_modal_projector.linear_1.weight",
    "model.mm_projector.0.bias": "model.multi_modal_projector.linear_1.bias",
    "model.mm_projector.2.weight": "model.multi_modal_projector.linear_2.weight",
    "model.mm_projector.2.bias": "model.multi_modal_projector.linear_2.bias",
}


def load_sharded_bin_state_dict(model_id: str) -> dict:
    """Handles the legacy `pytorch_model.bin.index.json` + sharded
    `pytorch_model-0000X-of-0000Y.bin` layout (no safetensors, no single
    `model_state_dict.bin` -- confirmed this is what SpursgoZmy/table-llava-v1.5-7b
    actually ships via the Hub API file listing)."""
    index_path = hf_hub_download(model_id, "pytorch_model.bin.index.json")
    index = json.loads(open(index_path).read())
    shard_files = sorted(set(index["weight_map"].values()))

    directory_path = snapshot_download(repo_id=model_id, allow_patterns=shard_files)
    state_dict = {}
    for shard_file in shard_files:
        shard_path = f"{directory_path}/{shard_file}"
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        state_dict.update(shard)
    return state_dict


def load_original_state_dict(model_id: str) -> dict:
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])
    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    if not original_state_dict:
        # No safetensors found -- fall back to the sharded legacy .bin format.
        original_state_dict = load_sharded_bin_state_dict(model_id)

    if "lm_head.weight" not in original_state_dict:
        original_state_dict["lm_head.weight"] = original_state_dict["model.embed_tokens.weight"].clone()
    if "model.image_newline" in original_state_dict:
        del original_state_dict["model.image_newline"]
    return original_state_dict


def _remap_key(key: str) -> str | None:
    """Returns the target (HF-native) key name for one raw checkpoint key,
    or None if this key should be dropped entirely (e.g. rotary_emb buffers,
    recomputed rather than persisted). Verified empirically against a fresh
    meta-device LlavaForConditionalGeneration's own state_dict() -- see
    module docstring point 2."""
    if key.endswith(".inv_freq"):
        return None
    if key.startswith("model.vision_tower.vision_tower.vision_model."):
        return key.replace("model.vision_tower.vision_tower.vision_model.", "model.vision_tower.")
    if key in _MM_PROJECTOR_KEYS:
        return _MM_PROJECTOR_KEYS[key]
    if key.startswith("model.layers."):
        return key.replace("model.layers.", "model.language_model.layers.")
    if key == "model.embed_tokens.weight":
        return "model.language_model.embed_tokens.weight"
    if key == "model.norm.weight":
        return "model.language_model.norm.weight"
    if key == "lm_head.weight":
        return key
    raise ValueError(f"Unrecognized checkpoint key with no known mapping: {key!r}")


def convert_state_dict_to_hf(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = _remap_key(key)
        if new_key is None:
            continue
        new_state_dict[new_key] = value
    return new_state_dict


def convert_table_llava_to_hf(text_model_id: str, vision_model_id: str, output_dir: str, old_state_dict_id: str):
    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    if "Qwen" not in text_model_id:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    # patch_size/vision_feature_select_strategy default to None on this
    # transformers version and must be set explicitly, or LlavaProcessor
    # crashes at call time (`TypeError: unsupported operand type(s) for //:
    # 'int' and 'NoneType'` in replace_image_token) -- found via the actual
    # smoke test, not documented in the stock conversion script. patch_size=14
    # matches CLIP ViT-L/14-336; "default" (drop the CLS token) is LLaVA-1.5's
    # standard strategy. num_additional_image_tokens=1 is required too --
    # replace_image_token's formula is `grid_count + num_additional - (1 if
    # default else 0)`, so with num_additional=0 it under-counts by exactly 1
    # vs. the model's actual output (576 patches after CLS-drop): confirmed
    # both by a live shape-mismatch smoke test AND by cross-checking
    # llava-hf/llava-1.5-7b-hf's own processor_config.json, which also sets
    # num_additional_image_tokens=1 for this same patch_size/strategy combo.
    processor = LlavaProcessor(
        tokenizer=tokenizer, image_processor=image_processor,
        patch_size=14, vision_feature_select_strategy="default",
        num_additional_image_tokens=1,
    )

    # vision_config left None: table-llava-v1.5-7b uses plain CLIP ViT-L/14-336
    # (not siglip), which is LlavaConfig's own default vision_config.
    config = LlavaConfig(text_config=text_config, vision_config=None)
    config.pad_token_id = 32001
    config.image_token_id = 32000

    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    if file_exists(old_state_dict_id, "model_state_dict.bin"):
        state_dict_path = hf_hub_download(old_state_dict_id, "model_state_dict.bin")
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    else:
        state_dict = load_original_state_dict(old_state_dict_id)

    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = model.model.language_model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    model.resize_token_embeddings(vocab_size + 2, pad_shape)
    model.model.language_model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(dist.sample() for _ in range(model.model.language_model.embed_tokens.weight.data[vocab_size:].shape[0])),
        dim=0,
    )
    model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple(dist.sample() for _ in range(model.lm_head.weight.data[vocab_size:].shape[0])),
        dim=0,
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved converted Table-LLaVA checkpoint to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text_model_id", required=True)
    parser.add_argument("--vision_model_id", required=True)
    parser.add_argument("--old_state_dict_id", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert_table_llava_to_hf(args.text_model_id, args.vision_model_id, args.output_dir, args.old_state_dict_id)


if __name__ == "__main__":
    main()
