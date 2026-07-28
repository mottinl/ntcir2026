#!/usr/bin/env python3
"""Classify SciClaimEval figure-type evidence images with Qwen3-VL.

For each figure image, asks the model for:
  - the dominant chart type (bar chart, line chart, ...)
  - how many distinct sub-figures/panels the image contains (a composite
    figure like "(a) ... (b) ... (c) ..." counts as N > 1)

This lets us estimate how many figure-evidence images are actually composite
figures made of several sub-figures, which matters for evaluation since the
claim/evidence pair may only refer to one panel out of several.

Usage:
    python figures_analysis.py --split dev
    python figures_analysis.py --split dev --limit 10   # smoke test
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/models")

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# Some dev figures are huge (e.g. 6861x3748) and blow past the vision
# encoder's token budget, spiking activation memory enough to OOM even
# though the model weights themselves fit comfortably. Cap the longest side
# before handing the image to the processor -- plenty for chart-type /
# panel-count classification, which doesn't need full resolution.
MAX_IMAGE_SIDE = 2048

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DATA_ROOT = REPO_ROOT / "evaluation_pipeline" / "data"

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

CHART_TYPES = [
    "bar chart", "pie chart", "line chart", "scatter plot",
    "box plot", "radar chart", "heatmap", "histogram",
    "table", "diagram", "other",
]

PROMPT = f"""You are an expert at analyzing scientific figures from research papers.

Look at this image and answer two questions:

1. num_panels: how many distinct sub-figures/panels does this image contain?
   Papers often label sub-figures "(a)", "(b)", "(c)", ... or lay out several
   separate plots/diagrams side by side or in a grid within one figure. Count
   each such distinct plot/diagram/chart as one panel. A single unified plot
   (even with multiple lines/bars/series inside it) counts as num_panels=1.

2. type: the dominant chart type of the (main) panel, exactly one of:
   {", ".join(CHART_TYPES)}

Respond with a JSON object only, no explanation, no markdown:
{{"num_panels": <int>, "type": "<category>", "confidence": <0.0 to 1.0>}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def figures_dir_for(split: str) -> Path:
    return EVAL_DATA_ROOT / split / "data" / "figures" / split


def unique_dev_figure_paths(split: str) -> list[Path]:
    """Every figure image referenced as evi_type=='figure' in task1 or task2."""
    data_dir = EVAL_DATA_ROOT / split / "data"
    task1 = json.loads((data_dir / f"{split}_task1_release.json").read_text())
    task2 = json.loads((data_dir / f"{split}_task2_release.json").read_text())

    names = set()
    for item in task1:
        if item.get("evi_type") == "figure":
            names.add(Path(item["evi_path"]).name)
    for item in task2:
        if item.get("evi_type") == "figure":
            for key in ("evidence_id_1", "evidence_id_2"):
                path = item.get(key, "")
                if path.startswith("figures/"):
                    names.add(Path(path).name)

    fig_dir = figures_dir_for(split)
    return sorted(fig_dir / name for name in names)


def build_device_map(num_text_layers: int = 64, n_gpus: int = 2) -> dict:
    """Explicit, balanced device map across n_gpus.

    device_map="auto" (even with an explicit max_memory budget) was observed
    to send most of the ~32B-param language model to GPU 0 while GPU 1 sat
    idle, OOMing well before the model was fully loaded -- likely because
    accelerate's balanced-memory heuristic doesn't account well for the mix
    of 4-bit-quantized decoder layers with the unquantized (bf16) vision
    tower / embeddings / lm_head kept in `llm_int8_skip_modules`. Splitting
    the 64 text decoder layers evenly and putting the (small, ~1-2GB each)
    vision tower / embeddings / lm_head on different GPUs balances actual
    memory use to ~10-12GB per GPU, comfortably inside the 20GB budget.
    """
    half = num_text_layers // 2
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.norm": 1,
        "model.language_model.rotary_emb": 1,
        "lm_head": 1,
    }
    for i in range(num_text_layers):
        device_map[f"model.language_model.layers.{i}"] = 0 if i < half else 1
    return device_map


def load_model():
    print(f"Loading {MODEL_ID} ...")
    # Quantize the language model to 4-bit but keep the vision tower in
    # bf16 -- quantizing "visual" has been observed to produce broken
    # Linear4bit layers in the vision attention blocks (shape assertions
    # fail at generate() time), likely because its irregular qkv shapes
    # don't round-trip through bnb's nf4 packing correctly.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "vision_tower", "multi_modal_projector", "merger"],
    )
    n_gpus = torch.cuda.device_count()
    device_map = build_device_map(n_gpus=n_gpus) if n_gpus > 1 else "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device_map,
        quantization_config=quant_config,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print("Model loaded.")
    return model, processor


def classify_image(model, processor, image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(image.size)
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    raw = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    del inputs
    torch.cuda.empty_cache()

    match = _JSON_RE.search(raw)
    if not match:
        return {"type": "parse_error", "num_panels": None, "confidence": 0.0, "raw": raw}
    try:
        parsed = json.loads(match.group(0))
        parsed.setdefault("num_panels", None)
        parsed.setdefault("type", "unknown")
        parsed.setdefault("confidence", 0.0)
        return parsed
    except json.JSONDecodeError:
        return {"type": "parse_error", "num_panels": None, "confidence": 0.0, "raw": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()

    output_path = Path(args.output or f"figures_classification_{args.split}_qwen3vl32b.json")

    images = unique_dev_figure_paths(args.split)
    missing = [p for p in images if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} referenced figure files are missing on disk, e.g. {missing[:3]}")
        images = [p for p in images if p.exists()]
    if args.limit:
        images = images[: args.limit]
    print(f"Found {len(images)} unique figure images for split={args.split}\n")

    results: dict[str, dict] = {}
    if output_path.exists():
        results = json.loads(output_path.read_text())
        print(f"Resuming: {len(results)} results already in {output_path}")

    model, processor = load_model()

    for i, img_path in enumerate(images):
        if img_path.name in results:
            continue
        print(f"[{i+1}/{len(images)}] {img_path.name}", end=" ... ", flush=True)
        try:
            result = classify_image(model, processor, img_path)
            results[img_path.name] = result
            print(f"{result.get('type', '?')} | num_panels={result.get('num_panels')}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[img_path.name] = {"type": "error", "num_panels": None, "description": str(e)}

        if (i + 1) % args.save_every == 0:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    counts = Counter(v.get("type") for v in results.values())
    print("\n--- Chart type counts ---")
    for chart_type, count in counts.most_common():
        print(f"  {chart_type:20s} : {count}")

    panel_counts = [v.get("num_panels") for v in results.values() if isinstance(v.get("num_panels"), int)]
    n_composite = sum(1 for n in panel_counts if n > 1)
    n_single = sum(1 for n in panel_counts if n == 1)
    n_unparsed = len(results) - len(panel_counts)
    print("\n--- Composite figure (multi-panel) estimate ---")
    print(f"  total images analyzed      : {len(results)}")
    print(f"  single-panel (num_panels=1): {n_single} ({100*n_single/len(results):.1f}%)")
    print(f"  composite (num_panels>1)   : {n_composite} ({100*n_composite/len(results):.1f}%)")
    if n_unparsed:
        print(f"  unparsed / no num_panels   : {n_unparsed}")
    panel_dist = Counter(panel_counts)
    print("\n  num_panels distribution:")
    for n, c in sorted(panel_dist.items()):
        print(f"    {n:3d} panels : {c}")

    print(f"\nResults written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
