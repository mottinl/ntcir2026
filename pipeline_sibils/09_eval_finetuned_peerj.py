#!/usr/bin/env python3
"""Step 4: evaluates the QLoRA adapter (`08_finetune_qwen3vl8b.py`) on the
`domain=peerj` subset of the **official** dev set (197 examples, never seen
during training -- the SIBiLS corpus already excludes all dev/test PMCIDs,
see `00_exclusion_list.py`), and compares against pipeline_baseline's
zero-shot predictions restricted to the same subset (pipeline_baseline's
overall scores are diluted by nlp/ml, out of scope for this pipeline --
this is an apples-to-apples comparison).

Usage:
    python 09_eval_finetuned_peerj.py --adapter qwen3vl8b_qlora_table/final
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_TASK1 = REPO_ROOT / "evaluation_pipeline" / "data" / "dev" / "data" / "dev_task1_release.json"
DEV_DATA_ROOT = REPO_ROOT / "evaluation_pipeline" / "data" / "dev" / "data"
BASELINE_PRED = REPO_ROOT / "pipeline_baseline" / "predictions_task1_dev.json"

sys.path.insert(0, str(REPO_ROOT / "evaluation_pipeline" / "eval"))

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_IMAGE_SIDE = 1536  # aligned with 08_finetune_qwen3vl8b.py's reduced cap
MAX_IMAGE_WIDTH = 1240  # (train/eval must see the same resolution)

PROMPT_TEMPLATE = """\
You are verifying a scientific claim against a table.

Claim: {claim}

Caption: {caption}

Context: {context}

Look carefully at the numbers and values in the table image, then decide \
whether the claim is Supported or Refuted by the table.

Respond with exactly one line:
Final answer: Supported
(or)
Final answer: Refuted
"""


def build_prompt(record: dict) -> str:
    return PROMPT_TEMPLATE.format(
        claim=record["claim"],
        caption=record.get("caption") or "N/A",
        context=record.get("context") or "N/A",
    )


def load_image(path: Path):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    scale = min(MAX_IMAGE_SIDE / img.height, MAX_IMAGE_WIDTH / img.width, 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default="qwen3vl8b_qlora_table/final")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", default="predictions_task1_peerj_finetuned.json")
    args = parser.parse_args()

    from parsing_utils import parse_label

    all_records = json.loads(DEV_TASK1.read_text())
    peerj_records = [r for r in all_records if r.get("domain") == "peerj"]
    logger.info("peerj subset: %d / %d dev_task1 records", len(peerj_records), len(all_records))

    ground_truth_peerj_path = Path("dev_task1_release_peerj.json")
    ground_truth_peerj_path.write_text(json.dumps(peerj_records, indent=2, ensure_ascii=False))

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    logger.info("Loading base model %s (4-bit) + adapter %s ...", args.model, args.adapter)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "vision_tower", "multi_modal_projector", "merger"],
    )
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant_config,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    logger.info("Model + adapter loaded.")

    results = []
    n_fallback = 0
    for i, record in enumerate(peerj_records):
        image_path = DEV_DATA_ROOT / record["evi_path"]
        prompt = build_prompt(record)
        try:
            image = load_image(image_path)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = output_ids[:, inputs["input_ids"].shape[1]:]
            text = processor.batch_decode(gen, skip_special_tokens=True)[0]
            label = parse_label(text)
            if label is None:
                logger.warning("[%s] could not parse label, defaulting to Refuted. Tail: %r",
                                record["claim_id"], text[-200:])
                label = "Refuted"
                n_fallback += 1
        except Exception:
            logger.exception("[%s] generation failed, defaulting to Refuted", record["claim_id"])
            label = "Refuted"
            n_fallback += 1

        logger.info("[%d/%d] %s -> %s", i + 1, len(peerj_records), record["claim_id"], label)
        results.append({"claim_id": record["claim_id"], "pred_label": label})

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Saved %d predictions to %s (%d fallback labels)", len(results), args.output, n_fallback)

    from evaluation.eval_script import eval_task_1_individual, eval_task_1_pair

    print("\n" + "=" * 50)
    print("Fine-tuned Qwen3-VL-8B + LoRA -- peerj dev subset")
    print("=" * 50)
    print("Individual scores:", eval_task_1_individual(args.output, str(ground_truth_peerj_path)))
    print("Pair scores:", eval_task_1_pair(args.output, str(ground_truth_peerj_path)))

    if BASELINE_PRED.exists():
        baseline_all = json.loads(BASELINE_PRED.read_text())
        peerj_claim_ids = {r["claim_id"] for r in peerj_records}
        baseline_peerj = [r for r in baseline_all if r["claim_id"] in peerj_claim_ids]
        baseline_peerj_path = Path("predictions_task1_dev_4bit_peerj_subset.json")
        baseline_peerj_path.write_text(json.dumps(baseline_peerj, indent=2, ensure_ascii=False))
        print("\n" + "=" * 50)
        print("Baseline zero-shot (pipeline_baseline, Qwen3-VL-32B 4bit) -- same peerj dev subset")
        print("=" * 50)
        print("Individual scores:", eval_task_1_individual(str(baseline_peerj_path), str(ground_truth_peerj_path)))
        print("Pair scores:", eval_task_1_pair(str(baseline_peerj_path), str(ground_truth_peerj_path)))
    else:
        logger.warning("Baseline predictions not found at %s, skipping comparison", BASELINE_PRED)


if __name__ == "__main__":
    main()
