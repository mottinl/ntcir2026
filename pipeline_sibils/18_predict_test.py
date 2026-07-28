#!/usr/bin/env python3
"""Step 4: generates Task1 predictions on the **official** `test` split
(`test_task1_release.json`, unlabelled -- no score possible, just saved
predictions), restricted to the `domain=peerj` subset (288/917, the only
domain trained on, same logic as `09_eval_finetuned_peerj.py` on the dev side).

One script for both comparisons: no `--adapter` = zero-shot base
Qwen3-VL-8B (4-bit); with `--adapter` = base + fine-tuned QLoRA adapter
(`08_finetune_qwen3vl8b.py`). Same simple prompt as training/dev eval
(`build_prompt`, identical to `09_eval_finetuned_peerj.py`) --
train/eval/test must see the same format.

Usage:
    python 18_predict_test.py --output predictions_test_peerj_base.json
    python 18_predict_test.py --adapter <output-dir>/final --output predictions_test_peerj_finetuned.json
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_TASK1 = Path("/data/ntcir_data_test/data/test_task1_release.json")
TEST_DATA_ROOT = Path("/data/ntcir_data_test/data")

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
    parser.add_argument("--adapter", default=None,
                         help="Path to a QLoRA adapter (omit for pure zero-shot base model)")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from parsing_utils import parse_label

    all_records = json.loads(TEST_TASK1.read_text())
    peerj_records = [r for r in all_records if r.get("domain") == "peerj"]
    if args.limit:
        peerj_records = peerj_records[: args.limit]
    logger.info("peerj subset: %d / %d test_task1 records", len(peerj_records), len(all_records))

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    logger.info("Loading base model %s (4-bit) ...", args.model)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "vision_tower", "multi_modal_projector", "merger"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=quant_config,
    )
    if args.adapter:
        from peft import PeftModel
        logger.info("Loading adapter %s ...", args.adapter)
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    logger.info("Model loaded (adapter=%s).", args.adapter or "none -- zero-shot base")

    results = []
    n_fallback = 0
    for i, record in enumerate(peerj_records):
        image_path = TEST_DATA_ROOT / record["evi_path"]
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


if __name__ == "__main__":
    main()
