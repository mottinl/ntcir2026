#!/usr/bin/env python3
"""Step 3: QLoRA fine-tuning of Qwen3-VL-8B-Instruct on `training_task1.jsonl`
(claim + table PNG -> Supported/Refuted), produced by
`07_build_training_examples.py`.

Deliberately simple prompt (not pipeline_baseline's guardrailed Rec-3
prompt -- that one was designed to compensate, zero-shot, for the blind
spots of a model not fine-tuned on varied figures/tables; here the model is
trained directly on synthetic tables with a single perturbation operation,
so that scaffolding isn't needed). The same simple prompt must be reused at
evaluation time (`09_eval_finetuned_peerj.py`) -- train/eval must see the
same format.

QLoRA: 4-bit base (bitsandbytes NF4, vision tower not quantized --
`llm_int8_skip_modules`, same recipe as `pipeline_baseline/qwen_agent.py`'s
`_Bnb4BitQwenAgent`, already validated on this model), LoRA on the language
model's attention/MLP projections only (q/k/v/o_proj, gate/up/down_proj --
Qwen3-VL's vision tower has a different architecture, no name collision).
batch_size=1 per device + gradient accumulation: images' variable size
(hence variable visual-token count) makes real batch padding more complex
to handle correctly for a first attempt, batch=1 avoids that.

Usage:
    python 08_finetune_qwen3vl8b.py                 # default config
    python 08_finetune_qwen3vl8b.py --epochs 2 --val-pairs 50
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/models")
# Restrict this process to a single GPU *before* torch is imported anywhere:
# with both GPUs visible, transformers' Trainer auto-wraps the model in
# DataParallel (n_gpu=2) and silently doubles the effective batch size,
# which fights the model's own single-GPU device_map={"":0} placement below
# (found via a batch_size assertion tripping in the collator during a smoke
# test -- DataLoader was handing it 2-item batches, not the requested 1).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
# NOTE: PyTorch's OOM message suggests PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# for the step-16 fragmentation OOM seen on the v4 corpus, but that setting
# crashes model loading outright on this vGPU host ("CUDA driver error:
# operation not supported" in transformers' caching_allocator_warmup --
# same family of issue as Unsloth's "smart" allocator hit in an earlier,
# now-removed comparison pipeline: this virtualized GPU rejects some
# advanced CUDA memory APIs). Do NOT set it here. v3's identical
# MAX_IMAGE_WIDTH cap completed a full 248-step run without this issue, so
# it's likely order/seed-dependent rather than systematic -- if it recurs,
# add periodic torch.cuda.empty_cache() via a TrainerCallback instead.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_IMAGE_SIDE = 1536  # same guard as pipeline_baseline/qwen_agent.py (avoids a visual-token blowup on giant tables, e.g. 12517px tall)
MAX_IMAGE_WIDTH = 1240  # base width of the v3 renderer -- tables with many
# columns can grow up to MAX_WIDTH=2400 (06_render_table_png.py); without a
# cap here, single-GPU 4-bit training OOMs (CUDA OOM observed at step 9/248
# on the first v3 run, wider images than before = more visual tokens).
# Reduced from 2048/1655: the previous cap let a 3.39 Mpx peak through
# (PMC10721204_977, the corpus' largest image) landing at the end of an
# accumulation window (micro-batch 8/8 of step 210), deterministic given the
# fixed seed -- the identical OOM recurred twice (same 14.41 GiB allocated,
# same 2.42 GiB missing) despite on_substep_end. Clearing the cache more
# often doesn't help if the peak comes from an activation that's genuinely
# needed at that moment (not fragmentation): the peak itself had to shrink.
# -25% on both caps brings the corpus max down to 1.90 Mpx (mean 1.155 -> 0.528).

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


def load_and_split(train_file: Path, val_pairs: int, seed: int):
    rows = [json.loads(l) for l in train_file.read_text().splitlines() if l.strip()]
    pair_ids = sorted({r["claim_id_pair"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(pair_ids)
    val_pair_set = set(pair_ids[:val_pairs])
    train_rows = [r for r in rows if r["claim_id_pair"] not in val_pair_set]
    val_rows = [r for r in rows if r["claim_id_pair"] in val_pair_set]
    logger.info("Split: %d train examples (%d pairs), %d val examples (%d pairs)",
                len(train_rows), len(pair_ids) - val_pairs, len(val_rows), val_pairs)
    return train_rows, val_rows


class TableClaimDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _load_image(evi_path: str):
    from PIL import Image
    img = Image.open(evi_path).convert("RGB")
    scale = min(MAX_IMAGE_SIDE / img.height, MAX_IMAGE_WIDTH / img.width, 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return img


class VLCollator:
    """Tokenizes one (image, prompt, target) example at a time (batch_size=1
    upstream) and masks the prompt/image tokens out of the loss."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        assert len(batch) == 1, "this collator assumes per_device_batch_size=1"
        record = batch[0]
        image = _load_image(record["evi_path"])
        prompt = build_prompt(record)
        target = f"Final answer: {record['label']}"

        user_turn = {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
        assistant_turn = {"role": "assistant", "content": [{"type": "text", "text": target}]}

        prompt_only = self.processor.apply_chat_template(
            [user_turn], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        prompt_len = prompt_only["input_ids"].shape[1]

        full = self.processor.apply_chat_template(
            [user_turn, assistant_turn], tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )

        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full["labels"] = labels
        return full


def evaluate_accuracy(model, processor, rows: list[dict], max_new_tokens: int = 16) -> dict:
    import torch
    from parsing_utils import parse_label

    model.eval()
    correct, total, unparsed = 0, 0, 0
    for record in rows:
        image = _load_image(record["evi_path"])
        prompt = build_prompt(record)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = output_ids[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(gen, skip_special_tokens=True)[0]
        pred = parse_label(text)
        total += 1
        if pred is None:
            unparsed += 1
        elif pred == record["label"]:
            correct += 1
    model.train()
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total, "unparsed": unparsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-file", default="training_task1.jsonl")
    parser.add_argument("--output-dir", default="qwen3vl8b_qlora_table")
    parser.add_argument("--val-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--limit-train", type=int, default=None, help="Debug: cap train set size")
    parser.add_argument("--limit-val", type=int, default=None, help="Debug: cap val set size")
    args = parser.parse_args()

    train_rows, val_rows = load_and_split(Path(args.train_file), args.val_pairs, args.seed)
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    if args.limit_val:
        val_rows = val_rows[: args.limit_val]

    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from transformers import TrainingArguments, Trainer, TrainerCallback
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    torch.manual_seed(args.seed)

    logger.info("Loading %s (bitsandbytes 4-bit, single GPU) ...", args.model)
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
    processor = AutoProcessor.from_pretrained(args.model)
    logger.info("Model loaded.")

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    collator = VLCollator(processor)
    train_ds = TableClaimDataset(train_rows)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        # paged_adamw_8bit crashes on this host's GPU (bitsandbytes CUDA op
        # "not supported at line 670" -- a paging/prefetch call that this
        # virtualized MIG/GRID GPU apparently doesn't support). Only ~44M
        # LoRA params are trainable so plain AdamW's memory cost is trivial
        # anyway (~0.5GB for fp32 moments), no need for an 8-bit optimizer.
        optim="adamw_torch",
        dataloader_num_workers=2,
        seed=args.seed,
    )
    class EmptyCacheCallback(TrainerCallback):
        """Table/figure images vary a lot in size (v3+ renderer's dynamic
        width; figures are real downloaded JPEGs of very
        variable resolution too), and the CUDA allocator fragments over
        steps until a deterministic OOM (same step every run, given the
        fixed seed/data order -- confirmed on the v4 corpus, step 16/246
        every time). expandable_segments would be the standard fix but
        crashes model loading outright on this vGPU host (see comment above
        on PYTORCH_CUDA_ALLOC_CONF), so free reserved-but-unused memory
        explicitly instead, once per optimizer step.

        2026-07-19: first tried also hooking on_substep_end (clear after
        every micro-batch, not just every grad_accum=8 window) to fight an
        OOM at step 210/1698 -- but the *exact same* OOM (same step, same
        14.41 GiB allocated, same 2.42 GiB requested) recurred even with that
        change, proving it wasn't fragmentation at all: the last micro-batch
        of that accumulation window was PMC10721204_977, the single largest
        image in the corpus (3.39 Mpx, right at the old resolution cap) --
        real peak activation memory, not reclaimable cache. Fixed at the
        source instead (MAX_IMAGE_SIDE/WIDTH lowered above, corpus max now
        1.90 Mpx), so on_substep_end's extra CUDA syncs (~7x more calls
        across a ~1700-step run) are pure overhead with no benefit -- back to
        on_step_end only."""
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, data_collator=collator,
                       callbacks=[EmptyCacheCallback()])

    logger.info("Starting training: %d examples, %.1f epochs, effective batch %d -> ~%d optimizer steps",
                len(train_rows), args.epochs,
                args.per_device_batch_size * args.grad_accum,
                int(len(train_rows) * args.epochs / (args.per_device_batch_size * args.grad_accum)))
    trainer.train()

    model.config.use_cache = True
    logger.info("Training done. Evaluating on internal validation split (%d pairs, %d examples) ...",
                args.val_pairs, len(val_rows))
    metrics = evaluate_accuracy(model, processor, val_rows)
    logger.info("Internal validation: %s", metrics)

    final_dir = Path(args.output_dir) / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    Path(args.output_dir, "internal_val_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Adapter saved to %s", final_dir)


if __name__ == "__main__":
    main()
