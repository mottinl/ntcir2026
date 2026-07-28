"""Thin wrapper around Qwen3-VL for scientific claim verification.

Non-quantized checkpoints load through transformers, same as always. AWQ
checkpoints load through vLLM instead: transformers' AWQ backend needs
gptqmodel, which -- confirmed by testing -- fails to load this repo's AWQ
checkpoint entirely (both its Marlin and GEMM kernels reject the vision
tower's intermediate_size=4304, apparently because gptqmodel's
modules_to_not_convert handling doesn't correctly skip the "visual" module
for this architecture, unrelated to the protobuf/google-generativeai
question the module docstring used to warn about). vLLM ships its own AWQ
(Marlin) kernel that doesn't hit this bug, so AWQ checkpoints are routed
there instead via QwenVLAgent.__new__. On this host vLLM is restricted to a
single GPU (see _VLLMQwenAgent), which is tight enough to OOM on large
images.

Pass load_in_4bit=True to instead load a *non*-quantized checkpoint (e.g.
Qwen/Qwen3-VL-32B-Instruct) with on-the-fly bitsandbytes 4-bit quantization,
split across both GPUs -- see _Bnb4BitQwenAgent. Callers don't need to know
or care which backend loaded.

Pass backend="vllm" to force the vLLM backend even for a plain bf16
checkpoint that isn't AWQ (e.g. Qwen3-VL-8B-Instruct) -- vLLM doesn't
actually require quantization, that's just the case __new__'s automatic
routing originally covered. The reason to do this explicitly is
generate_batch(): vLLM schedules a whole list of prompts with continuous
batching, whereas every other backend here just loops generate() one item
at a time (no padding/batching implemented). Worthwhile once a workload is
many short, similar-shaped text-only calls in a row (e.g. bulk claim
normalization) rather than one-off per-image calls.
"""

import logging
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove a leading <think>...</think> block, if present.

    Instruct checkpoints don't emit one, so this is normally a no-op; kept so
    callers work unchanged if --model is pointed back at a -Thinking checkpoint.
    """
    without_think = _THINK_BLOCK_RE.sub("", text).strip()
    return without_think if without_think else text.strip()


@lru_cache(maxsize=None)
def _is_awq(model_name: str) -> bool:
    """Cached: __new__'s routing check and _VLLMQwenAgent.__init__'s dtype
    check both call this for the same model_name on every AWQ auto-routed
    load, which otherwise means two redundant AutoConfig.from_pretrained
    round-trips (network or disk) per load."""
    from transformers import AutoConfig
    quant_config = getattr(AutoConfig.from_pretrained(model_name), "quantization_config", None)
    return bool(quant_config) and quant_config.get("quant_method") == "awq"


class QwenVLAgent:
    """Loads a Qwen3-VL checkpoint once and exposes a simple generate() call.

    AWQ-quantized checkpoints are transparently served through vLLM, and
    load_in_4bit=True routes through bitsandbytes instead (see module
    docstring); everything else loads through plain transformers as before.
    """

    def __new__(cls, model_name: str = DEFAULT_MODEL, *args, load_in_4bit: bool = False,
                backend: str | None = None, **kwargs):
        if cls is QwenVLAgent:
            if backend == "vllm":
                cls = _VLLMQwenAgent
            elif backend == "bnb4bit" or (backend is None and load_in_4bit):
                cls = _Bnb4BitQwenAgent
            elif backend is None and _is_awq(model_name):
                cls = _VLLMQwenAgent
        return object.__new__(cls)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 4096,
        attn_implementation: str | None = "flash_attention_2",
        load_in_4bit: bool = False,
        backend: str | None = None,
    ):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        import torch

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        logger.info("Loading %s ...", model_name)
        load_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:
            if attn_implementation:
                logger.warning(
                    "Could not load with attn_implementation=%r (%s); "
                    "retrying with the default attention backend.",
                    attn_implementation, exc,
                )
                load_kwargs.pop("attn_implementation")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
            else:
                raise
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info("Model loaded (device_map=auto).")

    def generate(self, images: list, text: str, greedy: bool = False, max_new_tokens: int | None = None) -> str:
        """images: list of PIL.Image (already opened). text: user prompt."""
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens or self.max_new_tokens)
        if greedy:
            gen_kwargs.update(do_sample=False)
        else:
            # Qwen's recommended sampling settings for Instruct (non-thinking) checkpoints.
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)

        try:
            with self.torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            generated = output_ids[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        finally:
            # Release KV-cache / activation tensors promptly instead of
            # leaving them for lazy allocator reclaim -- cheap and shared by
            # both predict_task1.py and predict_task2.py's sequential loops.
            del inputs
            self.torch.cuda.empty_cache()

    def generate_batch(self, items: list[tuple[list, str]], greedy: bool = False,
                        max_new_tokens: int | None = None) -> list[str]:
        """items: list of (images, text) pairs, same shape as generate()'s args.

        Sequential fallback (one generate() call per item) -- correct but no
        faster than calling generate() in a loop. Only _VLLMQwenAgent
        overrides this with real batched/continuous-batching inference;
        plain transformers and bitsandbytes backends here process one
        sequence at a time regardless (no padding/batching implemented),
        so this exists mainly so callers can write backend-agnostic code.
        """
        return [self.generate(images, text, greedy=greedy, max_new_tokens=max_new_tokens)
                for images, text in items]


class _VLLMQwenAgent(QwenVLAgent):
    """vLLM-backed implementation. Auto-routed for AWQ-quantized checkpoints
    (see QwenVLAgent.__new__); can also be requested explicitly via
    backend="vllm" for a plain (non-quantized) checkpoint -- vLLM serves bf16
    weights directly, AWQ was never a hard requirement, just the case this
    backend was originally written for. The main reason to opt in for a
    non-AWQ checkpoint is generate_batch()'s real continuous batching, which
    the plain-transformers backend's sequential generate() loop can't do.

    The 32B-AWQ weights (~20GB) don't fit alongside KV cache/activations on
    a single 20GB GPU here, and this host's GPUs are virtualized (GRID vGPU)
    slices that reject NCCL's cross-device init, so vLLM's normal multi-GPU
    tensor-parallel split isn't available either. cpu_offload_gb keeps a
    slice of the weights in host RAM instead, letting the rest run on one GPU.
    A smaller bf16 checkpoint (e.g. the 8B used for text-only normalization)
    doesn't need the offload at all -- pass cpu_offload_gb=0.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 4096,
        max_model_len: int = 8192,
        cpu_offload_gb: float = 10,
        gpu_memory_utilization: float = 0.87,
        max_image_pixels: int = 3_000_000,
        tensor_parallel_size: int = 1,
        **_ignored,
    ):
        from vllm import LLM

        self.max_new_tokens = max_new_tokens
        self.max_image_pixels = max_image_pixels

        awq = _is_awq(model_name)
        llm_kwargs = dict(
            model=model_name,
            dtype="float16" if awq else "bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            cpu_offload_gb=cpu_offload_gb,
            # Restricted to 1 by this host's vGPU/NCCL limitation (see class
            # docstring), not a vLLM constraint -- raise if that ever lifts.
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 2},
        )
        if awq:
            llm_kwargs["quantization"] = "awq_marlin"

        logger.info("Loading %s via vLLM (%s%s) ...", model_name,
                     "AWQ, Marlin kernel" if awq else "bf16, unquantized",
                     f", cpu_offload={cpu_offload_gb}GB" if cpu_offload_gb else "")
        self.llm = LLM(**llm_kwargs)
        logger.info("Model loaded (vLLM).")

    def _sampling_params(self, greedy: bool, max_new_tokens: int | None):
        from vllm import SamplingParams

        if greedy:
            return SamplingParams(max_tokens=max_new_tokens or self.max_new_tokens, temperature=0)
        # Qwen's recommended sampling settings for Instruct (non-thinking) checkpoints.
        return SamplingParams(
            max_tokens=max_new_tokens or self.max_new_tokens,
            temperature=0.7, top_p=0.8, top_k=20,
        )

    def _cap_pixels(self, image):
        """Downscale before handing to vLLM.

        The checkpoint's own default (longest_edge=16777216, ~16k vision
        tokens/image) OOM'd this 20GB GPU on this dataset's largest
        table/figure images; mm_processor_kwargs={"max_pixels": ...} didn't
        take effect via the chat() image_pil content type, so resize here
        instead -- still plenty of resolution to read axis labels and table
        cells at max_image_pixels.
        """
        w, h = image.size
        pixels = w * h
        if pixels <= self.max_image_pixels:
            return image
        scale = (self.max_image_pixels / pixels) ** 0.5
        return image.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    def _to_message(self, images: list, text: str) -> list[dict]:
        images = [self._cap_pixels(img) for img in images]
        content = [{"type": "image_pil", "image_pil": img} for img in images]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def generate(self, images: list, text: str, greedy: bool = False, max_new_tokens: int | None = None) -> str:
        messages = self._to_message(images, text)
        sampling_params = self._sampling_params(greedy, max_new_tokens)
        outputs = self.llm.chat(messages, sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def generate_batch(self, items: list[tuple[list, str]], greedy: bool = False,
                        max_new_tokens: int | None = None) -> list[str]:
        """Real batched inference: vLLM's chat() takes a list of conversations
        and schedules them together with continuous batching, instead of one
        generate() call (one full decode loop) per item -- this is the actual
        point of using vLLM here rather than plain transformers, since the
        latter has no batching at all in this codebase (see base class'
        generate_batch sequential fallback)."""
        all_messages = [self._to_message(images, text) for images, text in items]
        sampling_params = self._sampling_params(greedy, max_new_tokens)
        outputs = self.llm.chat(all_messages, sampling_params, use_tqdm=len(all_messages) > 8)
        return [o.outputs[0].text for o in outputs]


class _Bnb4BitQwenAgent(QwenVLAgent):
    """transformers + bitsandbytes 4-bit implementation, split across 2 GPUs.

    For non-AWQ checkpoints too large to fit one 20GB GPU (e.g. the 32B
    Instruct model). device_map="auto" was observed (see
    analysis/figures_analysis.py, where this approach was first validated)
    to pile most of the language model onto GPU 0 while GPU 1 sat idle,
    OOMing before loading finished -- likely because accelerate's
    balanced-memory heuristic doesn't account well for the mix of
    4-bit-quantized decoder layers with the unquantized (bf16) vision tower.
    An explicit device_map that splits the decoder layers evenly instead
    balances actual memory use to ~10-12GB per GPU. This is plain
    transformers model-parallelism (per-module .to(device) placement, no
    NCCL collectives), so it isn't affected by the vGPU/NCCL restriction
    that confines _VLLMQwenAgent to a single GPU.

    The vision tower is kept in bf16 -- quantizing it has been observed to
    produce broken Linear4bit layers in the vision attention blocks (shape
    assertions fail at generate() time), likely because its irregular qkv
    shapes don't round-trip through bnb's nf4 packing correctly.
    """

    MAX_IMAGE_SIDE = 2048  # avoids vision-encoder token blowup OOMing on the dataset's largest images

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 4096,
        num_text_layers: int = 64,
        **_ignored,
    ):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        import torch

        self.torch = torch
        self.max_new_tokens = max_new_tokens

        logger.info("Loading %s via transformers (bitsandbytes 4-bit, 2-GPU split) ...", model_name)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "vision_tower", "multi_modal_projector", "merger"],
        )
        n_gpus = torch.cuda.device_count()
        device_map = self._build_device_map(num_text_layers, n_gpus) if n_gpus > 1 else "auto"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quant_config,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info("Model loaded (bitsandbytes 4-bit, device_map=%s).",
                     "explicit 2-GPU split" if n_gpus > 1 else "auto")

    @staticmethod
    def _build_device_map(num_text_layers: int, n_gpus: int) -> dict:
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

    def _cap_side(self, image):
        if max(image.size) <= self.MAX_IMAGE_SIDE:
            return image
        scale = self.MAX_IMAGE_SIDE / max(image.size)
        return image.resize((round(image.width * scale), round(image.height * scale)))

    def generate(self, images: list, text: str, greedy: bool = False, max_new_tokens: int | None = None) -> str:
        images = [self._cap_side(img) for img in images]
        return super().generate(images, text, greedy=greedy, max_new_tokens=max_new_tokens)
