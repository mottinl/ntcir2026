"""Table-LLaVA advisor: SpursgoZmy/table-llava-v1.5-7b, converted to a
HF-native LlavaForConditionalGeneration checkpoint by convert_table_llava.py
(run once, output at /data/models/table-llava-v1.5-7b-hf). LLaVA-1.5
architecture, specialized on table image understanding (MMTab). Tables only.
"""

import logging
import os

os.environ.setdefault("HF_HOME", "/data/models")

from advisor_common import Advisor

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "/data/models/table-llava-v1.5-7b-hf"

TABLE_LLAVA_PROMPT = (
    "USER: <image>\nRecognize the table content and organize it as a markdown table "
    "with all row and column headers and every cell value, in reading order.\nASSISTANT:"
)


class TableLlavaAdvisor(Advisor):
    NAME = "table_llava"
    SCOPE = "table"

    def __init__(self, model_name: str = DEFAULT_CHECKPOINT):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        import torch

        self.torch = torch
        if not os.path.isdir(model_name):
            raise FileNotFoundError(
                f"Converted Table-LLaVA checkpoint not found at {model_name!r} -- "
                "run convert_table_llava.py first (one-time offline conversion)."
            )
        logger.info("Loading %s ...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, dtype=torch.float16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        logger.info("Table-LLaVA loaded.")

    def describe(self, image) -> str:
        inputs = self.processor(text=TABLE_LLAVA_PROMPT, images=image, return_tensors="pt").to(
            self.model.device, self.model.dtype
        )
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated = output_ids[:, input_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
