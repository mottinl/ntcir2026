"""ChartGemma advisor: ahmed-masry/chartgemma, PaliGemma-based, instruction-tuned.

Unlike DePlot, this is instruction-tuned and accepts a free-form prompt --
asked to describe the chart's data in detail (axes, series, exact values)
rather than DePlot's fixed table-linearization-only output. Figures only.
"""

import logging
import os

os.environ.setdefault("HF_HOME", "/data/models")

from advisor_common import Advisor

logger = logging.getLogger(__name__)

CHARTGEMMA_PROMPT = (
    "Describe all the data shown in this chart in detail, including axis labels, "
    "categories, series/legend labels, and exact values."
)


class ChartGemmaAdvisor(Advisor):
    NAME = "chartgemma"
    SCOPE = "figure"

    def __init__(self, model_name: str = "ahmed-masry/chartgemma"):
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        import torch

        self.torch = torch
        logger.info("Loading %s ...", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_name, dtype=torch.float16)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        logger.info("ChartGemma loaded.")

    def describe(self, image) -> str:
        inputs = self.processor(text=CHARTGEMMA_PROMPT, images=image, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, num_beams=4, max_new_tokens=512)
        generated = output_ids[:, input_len:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
