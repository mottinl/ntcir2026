"""PaddleOCR / PP-Structure (TableRecognitionPipelineV2) advisor: dedicated
table structure + OCR recognition, replacing Table-LLaVA. Tables only.

Must run under the ISOLATED venv `pipeline_advisors/.venv-paddleocr/`, not
the shared repo-root `.venv` -- paddlepaddle-gpu pulls its own pinned
nvidia-cu* wheel versions that conflict with the shared venv's torch build
(confirmed: `uv pip install paddlepaddle-gpu` downgraded nvidia-cublas/cudnn
under the shared venv, which torch/transformers/bitsandbytes also depend on
-- reverted via `uv sync`, never repeat that in the shared venv).

Also note: paddlepaddle 3.3.x has a known regression breaking CPU oneDNN
inference (`NotImplementedError: ConvertPirAttribute2RuntimeAttribute...`,
https://github.com/PaddlePaddle/Paddle/issues/77340) -- irrelevant here since
we run on GPU (a completely different execution path from oneDNN/MKLDNN,
which is CPU-only), so paddlepaddle-gpu==3.3.0 works fine. CPU-only
paddlepaddle in the shared venv is pinned to 3.2.0 specifically to dodge
that bug, in case it's ever used standalone.

GPU-only extraction: CPU inference measured ~30s-200s+/image (some
images never finished within a 3+ minute timeout) vs GPU ~3.5s/image
regardless of table complexity -- GPU is not optional for this advisor at
this corpus size (1005 images).
"""

import logging
import os

os.environ.setdefault("HF_HOME", "/data/models")

from advisor_common import Advisor

logger = logging.getLogger(__name__)


class PaddleOCRAdvisor(Advisor):
    NAME = "paddleocr"
    SCOPE = "table"

    def __init__(self, device: str = "gpu:0"):
        import numpy as np
        from paddleocr import TableRecognitionPipelineV2

        self.np = np
        logger.info("Loading PP-Structure TableRecognitionPipelineV2 (device=%s) ...", device)
        self.pipeline = TableRecognitionPipelineV2(
            use_layout_detection=False,  # input is already a cropped table image, not a full page
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device=device,
        )
        logger.info("PP-Structure loaded.")

    def describe(self, image) -> str:
        # PaddleOCR/OpenCV convention is BGR, not PIL's RGB.
        arr_bgr = self.np.array(image)[:, :, ::-1]
        results = list(self.pipeline.predict(arr_bgr))
        htmls = []
        for res in results:
            for table_res in res.get("table_res_list", []):
                html = table_res.get("pred_html")
                if html:
                    htmls.append(html)
        return "\n".join(htmls)
