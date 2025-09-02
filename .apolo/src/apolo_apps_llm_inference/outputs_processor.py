import logging
import typing as t

from apolo_app_types.outputs.base import BaseAppOutputsProcessor
from apolo_app_types.outputs.llm import get_llm_inference_outputs

from .types import VLLMInferenceOutputs


logger = logging.getLogger(__name__)


class VLLMInferenceOutputsProcessor(
    BaseAppOutputsProcessor[VLLMInferenceOutputs]
):
    async def _generate_outputs(
        self,
        helm_values: dict[str, t.Any],
        app_instance_id: str,
    ) -> VLLMInferenceOutputs:
        outputs = await get_llm_inference_outputs(helm_values, app_instance_id)
        msg = f"Got outputs: {outputs}"
        logger.info(msg)
        return VLLMInferenceOutputs.model_validate(outputs)
