import logging
import typing as t

from apolo_app_types.outputs.base import BaseAppOutputsProcessor
from apolo_app_types.outputs.llm import get_llm_inference_outputs
from apolo_app_types import LLMModelConfig

from .utils import fetch_max_model_len_from_server
from .app_types import VLLMInferenceOutputs

logger = logging.getLogger(__name__)


class VLLMInferenceOutputsProcessor(
    BaseAppOutputsProcessor[VLLMInferenceOutputs]
):
    async def _get_model_config(self, helm_values: dict[str, t.Any],
                                vllm_outputs_dict: dict[str, t.Any]) -> LLMModelConfig | None:
        # priority 1: --max-model-len from server args
        hf_model_name = helm_values["model"]["modelHFName"]

        # priority 1: explicit --max-model-len from server args
        if server_extra_args := helm_values.get("serverExtraArgs", []):
            for arg in server_extra_args:
                if arg.startswith("--max-model-len"):
                    try:
                        max_model_len = int(arg.split("=", 1)[-1])
                        return LLMModelConfig(
                            context_max_tokens=max_model_len,
                        )
                    except (ValueError, TypeError):
                        pass  # fall through to next priority

        # priority 2: ask the INTERNAL server /v1/models
        internal_host, internal_port = (vllm_outputs_dict["chat_internal_api"]["host"],
                                        vllm_outputs_dict["chat_internal_api"]["port"])
        try:
            server_len = await fetch_max_model_len_from_server(
                internal_host, int(internal_port), expected_model_id=hf_model_name
            )
            if server_len:
                return LLMModelConfig(
                    context_max_tokens=server_len,
                )
        except Exception as err:
            # swallow and try next priority
            pass

    async def _generate_outputs(
        self,
        helm_values: dict[str, t.Any],
        app_instance_id: str,
    ) -> VLLMInferenceOutputs:

        outputs = await get_llm_inference_outputs(helm_values, app_instance_id)
        model_config = await self._get_model_config(helm_values, outputs)
        msg = f"Got outputs: {outputs}"
        logger.info(msg)
        return VLLMInferenceOutputs.model_validate({
            **outputs,
            "llm_model_config": model_config
        })
