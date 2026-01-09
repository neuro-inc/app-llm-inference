import logging
import typing as t

from apolo_app_types.outputs.base import BaseAppOutputsProcessor
from apolo_app_types.outputs.llm import get_llm_inference_outputs
from apolo_app_types import LLMModelConfig
from .app_types import VLLMInferenceOutputs

from .utils import fetch_max_model_len_from_server, parse_max_model_len

logger = logging.getLogger(__name__)


def _transform_to_service_api(outputs: dict[str, t.Any]) -> dict[str, t.Any]:
    """Transform old format (chat_internal_api, chat_external_api) to new ServiceAPI format."""
    result = dict(outputs)

    # Transform chat APIs to ServiceAPI format
    if "chat_internal_api" in result or "chat_external_api" in result:
        result["chat_api"] = {
            "internal_url": result.pop("chat_internal_api", None),
            "external_url": result.pop("chat_external_api", None),
        }

    # Transform embeddings APIs to ServiceAPI format
    if "embeddings_internal_api" in result or "embeddings_external_api" in result:
        result["embeddings_api"] = {
            "internal_url": result.pop("embeddings_internal_api", None),
            "external_url": result.pop("embeddings_external_api", None),
        }

    return result


class VLLMInferenceOutputsProcessor(
    BaseAppOutputsProcessor[VLLMInferenceOutputs]
):
    async def _get_model_config(self, helm_values: dict[str, t.Any],
                                vllm_outputs_dict: dict[str, t.Any]) -> LLMModelConfig | None:
        # priority 1: --max-model-len from server args
        hf_model_name = helm_values["model"]["modelHFName"]
        api_key = None
        if server_extra_args := helm_values.get("serverExtraArgs", []):
            for arg in server_extra_args:
                if arg.startswith("--max-model-len"):
                    try:
                        max_model_len = parse_max_model_len(arg.split("=", 1)[-1])
                        return LLMModelConfig(
                            context_max_tokens=max_model_len,
                        )
                    except (ValueError, TypeError):
                        pass  # fall through to next priority
                elif arg.startswith("--api-key"):
                    api_key = arg.split("=", 1)[-1]

        # priority 2: ask the INTERNAL server /v1/models
        internal_host, internal_port = (vllm_outputs_dict["chat_api"]["internal_url"]["host"],
                                        vllm_outputs_dict["chat_api"]["internal_url"]["port"])
        try:
            server_len = await fetch_max_model_len_from_server(
                internal_host, int(internal_port), expected_model_id=hf_model_name, api_key=api_key
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
        # Transform to new ServiceAPI format
        outputs = _transform_to_service_api(outputs)
        model_config = await self._get_model_config(helm_values, outputs)
        msg = f"Got outputs: {outputs}"
        logger.info(msg)
        return VLLMInferenceOutputs.model_validate({
            **outputs,
            "llm_model_config": model_config
        })
