import logging
import typing as t

from apolo_app_types.outputs.base import BaseAppOutputsProcessor
from apolo_app_types import LLMModelConfig, HuggingFaceModel
from apolo_app_types.outputs.common import (
    get_service_host_port,
    get_ingress_host_port,
    INSTANCE_LABEL,
)
from apolo_app_types.outputs.llm import parse_cli_args
from apolo_app_types.protocols.common.openai_compat import (
    OpenAICompatChatAPI,
    OpenAICompatEmbeddingsAPI,
)
from .app_types import VLLMInferenceOutputs

from .utils import fetch_max_model_len_from_server, parse_max_model_len

logger = logging.getLogger(__name__)


async def get_llm_inference_outputs(
    helm_values: dict[str, t.Any], app_instance_id: str
) -> dict[str, t.Any]:
    internal_host, internal_port = await get_service_host_port(
        match_labels={
            INSTANCE_LABEL: app_instance_id,
        }
    )
    server_extra_args = helm_values.get("serverExtraArgs", [])
    cli_args = parse_cli_args(server_extra_args)
    api_key = cli_args.get("api-key") or helm_values.get("env", {}).get("VLLM_API_KEY")

    model_name = helm_values["model"]["modelHFName"]
    tokenizer_name = helm_values["model"].get("tokenizerHFName", "")
    hf_model = HuggingFaceModel(model_hf_name=model_name)

    chat_internal = OpenAICompatChatAPI(
        host=internal_host, port=int(internal_port), protocol="http", hf_model=hf_model
    )
    embeddings_internal = OpenAICompatEmbeddingsAPI(
        host=internal_host, port=int(internal_port), protocol="http", hf_model=hf_model
    )

    ingress_host_port = await get_ingress_host_port(
        match_labels={INSTANCE_LABEL: app_instance_id}
    )

    chat_external = None
    embeddings_external = None
    if ingress_host_port:
        chat_external = OpenAICompatChatAPI(
            host=ingress_host_port[0], port=int(ingress_host_port[1]),
            protocol="https", hf_model=hf_model
        )
        embeddings_external = OpenAICompatEmbeddingsAPI(
            host=ingress_host_port[0], port=int(ingress_host_port[1]),
            protocol="https", hf_model=hf_model
        )

    return {
        "chat_api": {
            "internal_url": chat_internal.model_dump(),
            "external_url": chat_external.model_dump() if chat_external else None,
        },
        "embeddings_api": {
            "internal_url": embeddings_internal.model_dump(),
            "external_url": embeddings_external.model_dump() if embeddings_external else None,
        },
        "hugging_face_model": hf_model.model_dump(),
        "tokenizer_hf_name": tokenizer_name,
        "server_extra_args": server_extra_args,
        "llm_api_key": api_key,
    }


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
        model_config = await self._get_model_config(helm_values, outputs)
        msg = f"Got outputs: {outputs}"
        logger.info(msg)
        return VLLMInferenceOutputs.model_validate({
            **outputs,
            "llm_model_config": model_config
        })
