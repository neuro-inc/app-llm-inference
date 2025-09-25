import typing
from enum import Enum
from typing import Literal

from apolo_app_types.protocols.common import (
    ApoloSecret,
    AppInputs,
    SchemaExtraMetadata,
)
from apolo_app_types.protocols.common import (
    AppOutputs,
    HuggingFaceCache,
    HuggingFaceModel,
    IngressHttp,
    Preset,
    SchemaMetaType,
)
from apolo_app_types.protocols.common.autoscaling import AutoscalingKedaHTTP
from apolo_app_types.protocols.common.hugging_face import HF_SCHEMA_EXTRA
from apolo_app_types.protocols.common.hugging_face import HF_TOKEN_SCHEMA_EXTRA
from apolo_app_types.protocols.common.k8s import Env
from apolo_app_types.protocols.common.openai_compat import (
    OpenAICompatChatAPI,
    OpenAICompatEmbeddingsAPI,
)
from pydantic import Field
from pydantic import model_validator


class VLLMInferenceInputs(AppInputs):
    preset: Preset
    ingress_http: IngressHttp | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Public HTTP Ingress",
            description="Enable access to your application"
            " over the internet using HTTPS.",
        ).as_json_schema_extra(),
    )
    hugging_face_model: HuggingFaceModel = Field(
        ...,
        json_schema_extra=HF_SCHEMA_EXTRA.model_copy(
            update={
                "meta_type": SchemaMetaType.INLINE,
            }
        ).as_json_schema_extra(),
    )  # noqa: N815
    tokenizer_hf_name: str = Field(  # noqa: N815
        "",
        json_schema_extra=SchemaExtraMetadata(
            description="Set the name of the tokenizer "
            "associated with the Hugging Face model.",
            title="Hugging Face Tokenizer Name",
        ).as_json_schema_extra(),
    )
    server_extra_args: list[str] = Field(  # noqa: N815
        default_factory=list,
        json_schema_extra=SchemaExtraMetadata(
            title="Server Extra Arguments",
            description="Configure extra arguments "
            "to pass to the server (see VLLM doc, e.g. --max-model-len=131072).",
        ).as_json_schema_extra(),
    )
    extra_env_vars: list[Env] = Field(  # noqa: N815
        default_factory=list,
        json_schema_extra=SchemaExtraMetadata(
            title="Extra Environment Variables",
            description=(
                "Additional environment variables to inject into the container. "
                "These will override any existing environment variables "
                "with the same name."
            ),
        ).as_json_schema_extra(),
    )
    cache_config: HuggingFaceCache | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Cache Config", description="Configure Hugging Face cache."
        ).as_json_schema_extra(),
    )
    http_autoscaling: AutoscalingKedaHTTP | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="HTTP Autoscaling",
            description="Configure autoscaling based on HTTP request rate."
            " If you enable this, "
            "please ensure that cache config "
            "is enabled as well.",
            is_advanced_field=True,
        ).as_json_schema_extra(),
    )

    @model_validator(mode="after")
    def check_autoscaling_requires_cache(self) -> "LLMInputs":
        if self.http_autoscaling and not self.cache_config:
            msg = "If HTTP autoscaling is enabled, cache_config must also be set."
            raise ValueError(msg)
        return self


class VLLMInferenceOutputs(AppOutputs):
    chat_internal_api: OpenAICompatChatAPI | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Internal Chat API",
            description="Internal Chat API compatible with "
            "OpenAI standard for seamless integration.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )
    chat_external_api: OpenAICompatChatAPI | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="External Chat API",
            description="External Chat API compatible with OpenAI standard.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )
    embeddings_internal_api: OpenAICompatEmbeddingsAPI | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Internal Embeddings API",
            description="Internal Embeddings API compatible with OpenAI "
            "standard for seamless integration.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )
    embeddings_external_api: OpenAICompatEmbeddingsAPI | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="External Embeddings API",
            description="External Embeddings API compatible with OpenAI standard.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )
    hugging_face_model: HuggingFaceModel
    tokenizer_hf_name: str = Field(  # noqa: N815
        "",
        json_schema_extra=SchemaExtraMetadata(
            description="Set the name of the tokenizer "
            "associated with the Hugging Face model.",
            title="Hugging Face Tokenizer Name",
        ).as_json_schema_extra(),
    )
    server_extra_args: list[str] = Field(  # noqa: N815
        default_factory=list,
        json_schema_extra=SchemaExtraMetadata(
            title="Server Extra Arguments",
            description="Configure extra arguments "
            "to pass to the server (see VLLM doc, e.g. --max-model-len=131072).",
        ).as_json_schema_extra(),
    )
    llm_api_key: str | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="LLM Api Key",
            description="LLM Key for the API",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )


TSize = typing.TypeVar("TSize")


class Llama4Size(str, Enum):
    scout = "Llama-4-Scout-17B-16E"
    scout_instruct = "Llama-4-Scout-17B-16E-Instruct"


class DeepSeekR1Size(str, Enum):
    r1 = "R1"
    r1_zero = "R1-Zero"
    r1_distill_llama_70b = "R1-Distill-Llama-70B"
    r1_distill_llama_8b = "R1-Distill-Llama-8B"  # noqa: N815
    r1_distill_qwen_1_5_b = "R1-Distill-Qwen-1.5B"


class MistralSize(str, Enum):
    mistral_7b_v02 = "Mistral-7B-Instruct-v0.2"
    mistral_7b_v03 = "Mistral-7B-Instruct-v0.3"
    mistral_31_24b_instruct = "Mistral-Small-3.1-24B-Instruct-2503"
    mistral_32_24b_instruct = "Mistral-Small-3.2-24B-Instruct-2506"


class GptOssSize(str, Enum):
    gpt_oss_120b = "gpt-oss-120b"
    gpt_oss_20b = "gpt-oss-20b"


class LLMBundleInputs(AppInputs, typing.Generic[TSize]):
    """
    Base class for LLM bundle inputs.
    This class can be extended by specific LLM bundle input classes.
    """

    hf_token: ApoloSecret = Field(  # noqa: N815
        ...,
        json_schema_extra=HF_TOKEN_SCHEMA_EXTRA.as_json_schema_extra(),
    )
    autoscaling_enabled: bool = Field(  # noqa: N815
        default=False,
        json_schema_extra=SchemaExtraMetadata(
            description="Enable or disable autoscaling for the LLM.",
            title="Enable Autoscaling",
        ).as_json_schema_extra(),
    )

    size: TSize


class LLama4Inputs(LLMBundleInputs[Llama4Size]):
    """
    Inputs for the Llama4 bundle.
    This class extends LLMBundleInputs to include specific fields for Llama4.
    """

    size: Llama4Size
    llm_class: Literal["llama4"] = "llama4"


class GptOssInputs(LLMBundleInputs[GptOssSize]):
    """
    Inputs for the GptOss bundle.
    This class extends LLMBundleInputs to include specific fields for OpenAIs GptOss.
    """

    size: GptOssSize
    llm_class: Literal["gpt-oss"] = "gpt-oss"


class DeepSeekR1Inputs(LLMBundleInputs[DeepSeekR1Size]):
    """
    Inputs for the DeepSeekR1 bundle.
    This class extends LLMBundleInputs to include specific fields for DeepSeekR1.
    """

    llm_class: Literal["deepseek_r1"] = "deepseek_r1"
    size: DeepSeekR1Size


class MistralInputs(LLMBundleInputs[MistralSize]):
    """
    Inputs for the Mistral bundle.
    This class extends LLMBundleInputs to include specific fields for Mistral.
    """

    llm_class: Literal["mistral"] = "mistral"
    size: MistralSize
