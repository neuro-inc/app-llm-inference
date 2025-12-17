import typing
from enum import Enum
from typing import Literal

from apolo_app_types import LLMModelConfig, ContainerImage
from apolo_app_types.protocols.common import (
    ApoloSecret,
    AppInputs,
    SchemaExtraMetadata,
    ServiceAPI
)
from apolo_app_types.protocols.common import (
    AppOutputs,
    HuggingFaceModel,
    IngressHttp,
    Preset,
    SchemaMetaType,
)
from apolo_app_types.protocols.common.autoscaling import AutoscalingKedaHTTP
from apolo_app_types.protocols.common.hugging_face import HF_SCHEMA_EXTRA, HF_TOKEN_SCHEMA_EXTRA, HuggingFaceModelDetailDynamic
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
    hugging_face_model: HuggingFaceModel  | HuggingFaceModelDetailDynamic = Field(
        ...,)  # noqa: N815
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
    docker_image_config: ContainerImage | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Docker Image Config",
            description="Override container image for vLLM.",
            is_advanced_field=True,
        ).as_json_schema_extra(),
    )

    @model_validator(mode="after")
    def check_autoscaling_requires_cache(self) -> "VLLMInferenceInputs":
        if self.http_autoscaling:
            # Check cache based on model type
            has_cache = False
            if isinstance(self.hugging_face_model, HuggingFaceModelDetailDynamic):
                has_cache = self.hugging_face_model.files_path is not None
            else:
                has_cache = self.hugging_face_model.hf_cache is not None
            if not has_cache:
                msg = "If HTTP autoscaling is enabled, cache_config must also be set."
                raise ValueError(msg)
        return self


class VLLMInferenceOutputs(AppOutputs):
    chat_api: ServiceAPI[OpenAICompatChatAPI] | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Chat API",
            description="Chat API compatible with "
            "OpenAI standard for seamless integration.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )

    embeddings_api: ServiceAPI[OpenAICompatEmbeddingsAPI] | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="Embeddings API",
            description="Embeddings API compatible with OpenAI "
            "standard for seamless integration.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )

    hugging_face_model: HuggingFaceModel
    llm_model_config: LLMModelConfig | None = Field(
        default=None,
        json_schema_extra=SchemaExtraMetadata(
            title="LLM Model Config",
            description="Configuration details of the deployed LLM model.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )
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


class DeepSeekSize(str, Enum):
    # R1 models
    r1 = "R1"
    r1_zero = "R1-Zero"
    r1_distill_llama_70b = "R1-Distill-Llama-70B"
    r1_distill_llama_8b = "R1-Distill-Llama-8B"  # noqa: N815
    r1_distill_qwen_1_5_b = "R1-Distill-Qwen-1.5B"
    # V3.2 models
    v3_2_exp = "V3.2-Exp"
    v3_2 = "V3.2"


class MistralSize(str, Enum):
    mistral_7b_v02 = "Mistral-7B-Instruct-v0.2"
    mistral_7b_v03 = "Mistral-7B-Instruct-v0.3"
    mistral_31_24b_instruct = "Mistral-Small-3.1-24B-Instruct-2503"
    mistral_32_24b_instruct = "Mistral-Small-3.2-24B-Instruct-2506"


class GptOssSize(str, Enum):
    gpt_oss_120b = "gpt-oss-120b"
    gpt_oss_20b = "gpt-oss-20b"


class Kimi2Size(str, Enum):
    # Full-weight models
    k2_base = "K2-Base"
    k2_instruct = "K2-Instruct"
    k2_instruct_0905 = "K2-Instruct-0905"
    k2_thinking = "K2-Thinking"
    # GGUF quantized models (unsloth)
    k2_instruct_q2_k_xl = "K2-Instruct-Q2_K_XL"
    k2_instruct_q4_k_xl = "K2-Instruct-Q4_K_XL"
    k2_instruct_q8_0 = "K2-Instruct-Q8_0"


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


class DeepSeekInputs(LLMBundleInputs[DeepSeekSize]):
    """
    Inputs for the DeepSeek bundle.
    This class extends LLMBundleInputs to include specific fields for DeepSeek models.
    """

    llm_class: Literal["deepseek"] = "deepseek"
    size: DeepSeekSize


class MistralInputs(LLMBundleInputs[MistralSize]):
    """
    Inputs for the Mistral bundle.
    This class extends LLMBundleInputs to include specific fields for Mistral.
    """

    llm_class: Literal["mistral"] = "mistral"
    size: MistralSize


class Kimi2Inputs(LLMBundleInputs[Kimi2Size]):
    """
    Inputs for the Kimi2 bundle.
    This class extends LLMBundleInputs to include specific fields for Moonshot AI's
    Kimi K2 models.
    """

    llm_class: Literal["kimi2"] = "kimi2"
    size: Kimi2Size
