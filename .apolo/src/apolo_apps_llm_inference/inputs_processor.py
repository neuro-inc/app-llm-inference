import logging
import typing as t
from decimal import Decimal
from typing import NamedTuple

from apolo_app_types import HuggingFaceToken
from apolo_app_types.protocols.common.hugging_face import HuggingFaceModelDetailDynamic
from apolo_app_types.app_types import AppType
from apolo_app_types.helm.apps.base import BaseChartValueProcessor
from apolo_app_types.helm.apps.common import (
    KEDA_HTTP_PROXY_SERVICE,
    append_apolo_storage_integration_annotations,
    gen_apolo_storage_integration_labels,
    gen_extra_values,
    get_preset,
)
from apolo_app_types.helm.utils.deep_merging import merge_list_of_dicts

from apolo_app_types.protocols.common import (
    ApoloFilesMount,
    ApoloMountMode,
    MountPath,
)
from apolo_app_types.protocols.common import (
    ApoloFilesPath,
    IngressHttp,
    NoAuth,
    Preset,
)
from apolo_app_types.protocols.common.autoscaling import AutoscalingKedaHTTP
from apolo_app_types.protocols.common.secrets_ import serialize_optional_secret
from apolo_app_types.protocols.common.storage import ApoloMountModes
from apolo_sdk import Preset as SDKPreset
from apolo_apps_llm_inference.app_types import (
    VLLMInferenceInputs,
    DeepSeekInputs,
    DeepSeekSize,
    GptOssSize,
    LLama4Inputs,
    Llama4Size,
    MistralInputs,
    MistralSize,
    GptOssInputs,
    Kimi2Inputs,
    Kimi2Size,
)


class VLLMInferenceInputsProcessor(BaseChartValueProcessor[VLLMInferenceInputs]):
    def __init__(self, *args: t.Any, **kwargs: t.Any):
        super().__init__(*args, **kwargs)

    async def gen_extra_helm_args(self, *_: t.Any) -> list[str]:
        return ["--timeout", "30m"]

    def _configure_autoscaling(self, input_: VLLMInferenceInputs) -> dict[str, t.Any]:
        """
        Configure autoscaling.
        """
        if not input_.http_autoscaling:
            return {}
        return {
            "autoscaling": {
                "enabled": True,
                "replicas": {
                    "min": input_.http_autoscaling.min_replicas,
                    "max": input_.http_autoscaling.max_replicas,
                },
                "scaledownPeriod": input_.http_autoscaling.scaledown_period,
                "requestRate": {
                    "granularity": f"{input_.http_autoscaling.request_rate.granularity}"
                    f"s",
                    "targetValue": input_.http_autoscaling.request_rate.target_value,
                    "window": f"{input_.http_autoscaling.request_rate.window_size}s",
                },
                "externalKedaHttpProxyService": KEDA_HTTP_PROXY_SERVICE,
            }
        }

    def _configure_gpu_env(
        self,
        gpu_provider: str,
        gpu_count: int,
    ) -> dict[str, t.Any]:
        """Configure GPU-specific environment variables."""

        device_ids = ",".join(str(i) for i in range(gpu_count))
        gpu_env = {}
        if gpu_provider == "amd":
            gpu_env["envAmd"] = {
                "HIP_VISIBLE_DEVICES": device_ids,
                "ROCR_VISIBLE_DEVICES": device_ids,
            }
        elif gpu_provider == "nvidia":
            # nvidia/cuda:12.8.1-devel-ubuntu20.04 (vllm after v0.9.0)
            gpu_env["envNvidia"] = {
                "PATH": "/usr/local/cuda/bin:/usr/local/sbin:"
                "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)",
                "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:"
                "/usr/local/nvidia/lib64:$(LD_LIBRARY_PATH)",
            }

        return gpu_env

    def _configure_parallel_args(
        self, server_extra_args: list[str], gpu_count: int
    ) -> list[str]:
        """Configure parallel processing arguments."""
        parallel_server_args: list[str] = []

        has_tensor_parallel = any(
            "tensor-parallel-size" in arg for arg in server_extra_args
        )
        has_pipeline_parallel = any(
            "pipeline-parallel-size" in arg for arg in server_extra_args
        )

        if gpu_count > 1 and not has_tensor_parallel and not has_pipeline_parallel:
            parallel_server_args.append(f"--tensor-parallel-size={gpu_count}")

        return parallel_server_args

    def _configure_model(self, input_: VLLMInferenceInputs) -> dict[str, str]:
        return {
            "modelHFName": input_.hugging_face_model.id,
            "tokenizerHFName": input_.tokenizer_hf_name,
        }

    def _configure_env(
        self, input_: VLLMInferenceInputs, app_secrets_name: str
    ) -> dict[str, t.Any]:
        # Start with base environment variables
        hf_token = input_.hugging_face_model.hf_token
        env_vars = {
            "HUGGING_FACE_HUB_TOKEN": serialize_optional_secret(
                hf_token.token if hf_token else None,
                secret_name=app_secrets_name
            )
        }

        # Add extra environment variables with priority over base ones
        # User-provided extra_env_vars override any existing env vars with the same name
        for env_var in input_.extra_env_vars:
            value = env_var.deserialize_value(app_secrets_name)
            if isinstance(value, str | dict):
                env_vars[env_var.name] = value
            else:
                env_vars[env_var.name] = str(value)

        return env_vars

    def _configure_extra_annotations(self, input_: VLLMInferenceInputs) -> dict[str, str]:
        extra_annotations: dict[str, str] = {}
        cache_files_path = input_.hugging_face_model.files_path
        if cache_files_path:
            storage_mount = ApoloFilesMount(
                storage_uri=cache_files_path,
                mount_path=MountPath(path="/root/.cache/huggingface"),
                mode=ApoloMountMode(mode=ApoloMountModes.RW),
            )
            extra_annotations = append_apolo_storage_integration_annotations(
                extra_annotations, [storage_mount], self.client
            )
        return extra_annotations

    def _configure_extra_labels(self, input_: VLLMInferenceInputs) -> dict[str, str]:
        extra_labels: dict[str, str] = {}
        if input_.hugging_face_model.files_path is not None:
            extra_labels.update(
                **gen_apolo_storage_integration_labels(
                    client=self.client, inject_storage=True
                )
            )
        return extra_labels

    def _configure_model_download(self, input_: VLLMInferenceInputs) -> dict[str, t.Any]:
        hf_model = input_.hugging_face_model
        # If model is already cached (cached=True and files_path set),
        # skip download entirely - model files are already on the storage mount
        if hf_model.cached and hf_model.files_path is not None:
            return {
                "modelDownload": {
                    "hookEnabled": False,
                    "initEnabled": False,
                },
                "cache": {
                    "enabled": False,
                },
            }
        # If cache storage is configured but model not yet cached, use hook to download
        if hf_model.files_path is not None:
            return {
                "modelDownload": {
                    "hookEnabled": True,
                    "initEnabled": False,
                },
                "cache": {
                    "enabled": False,
                },
            }
        # No cache configured - use init container with emptyDir cache
        return {
            "modelDownload": {
                "hookEnabled": False,
                "initEnabled": True,
            },
            "cache": {
                "enabled": True,
            },
        }

    def _configure_image(self, input_: VLLMInferenceInputs) -> dict[str, t.Any]:
        if input_.docker_image_config:
            return {
                "image": {
                    "repository": input_.docker_image_config.repository,
                    "tag": input_.docker_image_config.tag,
                    "pullPolicy": input_.docker_image_config.pull_policy,
                },
                "nvidiaImage": {
                    "repository": input_.docker_image_config.repository,
                    "tag": input_.docker_image_config.tag,
                    "pullPolicy": input_.docker_image_config.pull_policy,
                },
                "amdImage": {
                    "repository": input_.docker_image_config.repository,
                    "tag": input_.docker_image_config.tag,
                    "pullPolicy": input_.docker_image_config.pull_policy,
                },
            }
        return {}

    async def gen_extra_values(
        self,
        input_: VLLMInferenceInputs,
        app_name: str,
        namespace: str,
        app_id: str,
        app_secrets_name: str,
        *_: t.Any,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """
        Generate extra Helm values for LLM configuration.
        Incorporates:
          - Existing autoscaling logic
          - GPU detection for parallel settings
        """
        app_type = kwargs.get("app_type", AppType.LLMInference)
        values = await gen_extra_values(
            self.client,
            input_.preset,
            app_id,
            app_type,
            input_.ingress_http,
            None,
            namespace,
        )
        values["podAnnotations"] = self._configure_extra_annotations(input_)
        values["podExtraLabels"] = self._configure_extra_labels(input_)
        values.update(self._configure_model_download(input_))

        preset_name = input_.preset.name
        preset: SDKPreset = get_preset(self.client, preset_name)
        nvidia_gpus = preset.nvidia_gpu.count if preset.nvidia_gpu else 0
        amd_gpus = preset.amd_gpu.count if preset.amd_gpu else 0

        gpu_count = nvidia_gpus + amd_gpus
        if amd_gpus > 0:
            gpu_provider = "amd"
        elif nvidia_gpus > 0:
            gpu_provider = "nvidia"
        else:
            gpu_provider = "none"

        values["gpuProvider"] = gpu_provider

        gpu_env = self._configure_gpu_env(gpu_provider, gpu_count)
        parallel_args = self._configure_parallel_args(
            input_.server_extra_args, gpu_count
        )
        image_config = self._configure_image(input_)
        server_extra_args = [
            *input_.server_extra_args,
            *parallel_args,
        ]
        model = self._configure_model(input_)
        env = self._configure_env(input_, app_secrets_name)
        autoscaling = self._configure_autoscaling(input_)
        return merge_list_of_dicts(
            [
                {
                    "serverExtraArgs": server_extra_args,
                    "model": model,
                    "llm": model,
                    "env": env,
                },
                gpu_env,
                values,
                autoscaling,
                image_config,
            ]
        )

class ModelSettings(NamedTuple):
    model_hf_name: str
    vram_min_required_gb: float


T = t.TypeVar("T", LLama4Inputs, DeepSeekInputs, MistralInputs, GptOssInputs, Kimi2Inputs)


logger = logging.getLogger(__name__)


class BaseLLMBundleMixin(BaseChartValueProcessor[T]):
    """
    Base class for LLM bundle value processors.
    This class provides common functionality for processing LLM inputs
    and generating extra values for LLM applications.
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        self.llm_val_processor = VLLMInferenceInputsProcessor(*args, **kwargs)
        super().__init__(*args, **kwargs)

    cache_prefix: str = "llm_bundles"
    model_map: dict[str, ModelSettings]
    app_type: AppType

    def _get_storage_path(self) -> str:
        """
        Returns the storage path for the LLM inputs.
        :param input_:
        :return:
        """
        cluster_name = self.client.config.cluster_name
        project_name = self.client.config.project_name
        org_name = self.client.config.org_name
        return f"storage://{cluster_name}/{org_name}/{project_name}/{self.cache_prefix}"

    async def _get_preset(
        self,
        input_: T,
    ) -> Preset:
        """Retrieve the appropriate preset based on the
        input size and GPU compatibility."""
        available_presets = dict(self.client.config.presets)
        jobs_capacity = await self.client.jobs.get_capacity()
        model_settings = self.model_map[input_.size]
        min_total_vram_gb = model_settings.vram_min_required_gb

        candidates: list[tuple[Decimal, int, float, int, str]] = []
        for preset_name, available_preset in available_presets.items():
            gpu = available_preset.nvidia_gpu or available_preset.amd_gpu
            if not gpu:
                msg = f"Ignoring preset {preset_name} because it has no GPU"
                logger.info(msg)
                continue

            instances_capacity = jobs_capacity.get(preset_name, 0)
            if instances_capacity <= 0:
                msg = f"Ignoring preset {preset_name} because it has no capacity"
                logger.info(msg)
                continue

            mem_bytes = gpu.memory or 0
            cnt = gpu.count
            if mem_bytes <= 0 or cnt <= 0:
                msg = f"Ignoring preset {preset_name} because its GPU memory is <= 0"
                logger.info(msg)
                continue

            mem_gb = mem_bytes / 1e9
            total_vram = float(mem_gb) * int(cnt)
            if not total_vram >= min_total_vram_gb:
                msg = f"Preset {preset_name} has not enough VRAM"
                logger.info(msg)
                continue
                # in most of the cases, credits price will be different for each preset
            candidates.append(
                (
                    available_preset.credits_per_hour,
                    -instances_capacity,
                    total_vram,
                    cnt,
                    preset_name,
                )
            )

        if not candidates:
            err_msg = (
                f"No preset satisfies total VRAM ≥ "
                f"{min_total_vram_gb} for size={input_.size!r}."
            )
            raise RuntimeError(err_msg)
        logger.info("Candidates: %s", candidates)
        # Prefer smallest total VRAM
        best_name = min(candidates)[-1]
        return Preset(name=best_name)

    async def _llm_inputs(self, input_: T) -> VLLMInferenceInputs:
        model_settings = self.model_map[input_.size]
        hf_model = HuggingFaceModelDetailDynamic(
            id=model_settings.model_hf_name,
            visibility="public",
            hf_token=HuggingFaceToken(
                token_name="llm_bundle_token",
                token=input_.hf_token
            ),
            files_path=ApoloFilesPath(path=self._get_storage_path()),
            cached=False,
        )
        preset_chosen = await self._get_preset(input_)
        logger.info("Preset chosen: %s", preset_chosen.name)
        return VLLMInferenceInputs(
            hugging_face_model=hf_model,
            tokenizer_hf_name=hf_model.id,
            ingress_http=IngressHttp(auth=NoAuth()),
            preset=preset_chosen,
            http_autoscaling=AutoscalingKedaHTTP(scaledown_period=300)
            if input_.autoscaling_enabled
            else None,
        )

    async def gen_extra_values(
        self,
        input_: T,
        app_name: str,
        namespace: str,
        app_id: str,
        app_secrets_name: str,
        *_: t.Any,
        **kwargs: t.Any,
    ) -> dict[str, t.Any]:
        """
        Generates additional key-value pairs for use in application-specific processing
        based on the provided input and other parameters. This method executes in an
        asynchronous manner, allowing for non-blocking operations.

        :param input_: An instance of LLamaInputs containing the input data required
                       for processing.
        :param app_name: The name of the application for which the extra values
                         are being generated.
        :param namespace: The namespace associated with the application.
        :param app_id: The identifier of the application.
        :param app_secrets_name: The name of the application's secret store or
                                 credentials configuration.
        :param _: Additional positional arguments.
        :param kwargs: Additional keyword arguments for further customization or
                       processing.
        :return: A dictionary containing the generated key-value pairs as extra
                 values for the specified application.
        """

        return await self.llm_val_processor.gen_extra_values(
            input_=await self._llm_inputs(input_),
            app_name=app_name,
            namespace=namespace,
            app_secrets_name=app_secrets_name,
            app_id=app_id,
            app_type=self.app_type,
        )

    async def gen_extra_helm_args(self, *_: t.Any) -> list[str]:
        return ["--timeout", "30m"]


class Llama4InferenceValueProcessor(BaseLLMBundleMixin[LLama4Inputs]):
    app_type = AppType.Llama4
    model_map = {
        Llama4Size.scout: ModelSettings(
            model_hf_name="meta-llama/Llama-4-Scout-17B-16E",
            vram_min_required_gb=80,
        ),
        Llama4Size.scout_instruct: ModelSettings(
            model_hf_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            vram_min_required_gb=80,
        ),
    }


class DeepSeekInferenceValueProcessor(BaseLLMBundleMixin[DeepSeekInputs]):
    app_type = AppType.DeepSeek
    model_map = {
        # R1 models
        DeepSeekSize.r1: ModelSettings(
            model_hf_name="deepseek-ai/DeepSeek-R1", vram_min_required_gb=1342.0
        ),
        DeepSeekSize.r1_zero: ModelSettings(
            model_hf_name="deepseek-ai/DeepSeek-R1-Zero", vram_min_required_gb=1342.0
        ),
        DeepSeekSize.r1_distill_llama_8b: ModelSettings(
            model_hf_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            vram_min_required_gb=18.0,
        ),
        DeepSeekSize.r1_distill_llama_70b: ModelSettings(
            model_hf_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            vram_min_required_gb=161.0,
        ),
        DeepSeekSize.r1_distill_qwen_1_5_b: ModelSettings(
            model_hf_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            vram_min_required_gb=3.9,
        ),
        # V3.2 models
        DeepSeekSize.v3_2_exp: ModelSettings(
            model_hf_name="deepseek-ai/DeepSeek-V3.2-Exp",
            vram_min_required_gb=1370.0,
        ),
        DeepSeekSize.v3_2: ModelSettings(
            model_hf_name="deepseek-ai/DeepSeek-V3.2",
            vram_min_required_gb=1370.0,
        ),
    }


class MistralInferenceValueProcessor(BaseLLMBundleMixin[MistralInputs]):
    app_type = AppType.Mistral
    model_map = {
        MistralSize.mistral_7b_v02: ModelSettings(
            model_hf_name="mistralai/Mistral-7B-Instruct-v0.2",
            vram_min_required_gb=5,
        ),
        MistralSize.mistral_7b_v03: ModelSettings(
            model_hf_name="mistralai/Mistral-7B-Instruct-v0.3",
            vram_min_required_gb=5,
        ),
        MistralSize.mistral_31_24b_instruct: ModelSettings(
            model_hf_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            vram_min_required_gb=16,
        ),
        MistralSize.mistral_32_24b_instruct: ModelSettings(
            model_hf_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            vram_min_required_gb=16,
        ),
    }


class GPTOSSInferenceValueProcessor(BaseLLMBundleMixin[MistralInputs]):
    app_type = AppType.GptOss
    model_map = {
        GptOssSize.gpt_oss_20b: ModelSettings(
            model_hf_name="openai/gpt-oss-20b",
            vram_min_required_gb=16.0,
        ),
        GptOssSize.gpt_oss_120b: ModelSettings(
            model_hf_name="openai/gpt-oss-120b",
            vram_min_required_gb=80.0,
        ),
    }


class Kimi2InferenceValueProcessor(BaseLLMBundleMixin[Kimi2Inputs]):
    # TODO: Update to AppType.Kimi2 when available in apolo_app_types package
    app_type = AppType.LLMInference
    model_map = {
        # Full-weight models
        Kimi2Size.k2_base: ModelSettings(
            model_hf_name="moonshotai/Kimi-K2-Base",
            vram_min_required_gb=1000.0,
        ),
        Kimi2Size.k2_instruct: ModelSettings(
            model_hf_name="moonshotai/Kimi-K2-Instruct",
            vram_min_required_gb=1000.0,
        ),
        Kimi2Size.k2_instruct_0905: ModelSettings(
            model_hf_name="moonshotai/Kimi-K2-Instruct-0905",
            vram_min_required_gb=1000.0,
        ),
        Kimi2Size.k2_thinking: ModelSettings(
            model_hf_name="moonshotai/Kimi-K2-Thinking",
            vram_min_required_gb=1000.0,
        ),
        # GGUF quantized models (unsloth)
        Kimi2Size.k2_instruct_q2_k_xl: ModelSettings(
            model_hf_name="unsloth/Kimi-K2-Instruct-GGUF",
            vram_min_required_gb=400.0,
        ),
        Kimi2Size.k2_instruct_q4_k_xl: ModelSettings(
            model_hf_name="unsloth/Kimi-K2-Instruct-GGUF",
            vram_min_required_gb=600.0,
        ),
        Kimi2Size.k2_instruct_q8_0: ModelSettings(
            model_hf_name="unsloth/Kimi-K2-Instruct-GGUF",
            vram_min_required_gb=1100.0,
        ),
    }
