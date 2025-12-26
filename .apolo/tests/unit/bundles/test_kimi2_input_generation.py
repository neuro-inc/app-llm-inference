from apolo_app_types.app_types import AppType
from apolo_app_types.helm.apps.common import (
    APOLO_ORG_LABEL,
    APOLO_PROJECT_LABEL,
    APOLO_STORAGE_LABEL,
)
from apolo_app_types.protocols.common import ApoloSecret

from apolo_app_types_fixtures.constants import APP_ID, APP_SECRETS_NAME, DEFAULT_NAMESPACE

from apolo_apps_llm_inference import Kimi2InferenceValueProcessor
from apolo_apps_llm_inference.app_types import Kimi2Inputs, Kimi2Size


async def test_values_kimi2_generation_gpu_default_preset(
    setup_clients, mock_get_preset_gpu_h100
):
    """Test Kimi2 model generates correct helm values with H100 cluster preset."""
    model_to_test = Kimi2Size.k2_instruct
    preset_name = "h100-8x"
    apolo_client = setup_clients
    input_processor = Kimi2InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=Kimi2Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.LLMInference,  # TODO: Update to AppType.Kimi2 when available
        app_name="kimi2",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == preset_name
    assert helm_params["model"]["modelHFName"] == "moonshotai/Kimi-K2-Instruct"
    assert helm_params["llm"]["modelHFName"] == "moonshotai/Kimi-K2-Instruct"
    assert helm_params["gpuProvider"] == "nvidia"
    # Kimi K2 requires ~1000GB VRAM, using 18x80GB GPUs with tensor parallelism
    assert "--tensor-parallel-size=18" in helm_params["serverExtraArgs"]


async def test_values_kimi2_thinking_model(
    setup_clients, mock_get_preset_gpu_h100
):
    """Test Kimi2 Thinking model generates correct helm values."""
    model_to_test = Kimi2Size.k2_thinking
    preset_name = "h100-8x"
    apolo_client = setup_clients
    input_processor = Kimi2InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=Kimi2Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.LLMInference,
        app_name="kimi2",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == preset_name
    assert helm_params["model"]["modelHFName"] == "moonshotai/Kimi-K2-Thinking"
    assert helm_params["llm"]["modelHFName"] == "moonshotai/Kimi-K2-Thinking"
    assert helm_params["gpuProvider"] == "nvidia"
    assert "--tensor-parallel-size=18" in helm_params["serverExtraArgs"]


async def test_values_kimi2_base_model(
    setup_clients, mock_get_preset_gpu_h100
):
    """Test Kimi2 Base model generates correct helm values."""
    model_to_test = Kimi2Size.k2_base
    preset_name = "h100-8x"
    apolo_client = setup_clients
    input_processor = Kimi2InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=Kimi2Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.LLMInference,
        app_name="kimi2",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == preset_name
    assert helm_params["model"]["modelHFName"] == "moonshotai/Kimi-K2-Base"
    assert helm_params["llm"]["modelHFName"] == "moonshotai/Kimi-K2-Base"
    assert helm_params["gpuProvider"] == "nvidia"


async def test_values_kimi2_instruct_0905_model(
    setup_clients, mock_get_preset_gpu_h100
):
    """Test Kimi2 Instruct 0905 model generates correct helm values."""
    model_to_test = Kimi2Size.k2_instruct_0905
    preset_name = "h100-8x"
    apolo_client = setup_clients
    input_processor = Kimi2InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=Kimi2Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.LLMInference,
        app_name="kimi2",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == preset_name
    assert helm_params["model"]["modelHFName"] == "moonshotai/Kimi-K2-Instruct-0905"
    assert helm_params["llm"]["modelHFName"] == "moonshotai/Kimi-K2-Instruct-0905"
    assert helm_params["gpuProvider"] == "nvidia"


# GGUF Quantized Model Tests


async def test_values_kimi2_gguf_q2_k_xl(
    setup_clients, mock_get_preset_gpu_gguf
):
    """Test Kimi2 Q2_K_XL quantized model (400GB VRAM requirement)."""
    model_to_test = Kimi2Size.k2_instruct_q2_k_xl
    preset_name = "h100-6x"  # 6x80GB = 480GB, fits 400GB requirement
    apolo_client = setup_clients
    input_processor = Kimi2InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=Kimi2Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.LLMInference,
        app_name="kimi2",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == preset_name
    assert helm_params["model"]["modelHFName"] == "unsloth/Kimi-K2-Instruct-GGUF"
    assert helm_params["llm"]["modelHFName"] == "unsloth/Kimi-K2-Instruct-GGUF"
    assert helm_params["gpuProvider"] == "nvidia"
    assert "--tensor-parallel-size=6" in helm_params["serverExtraArgs"]


async def test_values_kimi2_gguf_q4_k_xl(
    setup_clients, mock_get_preset_gpu_gguf
):
    """Test Kimi2 Q4_K_XL quantized model (600GB VRAM requirement)."""
    model_to_test = Kimi2Size.k2_instruct_q4_k_xl
    preset_name = "h100-8x"  # 8x80GB = 640GB, fits 600GB requirement
    apolo_client = setup_clients
    input_processor = Kimi2InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=Kimi2Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.LLMInference,
        app_name="kimi2",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == preset_name
    assert helm_params["model"]["modelHFName"] == "unsloth/Kimi-K2-Instruct-GGUF"
    assert helm_params["llm"]["modelHFName"] == "unsloth/Kimi-K2-Instruct-GGUF"
    assert helm_params["gpuProvider"] == "nvidia"
    assert "--tensor-parallel-size=8" in helm_params["serverExtraArgs"]


async def test_values_kimi2_gguf_q8_0(
    setup_clients, mock_get_preset_gpu_gguf
):
    """Test Kimi2 Q8_0 quantized model (1100GB VRAM requirement)."""
    model_to_test = Kimi2Size.k2_instruct_q8_0
    preset_name = "h100-14x"  # 14x80GB = 1120GB, fits 1100GB requirement
    apolo_client = setup_clients
    input_processor = Kimi2InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=Kimi2Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.LLMInference,
        app_name="kimi2",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == preset_name
    assert helm_params["model"]["modelHFName"] == "unsloth/Kimi-K2-Instruct-GGUF"
    assert helm_params["llm"]["modelHFName"] == "unsloth/Kimi-K2-Instruct-GGUF"
    assert helm_params["gpuProvider"] == "nvidia"
    assert "--tensor-parallel-size=14" in helm_params["serverExtraArgs"]
