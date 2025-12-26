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
