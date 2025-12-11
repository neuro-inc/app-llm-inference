from apolo_app_types import HuggingFaceToken

from apolo_app_types.app_types import AppType
from apolo_app_types.helm.apps.bundles.llm import DeepSeekValueProcessor
from apolo_app_types.helm.apps.common import (
    APOLO_ORG_LABEL,
    APOLO_PROJECT_LABEL,
    APOLO_STORAGE_LABEL,
)
from apolo_app_types.protocols.common import ApoloSecret

from apolo_app_types_fixtures.constants import APP_ID, APP_SECRETS_NAME, DEFAULT_NAMESPACE

from apolo_apps_llm_inference import DeepSeekInferenceValueProcessor
from apolo_apps_llm_inference.app_types import DeepSeekInputs, DeepSeekSize



async def test_values_llm_generation_gpu_default_preset(
    setup_clients, mock_get_preset_gpu
):
    model_to_test = DeepSeekSize.r1_distill_qwen_1_5_b
    preset_name = "t4-medium"
    apolo_client = setup_clients
    input_processor = DeepSeekInferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=DeepSeekInputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        app_type=AppType.DeepSeek,
        app_name="deepseek",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params == {
        "serverExtraArgs": [],
        "model": {
            "modelHFName": DeepSeekValueProcessor.model_map[
                model_to_test
            ].model_hf_name,
            "tokenizerHFName": DeepSeekValueProcessor.model_map[
                model_to_test
            ].model_hf_name,
        },
        "llm": {
            "modelHFName": DeepSeekValueProcessor.model_map[
                model_to_test
            ].model_hf_name,
            "tokenizerHFName": DeepSeekValueProcessor.model_map[
                model_to_test
            ].model_hf_name,
        },
        "env": {
            "HUGGING_FACE_HUB_TOKEN": {
                "valueFrom": {
                    "secretKeyRef": {"name": "apps-secrets", "key": "FakeSecret"}
                }
            }
        },
        "preset_name": preset_name,
        "resources": {
            "requests": {"cpu": "2000.0m", "memory": "0M", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "2000.0m", "memory": "0M", "nvidia.com/gpu": "1"},
        },
        "tolerations": [
            {
                "effect": "NoSchedule",
                "key": "platform.neuromation.io/job",
                "operator": "Exists",
            },
            {
                "effect": "NoExecute",
                "key": "node.kubernetes.io/not-ready",
                "operator": "Exists",
                "tolerationSeconds": 300,
            },
            {
                "effect": "NoExecute",
                "key": "node.kubernetes.io/unreachable",
                "operator": "Exists",
                "tolerationSeconds": 300,
            },
            {"effect": "NoSchedule", "key": "nvidia.com/gpu", "operator": "Exists"},
        ],
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "platform.neuromation.io/nodepool",
                                    "operator": "In",
                                    "values": ["gpu_pool"],
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "ingress": {
            "grpc": {"enabled": False},
            "enabled": True,
            "className": "traefik",
            "hosts": [
                {
                    "host": f"{AppType.DeepSeek.value}--{APP_ID}.apps.some.org.neu.ro",
                    "paths": [{"path": "/", "pathType": "Prefix", "portName": "http"}],
                }
            ],
        },
        "podAnnotations": {
            APOLO_STORAGE_LABEL: '[{"storage_uri": "storage://cluster/test-org/test-project/llm_bundles", "mount_path": "/root/.cache/huggingface", "mount_mode": "rw"}]'  # noqa: E501
        },
        "podExtraLabels": {
            APOLO_STORAGE_LABEL: "true",
            APOLO_ORG_LABEL: "test-org",
            APOLO_PROJECT_LABEL: "test-project",
        },
        "modelDownload": {"hookEnabled": True, "initEnabled": False},
        "cache": {"enabled": False},
        "gpuProvider": "nvidia",
        "podLabels": {
            "platform.apolo.us/component": "app",
            "platform.apolo.us/preset": preset_name,
        },
        "apolo_app_id": APP_ID,
        "envNvidia": {
            "PATH": "/usr/local/cuda/bin:/usr/local/sbin:"
            "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$(PATH)",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:"
            "/usr/local/nvidia/lib64:$(LD_LIBRARY_PATH)",
        },
    }


def test_deepseek_v32_model_map_configuration():
    """Test V3.2 models are correctly configured in the model map."""
    from apolo_apps_llm_inference.inputs_processor import (
        DeepSeekInferenceValueProcessor,
        ModelSettings,
    )

    # Verify V3.2 models are in the model map with correct settings
    model_map = DeepSeekInferenceValueProcessor.model_map

    # V3.2 Exp
    assert DeepSeekSize.v3_2_exp in model_map
    v32_exp = model_map[DeepSeekSize.v3_2_exp]
    assert v32_exp.model_hf_name == "deepseek-ai/DeepSeek-V3.2-Exp"
    assert v32_exp.vram_min_required_gb == 1370.0

    # V3.2
    assert DeepSeekSize.v3_2 in model_map
    v32 = model_map[DeepSeekSize.v3_2]
    assert v32.model_hf_name == "deepseek-ai/DeepSeek-V3.2"
    assert v32.vram_min_required_gb == 1370.0

    # V3.2 Exp AWQ
    assert DeepSeekSize.v3_2_exp_awq in model_map
    v32_exp_awq = model_map[DeepSeekSize.v3_2_exp_awq]
    assert v32_exp_awq.model_hf_name == "QuantTrio/DeepSeek-V3.2-Exp-AWQ"
    assert v32_exp_awq.vram_min_required_gb == 400.0

    # V3.2 AWQ
    assert DeepSeekSize.v3_2_awq in model_map
    v32_awq = model_map[DeepSeekSize.v3_2_awq]
    assert v32_awq.model_hf_name == "QuantTrio/DeepSeek-V3.2-AWQ"
    assert v32_awq.vram_min_required_gb == 400.0


def test_deepseek_size_enum_has_v32_variants():
    """Test DeepSeekSize enum includes all V3.2 variants."""
    # Verify V3.2 variants exist in the enum
    assert hasattr(DeepSeekSize, 'v3_2_exp')
    assert hasattr(DeepSeekSize, 'v3_2')
    assert hasattr(DeepSeekSize, 'v3_2_exp_awq')
    assert hasattr(DeepSeekSize, 'v3_2_awq')

    # Verify enum values
    assert DeepSeekSize.v3_2_exp.value == "V3.2-Exp"
    assert DeepSeekSize.v3_2.value == "V3.2"
    assert DeepSeekSize.v3_2_exp_awq.value == "V3.2-Exp-AWQ"
    assert DeepSeekSize.v3_2_awq.value == "V3.2-AWQ"
