import pytest

from apolo_app_types import HuggingFaceToken
from apolo_app_types.app_types import AppType
from apolo_app_types.helm.apps.bundles.llm import Llama4ValueProcessor
from apolo_app_types.helm.apps.common import (
    APOLO_ORG_LABEL,
    APOLO_PROJECT_LABEL,
    APOLO_STORAGE_LABEL,
)
from apolo_app_types.protocols.common import ApoloSecret

from apolo_app_types_fixtures.constants import (
    APP_ID, APP_SECRETS_NAME, DEFAULT_NAMESPACE, CPU_PRESETS, TEST_PRESETS_WITH_EXTRA_LARGE_GPU
)

from apolo_apps_llm_inference import Llama4InferenceValueProcessor
from apolo_apps_llm_inference.app_types import LLama4Inputs, Llama4Size


@pytest.mark.parametrize("presets_available", [CPU_PRESETS], indirect=True)
async def test_values_llm_generation_gpu_default_preset(
    setup_clients, mock_get_preset_gpu
):
    model_to_test = Llama4Size.scout
    apolo_client = setup_clients
    with pytest.raises(RuntimeError) as err:
        input_processor = Llama4InferenceValueProcessor(
            client=apolo_client
        )
        helm_params = await input_processor.gen_extra_values(
            input_=LLama4Inputs(
                size=model_to_test,
                hf_token=ApoloSecret(key="FakeSecret"),
            ),
            apolo_client=apolo_client,
            app_type=AppType.Llama4,
            app_name="llm4",
            namespace=DEFAULT_NAMESPACE,
            app_secrets_name=APP_SECRETS_NAME,
            app_id=APP_ID,
        )

    assert err.value.args[0].startswith("No preset satisfies total VRAM")


@pytest.mark.parametrize(
    "presets_available", [TEST_PRESETS_WITH_EXTRA_LARGE_GPU], indirect=True
)
async def test_values_llm_generation_gpu_big_model(setup_clients, mock_get_preset_gpu):
    model_to_test = Llama4Size.scout
    preset_name = "a100-large"
    apolo_client = setup_clients
    input_processor = Llama4InferenceValueProcessor(
        client=apolo_client
    )
    helm_params = await input_processor.gen_extra_values(
        input_=LLama4Inputs(
            size=model_to_test,
            hf_token=ApoloSecret(key="FakeSecret"),
        ),
        apolo_client=apolo_client,
        app_type=AppType.Llama4,
        app_name="llm4",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params == {
        "serverExtraArgs": [],
        "model": {
            "modelHFName": Llama4ValueProcessor.model_map[model_to_test].model_hf_name,
            "tokenizerHFName": Llama4ValueProcessor.model_map[
                model_to_test
            ].model_hf_name,
        },
        "llm": {
            "modelHFName": Llama4ValueProcessor.model_map[model_to_test].model_hf_name,
            "tokenizerHFName": Llama4ValueProcessor.model_map[
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
            "requests": {"cpu": "8000.0m", "memory": "0M", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "8000.0m", "memory": "0M", "nvidia.com/gpu": "1"},
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
                    "host": f"{AppType.Llama4.value}--{APP_ID}.apps.some.org.neu.ro",
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
