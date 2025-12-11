from decimal import Decimal

import pytest
from apolo_sdk import Preset
from neuro_config_client import NvidiaGPUPreset

pytest_plugins = [
    "apolo_app_types_fixtures.apolo_clients",
    "apolo_app_types_fixtures.constants",
]


# H100 cluster preset for V3.2 model testing (1440GB total VRAM for 1370GB requirement)
TEST_PRESETS_WITH_H100_CLUSTER = {
    "cpu-small": Preset(
        cpu=2.0,
        memory=8,
        nvidia_gpu=NvidiaGPUPreset(count=0),
        credits_per_hour=Decimal("0.05"),
        available_resource_pool_names=("cpu_pool",),
    ),
    "h100-8x": Preset(
        cpu=128.0,
        memory=1024,
        nvidia_gpu=NvidiaGPUPreset(count=18, memory=80e9),
        credits_per_hour=Decimal("72"),
        available_resource_pool_names=("gpu_pool",),
    ),
}


@pytest.fixture
def mock_get_preset_gpu_h100(setup_clients):
    """Fixture that provides H100 8x GPU cluster preset for V3.2 model testing."""
    from unittest.mock import AsyncMock

    setup_clients.config.presets = TEST_PRESETS_WITH_H100_CLUSTER
    setup_clients.jobs.get_capacity = AsyncMock(
        return_value={name: 10 for name in TEST_PRESETS_WITH_H100_CLUSTER}
    )


@pytest.fixture
def mock_fetch_models(monkeypatch):
    """
    Monkeypatch load_hf_json(model, filename) to return canned dicts.
    Adjust the returned dicts to match your scenarios.
    """

    async def _fake_fetch_max_model_len(*_, **__):
        return 131_072  # simulate vLLM reporting this limit
    from apolo_apps_llm_inference import utils, outputs_processor
    monkeypatch.setattr(
        "apolo_apps_llm_inference.outputs_processor.fetch_max_model_len_from_server",
        _fake_fetch_max_model_len,
        raising=True,
    )
