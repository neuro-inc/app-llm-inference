import pytest

pytest_plugins = [
    "apolo_app_types_fixtures.apolo_clients",
    "apolo_app_types_fixtures.constants",
]


@pytest.fixture
def mock_fetch_models(monkeypatch):
    """
    Monkeypatch load_hf_json(model, filename) to return canned dicts.
    Adjust the returned dicts to match your scenarios.
    """

    async def _fake_fetch_max_model_len(host, port, expected_model_id, *, timeout_s=5.0):
        return 131_072  # simulate vLLM reporting this limit
    from apolo_apps_llm_inference import utils, outputs_processor
    monkeypatch.setattr(
        "apolo_apps_llm_inference.outputs_processor.fetch_max_model_len_from_server",
        _fake_fetch_max_model_len,
        raising=True,
    )
