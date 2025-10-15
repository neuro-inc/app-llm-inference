import pytest

pytest_plugins = [
    "apolo_app_types_fixtures.apolo_clients",
    "apolo_app_types_fixtures.constants",
]


@pytest.fixture
def mock_hf_files(monkeypatch):
    """
    Monkeypatch load_hf_json(model, filename) to return canned dicts.
    Adjust the returned dicts to match your scenarios.
    """

    def _fake_load_hf_json(model: str, filename: str):
        # Example: DeepSeek R1-like config with rope scaling baked in
        if filename == "config.json":
            return {
                "model_type": "deepseek_v3",
                "max_position_embeddings": 163_840,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 40,
                    "original_max_position_embeddings": 4_096,
                },
                # optional: sliding window if you want to surface it
                "sliding_window": 4_096,
            }
        if filename == "tokenizer_config.json":
            return {
                # Cap at 131072 to test min(...)
                "model_max_length": 131_072,
                # optional extras
                "chat_template": None,
            }
        return None

    from apolo_apps_llm_inference import utils
    monkeypatch.setattr(utils, "load_hf_json", _fake_load_hf_json)
