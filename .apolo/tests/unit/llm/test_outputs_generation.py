import pytest

from apolo_apps_llm_inference.outputs_processor import VLLMInferenceOutputsProcessor


async def test_llm(setup_clients, mock_kubernetes_client, app_instance_id, mock_hf_files):
    res = await VLLMInferenceOutputsProcessor().generate_outputs(
        helm_values={
            "model": {
                "modelHFName": "meta-llama/Llama-3.1-8B-Instruct",
                "tokenizerHFName": "meta-llama/Llama-3.1-8B-Instruct",
            },
            "serverExtraArgs": ["--api-key dummy-api-key"],
            "env": {"VLLM_API_KEY": "dummy"},
        },
        app_instance_id=app_instance_id,
    )
    assert res["hugging_face_model"] == {
        "model_hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hf_token": None,
        "__type__": "HuggingFaceModel",
    }
    assert res["tokenizer_hf_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert res["server_extra_args"] == ["--api-key dummy-api-key"]
    assert res["llm_api_key"] == "dummy-api-key"
    assert res["chat_internal_api"]["host"] == "app.default-namespace"
    assert res["chat_internal_api"]["endpoint_url"] == "/v1/chat"
    assert res["embeddings_internal_api"]["host"] == "app.default-namespace"
    assert res["embeddings_internal_api"]["endpoint_url"] == "/v1/embeddings"
    assert res["chat_external_api"]["host"] == "example.com"
    assert res["embeddings_external_api"]["host"] == "example.com"


async def test_llm_without_server_args(
    setup_clients, mock_kubernetes_client, app_instance_id, mock_hf_files
):
    res = await VLLMInferenceOutputsProcessor().generate_outputs(
        helm_values={
            "model": {
                "modelHFName": "meta-llama/Llama-3.1-8B-Instruct",
                "tokenizerHFName": "meta-llama/Llama-3.1-8B-Instruct",
            },
            "env": {"VLLM_API_KEY": "dummy-api-key"},
        },
        app_instance_id=app_instance_id,
    )

    assert res["hugging_face_model"] == {
        "model_hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hf_token": None,
        "__type__": "HuggingFaceModel",
    }
    assert res["tokenizer_hf_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert not res["server_extra_args"]
    assert res["llm_api_key"] == "dummy-api-key"
    assert res["chat_internal_api"]["host"] == "app.default-namespace"
    assert res["chat_internal_api"]["endpoint_url"] == "/v1/chat"
    assert res["embeddings_internal_api"]["host"] == "app.default-namespace"
    assert res["embeddings_internal_api"]["endpoint_url"] == "/v1/embeddings"
    assert res["chat_external_api"]["host"] == "example.com"
    assert res["embeddings_external_api"]["host"] == "example.com"
    assert res["llm_model_config"] == {
        "context_max_tokens": 131_072,
        "base_from_config": 131_072,
        "after_rope_scaling": 1048576,
        "tokenizer_model_max_length": 131_072,
        "sliding_window_tokens": None,
        "raw_config_has_rope_scaling": True,
        "__type__": "LLMModelConfig",
    }


async def test_llm_with_model_max_lenth(
    setup_clients, mock_kubernetes_client, app_instance_id, mock_hf_files
):
    max_model_len = 132_222
    res = await VLLMInferenceOutputsProcessor().generate_outputs(
        helm_values={
            "model": {
                "modelHFName": "meta-llama/Llama-3.1-8B-Instruct",
                "tokenizerHFName": "meta-llama/Llama-3.1-8B-Instruct",

            },
            "env": {"VLLM_API_KEY": "dummy-api-key"},
            "serverExtraArgs": [f"--max-model-len={132_222}"],
        },
        app_instance_id=app_instance_id,
    )

    assert res["hugging_face_model"] == {
        "model_hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "hf_token": None,
        "__type__": "HuggingFaceModel",
    }
    assert res["tokenizer_hf_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert res["llm_api_key"] == "dummy-api-key"
    assert res["chat_internal_api"]["host"] == "app.default-namespace"
    assert res["chat_internal_api"]["endpoint_url"] == "/v1/chat"
    assert res["embeddings_internal_api"]["host"] == "app.default-namespace"
    assert res["embeddings_internal_api"]["endpoint_url"] == "/v1/embeddings"
    assert res["chat_external_api"]["host"] == "example.com"
    assert res["embeddings_external_api"]["host"] == "example.com"
    assert res["llm_model_config"] == {
        "context_max_tokens": max_model_len,
        "base_from_config": max_model_len,
        "after_rope_scaling": max_model_len,
        "tokenizer_model_max_length": max_model_len,
        "sliding_window_tokens": None,
        "raw_config_has_rope_scaling": False,
        "__type__": "LLMModelConfig",
    }

async def test_llm_without_model(
    setup_clients, mock_kubernetes_client, app_instance_id, mock_hf_files
):
    with pytest.raises(KeyError) as exc_info:
        await VLLMInferenceOutputsProcessor().generate_outputs(
            helm_values={"env": {"VLLM_API_KEY": "dummy"}},
            app_instance_id=app_instance_id,
        )
    assert str(exc_info.value) == "'model'"
