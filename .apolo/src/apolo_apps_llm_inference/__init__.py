from apolo_apps_llm_inference.app_types import (
    DeepSeekInputs,
    GptOssInputs,
    Kimi2Inputs,
    LLama4Inputs,
    MistralInputs,
    VLLMInferenceInputs,
    VLLMInferenceOutputs,
)
from apolo_apps_llm_inference.inputs_processor import (
    DeepSeekInferenceValueProcessor,
    GPTOSSInferenceValueProcessor,
    Kimi2InferenceValueProcessor,
    Llama4InferenceValueProcessor,
    MistralInferenceValueProcessor,
    VLLMInferenceInputsProcessor,
)
from apolo_apps_llm_inference.outputs_processor import (
    VLLMInferenceOutputsProcessor,
)


APOLO_APP_TYPE = "llm-inference"


__all__ = [
    "APOLO_APP_TYPE",
    "VLLMInferenceInputsProcessor",
    "VLLMInferenceOutputsProcessor",
    "VLLMInferenceInputs",
    "VLLMInferenceOutputs",
    "GPTOSSInferenceValueProcessor",
    "MistralInferenceValueProcessor",
    "DeepSeekInferenceValueProcessor",
    "Llama4InferenceValueProcessor",
    "Kimi2InferenceValueProcessor",
    "MistralInputs",
    "GptOssInputs",
    "DeepSeekInputs",
    "LLama4Inputs",
    "Kimi2Inputs",
]
