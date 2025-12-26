from apolo_apps_llm_inference.inputs_processor import (
    VLLMInferenceInputsProcessor,
    GPTOSSInferenceValueProcessor,
    DeepSeekInferenceValueProcessor,
    Llama4InferenceValueProcessor,
    MistralInferenceValueProcessor,
    Kimi2InferenceValueProcessor,
)
from apolo_apps_llm_inference.outputs_processor import (
    VLLMInferenceOutputsProcessor,
)
from apolo_apps_llm_inference.app_types import (
    VLLMInferenceInputs,
    VLLMInferenceOutputs,
    MistralInputs,
    GptOssInputs,
    DeepSeekInputs,
    LLama4Inputs,
    Kimi2Inputs,
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
