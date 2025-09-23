from apolo_apps_llm_inference.inputs_processor import (
    VLLMInferenceInputsProcessor,
    GPTOSSInferenceValueProcessor,
    MistralInferenceValueProcessor,
    DeepSeekInferenceValueProcessor,
    Llama4InferenceValueProcessor,
)
from apolo_apps_llm_inference.outputs_processor import (
    VLLMInferenceOutputsProcessor,
)
from apolo_apps_llm_inference.types import (
    VLLMInferenceInputs,
    VLLMInferenceOutputs,
    MistralInputs,
    GptOssInputs,
    DeepSeekR1Inputs,
    LLama4Inputs,
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
    "MistralInputs",
    "GptOssInputs",
    "DeepSeekR1Inputs",
    "LLama4Inputs",
]
