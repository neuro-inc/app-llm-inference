from apolo_apps_llm_inference.inputs_processor import (
    VLLMInferenceInputsProcessor,
)
from apolo_apps_llm_inference.outputs_processor import (
    VLLMInferenceOutputsProcessor,
)
from apolo_apps_llm_inference.types import (
    VLLMInferenceInputs,
    VLLMInferenceOutputs,
)


APOLO_APP_TYPE = "llm-inference"


__all__ = [
    "APOLO_APP_TYPE",
    "VLLMInferenceInputsProcessor",
    "VLLMInferenceOutputsProcessor",
    "VLLMInferenceInputs",
    "VLLMInferenceOutputs",
]
