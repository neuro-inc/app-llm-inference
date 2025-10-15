import logging
import typing as t

from apolo_app_types.outputs.base import BaseAppOutputsProcessor
from apolo_app_types.outputs.llm import get_llm_inference_outputs
from apolo_app_types import LLMModelConfig

from utils import load_hf_json, base_context_from_config, apply_rope_scaling, tokenizer_cap
from .app_types import VLLMInferenceOutputs

logger = logging.getLogger(__name__)


class VLLMInferenceOutputsProcessor(
    BaseAppOutputsProcessor[VLLMInferenceOutputs]
):
    def _get_model_config(self, helm_values: dict[str, t.Any]) -> LLMModelConfig | None:
        hf_model_name = helm_values["model"]["modelHFName"]
        # priority 1: --max-model-len from server args
        if server_extra_args := helm_values.get("serverExtraArgs", []):
            for arg in server_extra_args:
                if arg.startswith("--max-model-len"):
                    max_model_len = arg.split("=")[-1]
                    return LLMModelConfig(
                        context_max_tokens=int(max_model_len),
                        base_from_config=int(max_model_len),
                        after_rope_scaling=int(max_model_len),
                        tokenizer_model_max_length=int(max_model_len),
                        sliding_window_tokens=None,
                        raw_config_has_rope_scaling=False
                    )
        # priority 2: compute from HF files if available
        try:
            cfg = load_hf_json(hf_model_name, "config.json") or {}
            tok = load_hf_json(hf_model_name, "tokenizer_config.json") or {}
            base_ctx = base_context_from_config(cfg)
            after_rope = apply_rope_scaling(base_ctx, cfg)
            tok_cap = tokenizer_cap(tok)

            if not after_rope and not tok_cap:
                effective = None
            elif after_rope is None:
                effective = tok_cap
            elif tok_cap is None:
                effective = after_rope
            else:
                effective = min(after_rope, tok_cap)

            return LLMModelConfig(
                context_max_tokens=int(effective) if effective else None,
                base_from_config=int(base_ctx) if base_ctx else None,
                after_rope_scaling=int(after_rope) if after_rope else None,
                tokenizer_model_max_length=int(tok_cap) if tok_cap else None,
                sliding_window_tokens=cfg.get("sliding_window"),
                raw_config_has_rope_scaling=bool(isinstance(cfg.get("rope_scaling"), dict))
            )
        # priority 3: unknown
        except Exception:
            return None

    async def _generate_outputs(
        self,
        helm_values: dict[str, t.Any],
        app_instance_id: str,
    ) -> VLLMInferenceOutputs:
        model_config = self._get_model_config(helm_values)
        outputs = await get_llm_inference_outputs(helm_values, app_instance_id)
        msg = f"Got outputs: {outputs}"
        logger.info(msg)
        return VLLMInferenceOutputs.model_validate({
            **outputs,
            "llm_model_config": model_config
        })
