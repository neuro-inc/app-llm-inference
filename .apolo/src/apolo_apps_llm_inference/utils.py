import typing as t
import huggingface_hub as hfh
import json


def _read_json(path: str) -> dict[str, t.Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hf_json(hf_model_path: str, filename: str) -> dict[str, t.Any] | None:
    """Load filename from local dir, or from HF Hub if repo id and huggingface_hub is available."""

    try:
        p = hfh.hf_hub_download(repo_id=hf_model_path, filename=filename)
        return _read_json(p)
    except Exception:
        return None


COMMON_CONTEXT_KEYS = (
    # llama/mistral/qwen/gemma/neox/bert/deepseek_v3
    "max_position_embeddings",
    # gpt2/gpt-j
    "n_positions",
    # mpt
    "max_seq_len",
    "seq_length",
    "context_length",
)

def base_context_from_config(cfg: dict[str, t.Any]) -> int | None:
    for k in COMMON_CONTEXT_KEYS:
        v = cfg.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def maybe_rope_scaled_already(base_ctx: int, rope: dict[str, t.Any]) -> bool:
    """
    Detect configs that already baked the scaling into base_ctx,
    e.g. DeepSeek: base == original_max_position_embeddings * factor.
    """
    original = rope.get("original_max_position_embeddings")
    factor = rope.get("factor")
    if isinstance(original, (int, float)) and isinstance(factor, (int, float)) and factor > 0:
        try:
            return int(original * factor) == int(base_ctx)
        except Exception:
            return False
    return False

def apply_rope_scaling(base_ctx: int | None, cfg: dict[str, t.Any]) -> int | None:
    """
    Apply rope scaling ONLY if needed. Avoid double-scaling when the config
    already contains the scaled context (e.g., DeepSeek V3).
    """
    if base_ctx is None:
        return None
    rope = cfg.get("rope_scaling")
    if not isinstance(rope, dict):
        return base_ctx

    # Many configs: {"type": "...", "factor": X, "original_max_position_embeddings": Y}
    factor = rope.get("factor")
    if not isinstance(factor, (int, float)) or factor <= 0:
        return base_ctx

    # If base already equals original * factor, do NOT rescale again.
    if maybe_rope_scaled_already(base_ctx, rope):
        return int(base_ctx)

    # Otherwise scale once.
    try:
        return int(round(base_ctx * float(factor)))
    except Exception:
        return int(base_ctx)

def tokenizer_cap(tok_cfg: dict[str, t.Any]) -> int | None:
    # tokenizer_config.json may set model_max_length; ignore absurd sentinels (e.g., 1e30).
    cap = tok_cfg.get("model_max_length")
    if isinstance(cap, (int, float)) and cap < 10 ** 9:
        return int(cap)
    return None
