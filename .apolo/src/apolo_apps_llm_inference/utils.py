import httpx


async def fetch_max_model_len_from_server(
    host: str,
    port: int,
    expected_model_id: str,
    *,
    timeout_s: float = 5.0,
) -> int | None:
    """
    Query the internal vLLM server /v1/models and return max_model_len for the expected model.
    Tries to match by id, then by root field.
    """
    url = f"http://{host}:{port}/v1/models"
    headers = {"Accept": "application/json"}

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        payload = r.json()

    models: list[dict] = payload.get("data", [])
    # Prefer exact id match, then root match, else first model
    candidates = (
        [m for m in models if m.get("id") == expected_model_id]
        or [m for m in models if m.get("root") == expected_model_id]
        or models[:1]
    )
    if not candidates:
        return None

    m = candidates[0]
    # vLLM uses max_model_len; guard for alternative spellings just in case
    val = m.get("max_model_len") or m.get("max_model_length")
    return int(val) if isinstance(val, int) else None
