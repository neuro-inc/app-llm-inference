import httpx


async def fetch_max_model_len_from_server(
    host: str,
    port: int,
    expected_model_id: str,
    api_key: str | None = None,
    *,
    timeout_s: float = 5.0,
) -> int | None:
    """
    Query the internal vLLM server /v1/models and return max_model_len for the expected model.
    Tries to match by id, then by root field.
    """
    url = f"http://{host}:{port}/v1/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
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


def parse_max_model_len(raw) -> int:
    """
    Parse values like:
      131072      -> 131072
      "131072"    -> 131072
      "1k"        -> 1000
      "1K"        -> 1024
      "25.6k"     -> 25600
      "2g"        -> 2000000000
      "2G"        -> 2147483648
    Only the last character is checked for unit: k/m/g (decimal) or K/M/G (binary).
    """
    if isinstance(raw, (int, float)):
        return int(raw)

    s = str(raw).strip()
    if not s:
        raise ValueError("empty max model len")

    last = s[-1]
    if last.isdigit():                   # no unit, pure number
        return int(float(s))

    num_str = s[:-1].strip()
    if not num_str:
        raise ValueError(f"invalid number: {raw!r}")
    num = float(num_str)

    if last == "k": mult = 10**3
    elif last == "m": mult = 10**6
    elif last == "g": mult = 10**9
    elif last == "K": mult = 1 << 10
    elif last == "M": mult = 1 << 20
    elif last == "G": mult = 1 << 30
    else:
        raise ValueError(f"unknown unit: {last!r}")

    return int(num * mult)
