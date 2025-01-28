#!/usr/bin/env python3

import time
import requests
import statistics
import matplotlib.pyplot as plt

def measure_throughput(
    api_url: str,
    model_name: str,
    prompt: str = "Hello, I am a test prompt.",
    num_requests: int = 3,
    max_tokens: int = 64,
    is_chat: bool = False,
    api_key: str = None,
    verbose: bool = True,
) -> float:
    """
    Measures tokens per second by making repeated requests to a vLLM endpoint.
    """
    if is_chat:
        endpoint = f"{api_url}/chat/completions"
        payload_template = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
    else:
        endpoint = f"{api_url}/completions"
        payload_template = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    all_tps = []

    for i in range(num_requests):
        payload = payload_template.copy()
        start_time = time.perf_counter()
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=300)
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"[{api_url}] Request {i+1}: EXCEPTION => {e}")
            continue
        end_time = time.perf_counter()

        if resp.status_code != 200:
            if verbose:
                print(f"[{api_url}] Request {i+1}: ERROR {resp.status_code}: {resp.text[:200]}")
            continue

        data = resp.json()
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        elapsed = end_time - start_time
        if completion_tokens > 0 and elapsed > 0:
            tps = completion_tokens / elapsed
            all_tps.append(tps)
            if verbose:
                print(f"[{api_url}] Request {i+1}: {completion_tokens} tokens in {elapsed:.2f}s => {tps:.2f} tokens/s")
        else:
            if verbose:
                print(f"[{api_url}] Request {i+1}: no tokens or missing usage in response.")

    if all_tps:
        avg_tps = statistics.mean(all_tps)
        if verbose:
            print(f"\n[{api_url}] Average tokens/s = {avg_tps:.2f}")
        return avg_tps
    else:
        if verbose:
            print(f"[{api_url}] No valid TPS measurements collected.")
        return 0.0

if __name__ == "__main__":
    """
    Map each preset to a running vLLM server URL.
    Fill in real endpoints for each hardware type.
    """
    vllm_servers = {
        "cpu-small":   "http://cpu-small.example.com:8000/v1",
        "cpu-medium":  "http://cpu-medium.example.com:8000/v1",
        "cpu-large":   "http://cpu-large.example.com:8000/v1",
        "gpu-small":   "http://gpu-small.example.com:8000/v1",
        "gpu-medium":  "http://gpu-medium.example.com:8000/v1",
        "gpu-large":   "http://gpu-large.example.com:8000/v1",
        "gpu-xlarge":  "http://gpu-xlarge.example.com:8000/v1",
        "mi210x1":     "http://mi210x1.example.com:8000/v1",
        "mi210x2":     "http://mi210x2.example.com:8000/v1",
        "H100X1":      "http://h100x1.example.com:8000/v1",
        "H100X2":      "http://h100x2.example.com:8000/v1",
    }

    model_to_test = "lmsys/vicuna-7b-v1.3"
    prompt_text = "Summarize the following text about machine learning..."

    results = {}

    # Run the throughput test on each preset
    for preset, url in vllm_servers.items():
        print(f"\n=== Measuring throughput on {preset} ({url}) ===")
        tps = measure_throughput(
            api_url=url,
            model_name=model_to_test,
            prompt=prompt_text,
            num_requests=3,
            max_tokens=64,
            is_chat=False,
            api_key=None,
            verbose=True
        )
        results[preset] = tps

    # Create a bar chart with one bar per preset
    presets_order = sorted(results.keys())   # Sort alphabetically or any order you like
    x_vals = range(len(presets_order))
    tps_vals = [results[p] for p in presets_order]

    plt.figure(figsize=(10, 5))
    plt.bar(x_vals, tps_vals, color="skyblue")
    plt.xticks(x_vals, presets_order, rotation=45, ha="right")
    plt.ylabel("Tokens/second (avg)")
    plt.title(f"vLLM Throughput ({model_to_test}) on Different Presets")
    plt.tight_layout()
    plt.savefig("vllm_throughput_comparison.png")

    print("\n=== Final Results (tokens/s) ===")
    for preset in presets_order:
        val = results[preset]
        print(f"{preset}: {val:.2f} tokens/s")

    print("\nBar chart saved to 'vllm_throughput_comparison.png'.")
    # If you want to pop up a window interactively:
    # plt.show()
