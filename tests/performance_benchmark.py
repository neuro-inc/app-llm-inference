import subprocess
import time
import requests
import statistics
import csv
import os
import matplotlib.pyplot as plt
from typing import Optional

CSV_FILENAME = "vllm_benchmark_results.csv"

def append_csv_row(
    preset: str,
    model_name: str,
    gpu_provider: str,
    tokens_per_sec: float
):
    """
    Opens the CSV in append mode and writes a single row:
    [preset, model_name, gpu_provider, tokens_per_sec]

    If the file doesn't exist yet, we write the header first.
    """
    file_exists = os.path.exists(CSV_FILENAME)
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["preset", "model", "gpu_provider", "tokens_per_second"])
        writer.writerow([preset, model_name, gpu_provider, tokens_per_sec])


def deploy_preset(preset: str) -> None:
    """
    Runs an Apolo command to deploy the vLLM on the given preset.
    Adjust the command as needed for your actual environment.
    """
    # Decide AMD vs. NVIDIA vs. CPU
    if preset in ["mi210x1", "mi210x2"]:
        gpu_provider = "amd"
    elif preset in ["gpu-small", "gpu-medium", "gpu-large", "gpu-xlarge", "H100X1", "H100X2"]:
        gpu_provider = "nvidia"
    else:
        gpu_provider = "cpu"

    # Example environment overrides (like HIP_VISIBLE_DEVICES or NVIDIA_VISIBLE_DEVICES)
    env_flags = []
    if gpu_provider == "amd":
        # Single or dual AMD
        if preset == "mi210x1":
            env_flags.append('--set "envAmd.HIP_VISIBLE_DEVICES=0"')
        else:
            env_flags.append('--set "envAmd.HIP_VISIBLE_DEVICES=0,1"')
    elif gpu_provider == "nvidia":
        # Decide how many GPUs
        if preset in ["gpu-small", "H100X1"]:
            env_flags.append('--set "envNvidia.NVIDIA_VISIBLE_DEVICES=0"')
        elif preset in ["gpu-medium", "H100X2"]:
            env_flags.append('--set "envNvidia.NVIDIA_VISIBLE_DEVICES=0,1"')
        elif preset == "gpu-large":
            env_flags.append('--set "envNvidia.NVIDIA_VISIBLE_DEVICES=0,1,2,3"')
        elif preset == "gpu-xlarge":
            env_flags.append('--set "envNvidia.NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"')

    base_command = [
        "apolo",
        "run",
        "--pass-config",
        "image://novoserve/apolo/taddeus/app-deployment",
        "--",
        "install",
        "https://github.com/neuro-inc/app-llm-inference",
        "llm-inference",   # helm release name
        preset,            # you can use the preset as the instance name
        "charts/llm-inference-app",
        "--timeout=5m",
        f'--set "preset_name={preset}"',
        f'--set "gpuProvider={gpu_provider}"',
        '--set "env.HUGGING_FACE_HUB_TOKEN=YOUR_TOKEN_HERE"',
        '--set "ingress.enabled=true"',
        '--set "ingress.clusterName=novoserve"',
        '--set "model.modelRevision=main"',
        '--set "model.tokenizerRevision=main"',
    ] + env_flags

    print(f"Deploying preset={preset} with command:\n  {' \\\n  '.join(base_command)}\n")
    subprocess.run(" ".join(base_command), shell=True, check=False)


def wait_for_endpoint(api_url: str, timeout_seconds: int = 300) -> bool:
    """
    Polls the /v1/models (or a custom endpoint) every 5 seconds until
    the service is up or we hit timeout_seconds.
    Returns True if the endpoint responded 200 OK, else False.
    """
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            resp = requests.get(api_url + "/models", timeout=5)
            if resp.status_code == 200:
                print(f"[{api_url}] is online!")
                return True
        except requests.RequestException:
            pass
        print(f"Waiting for [{api_url}] to come online...")
        time.sleep(5)
    print(f"Timed out waiting for [{api_url}] to become available.")
    return False


def measure_throughput(
    api_url: str,
    model_name: str,
    prompt: str = "Hello, I am a test prompt.",
    num_requests: int = 3,
    max_tokens: int = 64,
    is_chat: bool = False,
    api_key: Optional[str] = None,
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


def is_multi_gpu_preset(preset: str) -> bool:
    """
    Simple helper to identify multi-GPU presets
    (so we can decide to test large 32B model).
    """
    multi_gpu_list = ["gpu-medium", "gpu-large", "gpu-xlarge", "mi210x2", "H100X2"]
    return preset in multi_gpu_list


def main():
    # Preset -> endpoint
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

    # Models to test
    llama_8b = "meta-llama/Llama-3.1-8B"
    qwen_32b = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

    # We'll also keep an in-memory list for final chart building
    results_in_memory = []

    for preset, url in vllm_servers.items():
        print(f"\n=== Deploying & Benchmarking: {preset} ===\n")

        # 1) Deploy
        deploy_preset(preset)

        # 2) Wait until endpoint is up
        online = wait_for_endpoint(url, timeout_seconds=300)
        if not online:
            print(f"[{preset}] Endpoint never came online. Skipping.")
            continue

        # 3) Decide which model(s) to test
        test_models = [llama_8b]
        if is_multi_gpu_preset(preset):
            test_models.append(qwen_32b)

        # 4) Determine gpu provider
        if preset in ["mi210x1", "mi210x2"]:
            gpu_provider = "amd"
        elif preset.startswith("gpu") or preset.startswith("H100"):
            gpu_provider = "nvidia"
        else:
            gpu_provider = "cpu"

        # 5) For each model, measure throughput & append results to CSV
        for model_name in test_models:
            print(f"\n=== Measuring throughput on {preset} with {model_name} ===")
            tps = measure_throughput(
                api_url=url,
                model_name=model_name,
                prompt="Summarize the following text about machine learning...",
                num_requests=3,
                max_tokens=64,
                is_chat=False,
                api_key=None,
                verbose=True
            )

            # Immediately append to CSV
            append_csv_row(preset, model_name, gpu_provider, tps)
            # Also store in memory for final chart
            results_in_memory.append((preset, model_name, gpu_provider, tps))

    # 6) Create bar chart from in-memory results
    if not results_in_memory:
        print("\nNo results to chart, exiting.")
        return

    print("\nCreating bar chart from in-memory results...")

    # Unique presets in order of first appearance
    seen_presets = []
    for r in results_in_memory:
        if r[0] not in seen_presets:
            seen_presets.append(r[0])

    # Unique models in order of first appearance
    seen_models = []
    for r in results_in_memory:
        if r[1] not in seen_models:
            seen_models.append(r[1])

    # Build data dict: data[(preset, model)] = tps
    data_dict = {}
    for (preset, model, provider, tps) in results_in_memory:
        data_dict[(preset, model)] = tps

    fig, ax = plt.subplots(figsize=(12,6))
    x_range = range(len(seen_presets))

    # If multiple models, we do grouped bars
    num_models = len(seen_models)
    if num_models <= 1:
        bar_width = 0.5
        offsets = [0]
    else:
        bar_width = 0.8 / num_models
        offsets = [(-0.4 + (i * bar_width)) for i in range(num_models)]

    for i, model_name in enumerate(seen_models):
        x_positions = []
        heights = []
        for j, preset in enumerate(seen_presets):
            x_positions.append(j + offsets[i])
            tps_val = data_dict.get((preset, model_name), 0)
            heights.append(tps_val)
        ax.bar(x_positions, heights, width=bar_width, label=model_name)

    ax.set_xticks(range(len(seen_presets)))
    ax.set_xticklabels(seen_presets, rotation=45, ha="right")
    ax.set_ylabel("Tokens/second (avg)")
    ax.set_title("vLLM Throughput Comparison")
    ax.legend()
    plt.tight_layout()
    chart_filename = "vllm_throughput_comparison.png"
    plt.savefig(chart_filename)
    print(f"Chart saved to {chart_filename}.")

    print("\n=== Final Results ===")
    for row in results_in_memory:
        p, m, gp, t = row
        print(f"Preset={p}, Model={m}, GPU={gp}, TPS={t:.2f}")


if __name__ == "__main__":
    main()
