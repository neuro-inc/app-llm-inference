import subprocess
import time
import requests
import csv
import os
import re
import argparse
import threading
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple

################################################################################
# Configuration
################################################################################

CSV_FILENAME = "vllm_benchmark_results.csv"

# Default GPU presets
GPU_PRESETS = [
    "gpu-small",
    "gpu-medium",
    "gpu-large",
    "gpu-xlarge",
    "mi210x1",
    "mi210x2",
    "H100X1",
    "H100X2",
]

# Default models
TEST_MODELS = [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Llama-3.1-8B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

MODEL_VRAM_REQ = {
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 15,
    "meta-llama/Llama-3.1-8B-Instruct": 17,
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 68,
}

PRESET_GPU_COUNT = {
    "gpu-small": 1,
    "gpu-medium": 2,
    "gpu-large": 4,
    "gpu-xlarge": 8,
    "mi210x1": 1,
    "mi210x2": 2,
    "H100X1": 1,
    "H100X2": 2,
}
PRESET_VRAM_PER_GPU = {
    "gpu-small": 16,
    "gpu-medium": 16,
    "gpu-large": 16,
    "gpu-xlarge": 16,
    "mi210x1": 64,
    "mi210x2": 64,
    "H100X1": 80,
    "H100X2": 80,
}


# Domain pattern
DOMAIN_FORMAT = "https://app-apolo--taddeus--{}.apps.novoserve.org.neu.ro"

# Insert your real HF token
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
GIT_BRANCH = "MLO-12-config-matrix"

# We'll measure these 7 metrics in the background collector:
# 1) avg_prompt_throughput_toks_per_s  (derived from prompt_tokens_total)
# 2) avg_generation_throughput_toks_per_s (derived from generation_tokens_total)
# 3) num_requests_running
# 4) num_requests_swapped
# 5) num_requests_waiting
# 6) gpu_cache_usage_perc
# 7) cpu_cache_usage_perc
#
# We will add an 8th metric to capture the average "response tokens/s" from
# actual completions (counted directly in the requests):
# 8) avg_response_tokens_per_s
METRIC_NAMES = [
    "avg_prompt_throughput_toks_per_s",
    "avg_generation_throughput_toks_per_s",
    "num_requests_running",
    "num_requests_swapped",
    "num_requests_waiting",
    "gpu_cache_usage_perc",
    "cpu_cache_usage_perc",
    "avg_response_tokens_per_s",   # <-- new metric
]

################################################################################
# CSV and Results
################################################################################

def append_csv_row(
    preset: str,
    model_name: str,
    # the 7 + 1 new averaged metrics
    avg_metrics: Dict[str,float],
    # new columns for latency and error count
    avg_latency: float,
    error_count: int
):
    """
    Appends row:
      [preset, model, promptTPS, genTPS, running, swapped, pending,
       gpuCache, cpuCache, avg_latency, errors, resp_tps]
    to the CSV.

    Note: We add `resp_tps` as a new last column so we don't break existing logic.
    """
    row_data = [
        preset,
        model_name,
        f"{avg_metrics['avg_prompt_throughput_toks_per_s']:.2f}",
        f"{avg_metrics['avg_generation_throughput_toks_per_s']:.2f}",
        f"{avg_metrics['num_requests_running']:.2f}",
        f"{avg_metrics['num_requests_swapped']:.2f}",
        f"{avg_metrics['num_requests_waiting']:.2f}",
        f"{avg_metrics['gpu_cache_usage_perc']:.2f}",
        f"{avg_metrics['cpu_cache_usage_perc']:.2f}",
        f"{avg_latency:.4f}",
        f"{error_count}",
        f"{avg_metrics['avg_response_tokens_per_s']:.2f}",  # new column at the end
    ]

    file_exists = os.path.exists(CSV_FILENAME)
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "preset", "model",
                "prompt_tps", "gen_tps",
                "running", "swapped", "pending",
                "gpu_cache_percent", "cpu_cache_percent",
                "avg_latency_s", "errors",
                "resp_tps",  # new header for the new column
            ])
        writer.writerow(row_data)

################################################################################
# Building & Deploy
################################################################################

def build_apolo_deploy_command(preset: str, model_hf_name: str) -> List[str]:
    # GPU provider
    gpu_provider = "amd" if preset.startswith("mi210") else "nvidia"
    n_gpus = PRESET_GPU_COUNT.get(preset, 1)

    def escape_commas(s: str) -> str:
        return s.replace(",", "\\,")

    device_list_unescaped = ",".join(str(i) for i in range(n_gpus))
    device_list_escaped = escape_commas(device_list_unescaped)

    env_sets = []
    if gpu_provider == "amd":
        env_sets.append(f'--set "envAmd.HIP_VISIBLE_DEVICES={device_list_escaped}"')
        env_sets.append(f'--set "envAmd.ROCR_VISIBLE_DEVICES={device_list_escaped}"')
    else:
        env_sets.append(f'--set "envNvidia.CUDA_VISIBLE_DEVICES={device_list_escaped}"')

    server_extra_args = []
    if n_gpus > 1:
        server_extra_args.append(f'--tensor-parallel-size={n_gpus}')
    # server_extra_args.append('--dtype=half')
    server_extra_args.append('--max-model-len=128000')
    # server_extra_args.append('--enforce-eager')
    # server_extra_args.append('--trust-remote-code')
   
    server_arg_sets = []
    for i, val in enumerate(server_extra_args):
        server_arg_sets.append(f'--set "serverExtraArgs[{i}]={val}"')

    base_cmd = [
        "apolo",
        "run",
        "--pass-config",
        "ghcr.io/neuro-inc/app-deployment",
        "--",
        "install",
        "https://github.com/neuro-inc/app-llm-inference",
        "llm-inference",
        preset.lower(),
        "charts/llm-inference-app",
        "--timeout=15m",
        # f"--git-branch={GIT_BRANCH}",
        f'--set "preset_name={preset}"',
        f'--set "gpuProvider={gpu_provider}"',
        f'--set "model.modelHFName={model_hf_name}"',
        '--set "model.modelRevision=main"',
        f'--set "env.HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_TOKEN}"',
        '--set "ingress.enabled=true"',
        '--set "ingress.clusterName=novoserve"',
    ]
    base_cmd.extend(env_sets)
    base_cmd.extend(server_arg_sets)
    return base_cmd

def deploy_model_on_preset(preset: str, model_hf_name: str) -> Optional[str]:
    cmd_list = build_apolo_deploy_command(preset, model_hf_name)
    cmd_str = " ".join(cmd_list)
    print(f"\n[DEPLOY] Preset={preset}, Model={model_hf_name}\nCMD:\n  {cmd_str}\n")

    try:
        result = subprocess.run(cmd_str, shell=True, check=False, text=True, capture_output=True)
    except Exception as e:
        print(f"[ERROR] apolo run => {e}")
        return None
    if result.returncode != 0:
        print(f"[ERROR] apolo command returned code {result.returncode}")
        print("[STDOUT]\n", result.stdout)
        print("[STDERR]\n", result.stderr)
        return None

    job_id = None
    lines = result.stdout.splitlines()
    job_id_pattern = re.compile(r"Job ID:\s+(job-[a-z0-9-]+)")
    for line in lines:
        match = job_id_pattern.search(line)
        if match:
            job_id = match.group(1)
            break

    if job_id:
        print(f"[INFO] Found job_id={job_id}")
    else:
        print("[WARN] Could not parse job id from stdout.")
        print(result.stdout)

    return job_id

def delete_namespace_for_preset(preset: str):
    ns_name = f"app-apolo--taddeus--{preset.lower()}"
    cmd = f"kubectl delete namespace {ns_name}"
    # Wait a bit before deleting
    time.sleep(5*60)
    print(f"[CLEANUP] {cmd}")
    subprocess.run(cmd, shell=True, check=False)

################################################################################
# Wait for readiness
################################################################################

def wait_for_endpoint(preset: str, max_wait_seconds: int = 1800) -> bool:
    """
    We'll check every 5s, up to max_wait_seconds=30min.
    If /v1/models 200 => online
    """
    base_url = DOMAIN_FORMAT.format(preset.lower())
    url = base_url + "/v1/models"
    start = time.time()
    while time.time() - start < max_wait_seconds:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"[READY] {url} responded 200 => online.")
                return True
        except requests.RequestException:
            pass
        print(f"[WAIT] Checking {url} ...")
        time.sleep(5)
    print(f"[TIMEOUT] {url} not online in {max_wait_seconds/60:.1f} min.")
    return False

################################################################################
# Metric Polling: compute throughput from counters
################################################################################

# We'll store the previous prompt/generation counters so we can compute tokens/s
# Key is a Tuple[str, str] = (preset, model_name).
# Value is a Dict[str,float] with "prompt_total", "generation_total", "time".
last_counters: Dict[Tuple[str, str], Dict[str, float]] = {}

def parse_realtime_metrics(metrics_text: str, preset: str, model_hf_name: str) -> Dict[str,float]:
    """
    We'll parse from the text lines for:
      vllm:prompt_tokens_total
      vllm:generation_tokens_total
      vllm:num_requests_running
      vllm:num_requests_swapped
      vllm:num_requests_waiting
      vllm:gpu_cache_usage_perc
      vllm:cpu_cache_usage_perc

    Then we'll compute:
      avg_prompt_throughput_toks_per_s  = (delta prompt tokens) / (delta time)
      avg_generation_throughput_toks_per_s = ...
      [We always set avg_response_tokens_per_s = 0.0 here; it's updated separately.]

    Return a dict with all final metrics.
    """
    model_tag = f'model_name="{model_hf_name}"'
    now = time.time()

    # Start with zero (and 0 for the new metric placeholder)
    result = {
        "avg_prompt_throughput_toks_per_s": 0.0,
        "avg_generation_throughput_toks_per_s": 0.0,
        "num_requests_running": 0.0,
        "num_requests_swapped": 0.0,
        "num_requests_waiting": 0.0,
        "gpu_cache_usage_perc": 0.0,
        "cpu_cache_usage_perc": 0.0,
        "avg_response_tokens_per_s": 0.0,  # new metric placeholder
    }

    prompt_total = None
    generation_total = None

    for line in metrics_text.splitlines():
        line = line.strip()
        if model_tag not in line:
            continue

        if line.startswith('vllm:prompt_tokens_total{'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    prompt_total = float(parts[1])
                except ValueError:
                    pass

        elif line.startswith('vllm:generation_tokens_total{'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    generation_total = float(parts[1])
                except ValueError:
                    pass

        elif line.startswith('vllm:num_requests_running{'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    result["num_requests_running"] = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith('vllm:num_requests_swapped{'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    result["num_requests_swapped"] = float(parts[1])
                except ValueError:
                    pass
        elif line.startswith('vllm:num_requests_waiting{'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    result["num_requests_waiting"] = float(parts[1])
                except ValueError:
                    pass

        elif line.startswith('vllm:gpu_cache_usage_perc{'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    # 0..1 => multiply by 100
                    result["gpu_cache_usage_perc"] = float(parts[1]) * 100.0
                except ValueError:
                    pass

        elif line.startswith('vllm:cpu_cache_usage_perc{'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    result["cpu_cache_usage_perc"] = float(parts[1]) * 100.0
                except ValueError:
                    pass

    # Now compute throughput from counters
    key = (preset, model_hf_name)
    old_data = last_counters.get(key, None)
    if prompt_total is not None and generation_total is not None:
        if old_data is not None:
            dt = now - old_data["time"]
            if dt > 0:
                dprompt = prompt_total - old_data["prompt_total"]
                dgen = generation_total - old_data["generation_total"]
                if dprompt < 0:
                    dprompt = 0
                if dgen < 0:
                    dgen = 0
                result["avg_prompt_throughput_toks_per_s"] = dprompt / dt
                result["avg_generation_throughput_toks_per_s"] = dgen / dt

        # Update last_counters with new values
        last_counters[key] = {
            "time": now,
            "prompt_total": prompt_total,
            "generation_total": generation_total
        }

    return result

def fetch_realtime_metrics(preset: str, model_hf_name: str) -> Dict[str,float]:
    base_url = DOMAIN_FORMAT.format(preset.lower())
    url = base_url + "/metrics"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return parse_realtime_metrics(r.text, preset, model_hf_name)
        else:
            print(f"[WARN] /metrics => {r.status_code}")
            return {k:0.0 for k in METRIC_NAMES}
    except requests.RequestException as e:
        print(f"[ERROR] /metrics => {e}")
        return {k:0.0 for k in METRIC_NAMES}

collected_samples : List[Dict[str,float]] = []
metric_stop_event = threading.Event()

# Globals to track actual response tokens
response_tokens_global = 0
response_tokens_lock = threading.Lock()
last_response_tokens_data = {"time": time.time(), "count": 0}

def background_collector_thread(preset: str, model_name: str, poll_interval: float):
    global collected_samples
    while not metric_stop_event.is_set():
        data = fetch_realtime_metrics(preset, model_name)

        # Compute "avg_response_tokens_per_s" from the difference of the global counter
        now = time.time()
        with response_tokens_lock:
            current_count = response_tokens_global
        delta_t = now - last_response_tokens_data["time"]
        if delta_t > 0:
            delta_tokens = current_count - last_response_tokens_data["count"]
            data["avg_response_tokens_per_s"] = delta_tokens / delta_t
        # Update "last_response_tokens_data"
        last_response_tokens_data["time"] = now
        last_response_tokens_data["count"] = current_count

        collected_samples.append(data)
        print("[METRICS]",
              f"Prompt TPS: {data['avg_prompt_throughput_toks_per_s']:.1f}, "
              f"Gen TPS: {data['avg_generation_throughput_toks_per_s']:.1f}, "
              f"Running: {data['num_requests_running']:.0f}, "
              f"Swapped: {data['num_requests_swapped']:.0f}, "
              f"Pending: {data['num_requests_waiting']:.0f}, "
              f"GPU Cache: {data['gpu_cache_usage_perc']:.1f}%, "
              f"CPU Cache: {data['cpu_cache_usage_perc']:.1f}%, "
              f"Resp TPS: {data['avg_response_tokens_per_s']:.1f}")
        time.sleep(poll_interval)

################################################################################
# Parallel requests
################################################################################

request_latencies = []
request_latencies_lock = threading.Lock()
error_count = 0
error_count_lock = threading.Lock()

def single_request_blocking(completions_url: str, model_name: str):
    global error_count, response_tokens_global
    start_t = time.time()
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": "Lets explore some architecture patterns for microservices",
        "temperature": 0.7,
        "max_tokens": 4000
    }
    try:
        r = requests.post(completions_url, json=payload, headers=headers, timeout=300)
        elapsed = time.time() - start_t
        if r.status_code == 200:
            # Count tokens from the response
            try:
                data = r.json()
                print(data)
                # If usage-style data is present:
                usage = data.get("usage", {})
                if "completion_tokens" in usage:
                    resp_count = usage["completion_tokens"]
                else:
                    # fallback: naive token count from first choice
                    choices = data.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        resp_count = len(text.split())
                    else:
                        resp_count = 0
                with response_tokens_lock:
                    response_tokens_global += resp_count
            except Exception:
                pass

            with request_latencies_lock:
                request_latencies.append(elapsed)
        else:
            with error_count_lock:
                error_count += 1
            print(f"[WARN] {completions_url} => {r.status_code}, {r.text[:200]}")
    except requests.RequestException as e:
        with error_count_lock:
            error_count += 1
        print(f"[ERROR] single_request => {e}")

def run_load_test(preset: str, model_name: str, num_requests: int, concurrency: int):
    base_url = DOMAIN_FORMAT.format(preset.lower())
    completions_url = base_url + "/v1/completions"

    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = []
        for _ in range(num_requests):
            futs.append(pool.submit(single_request_blocking, completions_url, model_name))
        for f in as_completed(futs):
            pass
    end = time.time()
    print(f"[LOADTEST] {num_requests} requests at concurrency={concurrency} => {end-start:.2f}s total")

################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Deploy each (preset, model), measure metrics while we send requests.")
    parser.add_argument("--preset",
                        type=str,
                        default=None,
                        help="If set, only run that single preset.")
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="If set, only run that single model.")
    parser.add_argument("--resume",
                        action="store_true",
                        help="Skip combos found in CSV.")
    parser.add_argument("--num-requests",
                        type=int,
                        default=100,
                        help="Total requests to send in parallel.")
    parser.add_argument("--concurrency",
                        type=int,
                        default=1,
                        help="Number of parallel requests.")
    parser.add_argument("--poll-interval",
                        type=float,
                        default=1.0,
                        help="How often to poll /metrics in seconds.")
    args = parser.parse_args()

    if args.preset:
        if args.preset not in GPU_PRESETS:
            print(f"[ERROR] Unknown preset {args.preset}. Must be in {GPU_PRESETS}")
            return
        selected_presets = [args.preset]
    else:
        selected_presets = GPU_PRESETS[:]

    if args.model:
        if args.model not in TEST_MODELS:
            print(f"[ERROR] Unknown model {args.model}, must be in {TEST_MODELS}")
            return
        selected_models = [args.model]
    else:
        selected_models = TEST_MODELS[:]

    # Load existing combos if --resume
    existing_combos = set()
    if args.resume and os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, "r", encoding="utf-8") as f:
            rd = csv.reader(f)
            hdr = next(rd, None)
            if hdr and len(hdr) >= 11:
                for row in rd:
                    if len(row) < 11:
                        continue
                    p, m = row[0], row[1]
                    existing_combos.add((p,m))

    for preset in selected_presets:
        ng = PRESET_GPU_COUNT.get(preset, 1)
        vram = PRESET_VRAM_PER_GPU.get(preset, 0)
        total_vram = ng * vram

        for model_name in selected_models:
            if (preset, model_name) in existing_combos:
                print(f"[SKIP] Already in CSV => {preset}/{model_name}")
                continue

            req_vram = MODEL_VRAM_REQ.get(model_name, 0)
            # if total_vram < req_vram:
            #     print(f"[SKIP] {preset} has {total_vram}GB, model needs {req_vram}GB")
            #     continue

            print(f"\n=== Deploying {model_name} on {preset} ===")
            job_id = deploy_model_on_preset(preset, model_name)
            if not job_id:
                print("[ERROR] Deployment failed, skipping.")
                continue

            ready = wait_for_endpoint(preset, max_wait_seconds=300)
            if not ready:
                print(f"[{preset}/{model_name}] Not online => Cleanup.")
                delete_namespace_for_preset(preset)
                continue

            global collected_samples, metric_stop_event
            global request_latencies, error_count
            global last_counters
            global response_tokens_global, last_response_tokens_data

            collected_samples = []
            request_latencies = []
            error_count = 0
            metric_stop_event = threading.Event()

            # Initialize the "last_counters" so we can measure tokens/s
            last_counters[(preset, model_name)] = {
                "time": time.time(),
                "prompt_total": 0.0,
                "generation_total": 0.0,
            }

            # Also reset response tokens tracking
            response_tokens_global = 0
            last_response_tokens_data = {"time": time.time(), "count": 0}

            # Start background collector
            collector = threading.Thread(
                target=background_collector_thread,
                args=(preset, model_name, args.poll_interval),
                daemon=True
            )
            collector.start()

            # Send load test
            run_load_test(preset, model_name, args.num_requests, args.concurrency)

            # Wait for all requests to finalize
            print("[INFO] Waiting for all requests to finalize in vLLM...")
            while True:
                last_metrics = fetch_realtime_metrics(preset, model_name)
                print("[METRICS]",
                      f"Prompt TPS: {last_metrics['avg_prompt_throughput_toks_per_s']:.1f}, "
                      f"Gen TPS: {last_metrics['avg_generation_throughput_toks_per_s']:.1f}, "
                      f"Running: {last_metrics['num_requests_running']:.0f}, "
                      f"Swapped: {last_metrics['num_requests_swapped']:.0f}, "
                      f"Pending: {last_metrics['num_requests_waiting']:.0f}, "
                      f"GPU Cache: {last_metrics['gpu_cache_usage_perc']:.1f}%, "
                      f"CPU Cache: {last_metrics['cpu_cache_usage_perc']:.1f}%.")

                total_pending = (last_metrics["num_requests_running"] +
                                 last_metrics["num_requests_swapped"] +
                                 last_metrics["num_requests_waiting"])
                if total_pending == 0:
                    break
                time.sleep(args.poll_interval)

            # Stop collector
            metric_stop_event.set()
            collector.join(timeout=10.0)

            # Compute average from all samples
            if len(collected_samples) == 0:
                print("[WARN] No samples collected => 0 for everything.")
                avg_result = {mn:0.0 for mn in METRIC_NAMES}
            else:
                sum_vals = {mn:0.0 for mn in METRIC_NAMES}
                for sample in collected_samples:
                    for mn in METRIC_NAMES:
                        sum_vals[mn] += sample[mn]
                n = len(collected_samples)
                avg_result = {mn: (sum_vals[mn]/n) for mn in METRIC_NAMES}

            # Compute average latency
            if len(request_latencies) > 0:
                avg_latency = sum(request_latencies) / len(request_latencies)
            else:
                avg_latency = 0.0

            # Write CSV
            append_csv_row(preset, model_name, avg_result, avg_latency, error_count)

            # Cleanup
            delete_namespace_for_preset(preset)

    # After all combos, produce bar charts (7 existing + 2 new).
    if not os.path.exists(CSV_FILENAME):
        print("\nNo CSV => no charts.")
        return

    results = []
    with open(CSV_FILENAME, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd, None)
        if not hdr or len(hdr) < 11:
            print("[WARN] CSV missing columns, skipping chart generation.")
            return
        # columns =>
        # [preset, model, prompt_tps, gen_tps, running, swapped, pending, gpu_cache%, cpu_cache%, avg_latency, errors, resp_tps]
        for row in rd:
            if len(row) < 11:
                continue
            p, m = row[0], row[1]
            try:
                prompt_tps = float(row[2])
                gen_tps = float(row[3])
                running = float(row[4])
                swapped = float(row[5])
                pending = float(row[6])
                gpu_cache = float(row[7])
                cpu_cache = float(row[8])
                avg_lat = float(row[9])
                errs = float(row[10])
                # We'll ignore extra columns for charting if present (like resp_tps).
            except ValueError:
                continue
            results.append((p,m,prompt_tps,gen_tps,running,swapped,pending,gpu_cache,cpu_cache,avg_lat,errs))

    if not results:
        print("No rows, skipping chart.")
        return

    # We'll separate them into 7 data groups + 2 new columns
    seen_presets = []
    seen_models = []
    for row in results:
        if row[0] not in seen_presets:
            seen_presets.append(row[0])
        if row[1] not in seen_models:
            seen_models.append(row[1])

    data_prompt = {}
    data_gen = {}
    data_run = {}
    data_swap = {}
    data_pending = {}
    data_gpu = {}
    data_cpu = {}
    data_latency = {}
    data_errors = {}

    for (p,m,pr,gn,run_,sw,pend,gpu,cp,lat,errs) in results:
        data_prompt[(p,m)] = pr
        data_gen[(p,m)] = gn
        data_run[(p,m)] = run_
        data_swap[(p,m)] = sw
        data_pending[(p,m)] = pend
        data_gpu[(p,m)] = gpu
        data_cpu[(p,m)] = cp
        data_latency[(p,m)] = lat
        data_errors[(p,m)] = errs

    import numpy as np

    def make_bar_chart(metric_key: str, data_dict, metric_title: str, y_label: str, out_filename: str):
        x_presets = seen_presets[:]
        x_indices = np.arange(len(x_presets))
        if len(seen_models) == 1:
            bar_width = 0.4
        else:
            bar_width = 0.15

        fig, ax = plt.subplots(figsize=(12,6))
        for i, modn in enumerate(seen_models):
            offsets = (i - (len(seen_models)-1)/2.0) * bar_width
            yvals = [data_dict.get((p, modn), 0.0) for p in x_presets]
            ax.bar(x_indices + offsets, yvals, width=bar_width, label=modn)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_presets, rotation=45, ha="right")
        ax.set_ylabel(y_label)
        ax.set_title(metric_title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_filename)
        print(f"Saved chart => {out_filename}")
        plt.close(fig)

    # 7 original metrics
    make_bar_chart("prompt_tps", data_prompt,
                   "Avg Prompt Throughput", "prompt tokens/s",
                   "chart_avg_prompt_throughput.png")
    make_bar_chart("gen_tps", data_gen,
                   "Avg Generation Throughput", "gen tokens/s",
                   "chart_avg_generation_throughput.png")
    make_bar_chart("running", data_run,
                   "Avg #Requests Running on GPU", "requests",
                   "chart_avg_requests_running.png")
    make_bar_chart("swapped", data_swap,
                   "Avg #Requests Swapped to CPU", "requests",
                   "chart_avg_requests_swapped.png")
    make_bar_chart("pending", data_pending,
                   "Avg #Requests Pending in Queue", "requests",
                   "chart_avg_requests_pending.png")
    make_bar_chart("gpu_cache", data_gpu,
                   "Avg GPU KV-cache usage", "percent",
                   "chart_gpu_cache_usage.png")
    make_bar_chart("cpu_cache", data_cpu,
                   "Avg CPU KV-cache usage", "percent",
                   "chart_cpu_cache_usage.png")

    # 2 new columns (avg_latency, errors)
    make_bar_chart("avg_latency", data_latency,
                   "Average Latency per Request", "seconds",
                   "chart_avg_latency.png")
    make_bar_chart("errors", data_errors,
                   "Total Errors", "count",
                   "chart_errors.png")

    print("\n=== Final Results from CSV ===")
    for row in results:
        p, m, pr, gn, run_, sw, pend, gpu, cp, lat, errs = row
        print(f"{p}/{m}: promptTPS={pr:.2f}, genTPS={gn:.2f}, "
              f"run={run_:.2f}, swap={sw:.2f}, pend={pend:.2f}, "
              f"GPU={gpu:.2f}%, CPU={cp:.2f}%, avg_latency={lat:.4f}s, errors={errs:.0f}")

if __name__ == "__main__":
    main()
