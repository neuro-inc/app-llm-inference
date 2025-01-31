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
from typing import Optional, List, Dict

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
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

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
MODEL_VRAM_REQ = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 15,
    # "meta-llama/Llama-3.1-8B-Instruct": 17,
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 68,
}

# Domain pattern
DOMAIN_FORMAT = "https://app-apolo--taddeus--{}.apps.novoserve.org.neu.ro"

# Insert your real HF token
HUGGING_FACE_TOKEN = ""
GIT_BRANCH = "MLO-12-config-matrix"

# We'll measure these 7 metrics in the background collector:
# 1) avg_prompt_throughput_toks_per_s
# 2) avg_generation_throughput_toks_per_s
# 3) num_requests_running
# 4) num_requests_swapped
# 5) num_requests_waiting
# 6) gpu_cache_usage_perc
# 7) cpu_cache_usage_perc

METRIC_NAMES = [
    "avg_prompt_throughput_toks_per_s",
    "avg_generation_throughput_toks_per_s",
    "num_requests_running",
    "num_requests_swapped",
    "num_requests_waiting",
    "gpu_cache_usage_perc",
    "cpu_cache_usage_perc",
]

################################################################################
# CSV and Results
################################################################################

def append_csv_row(
    preset: str,
    model_name: str,
    # the 7 averaged metrics
    avg_metrics: Dict[str,float],
):
    """
    Appends row:
      [preset, model, promptTPS, genTPS, running, swapped, pending, gpuCache, cpuCache]
    to the CSV.
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
    ]

    file_exists = os.path.exists(CSV_FILENAME)
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "preset", "model",
                "prompt_tps", "gen_tps",
                "running", "swapped", "pending",
                "gpu_cache_percent", "cpu_cache_percent"
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
        env_sets.append(f'--set "env.CUDA_VISIBLE_DEVICES={device_list_escaped}"')

    server_extra_args = []
    if n_gpus > 1:
        server_extra_args.append(f'--pipeline-parallel-size={n_gpus}')
        # server_extra_args.append(f'--tensor-parallel-size={n_gpus}')
    if gpu_provider == "nvidia":
        server_extra_args.append('--dtype=half')
    server_extra_args.append('--max-model-len=131072')

    server_arg_sets = []
    for i, val in enumerate(server_extra_args):
        server_arg_sets.append(f'--set "serverExtraArgs[{i}]={val}"')

    base_cmd = [
        "apolo",
        "run",
        "--pass-config",
        "ghcr.io/neuro-inc/app-deployment:development",
        "--",
        "install",
        "https://github.com/neuro-inc/app-llm-inference",
        "llm-inference",
        preset.lower(),
        "charts/llm-inference-app",
        "--timeout=30m",  # up to 30 min in case some presets are slow to start
        f"--git-branch={GIT_BRANCH}",
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

    # Attempt to see if job is started
    if job_id:
        print(f"[INFO] Found job_id={job_id}")
    else:
        print("[WARN] Could not parse job id from stdout.")
        print(result.stdout)

    # Additional check: see if we have an 'ERROR' or 'Failed' in stderr
    # If so, we might consider returning None. But let's skip for now.

    return job_id

def delete_namespace_for_preset(preset: str):
    ns_name = f"app-apolo--taddeus--{preset.lower()}"
    cmd = f"kubectl delete namespace {ns_name}"
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
        # we can also attempt a small test request or so
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
# Metric Polling
################################################################################

def parse_realtime_metrics(metrics_text: str, model_hf_name: str) -> Dict[str,float]:
    """
    For the model's lines, parse out:
      vllm:avg_prompt_throughput_toks_per_s
      vllm:avg_generation_throughput_toks_per_s
      vllm:num_requests_running
      vllm:num_requests_swapped
      vllm:num_requests_waiting
      vllm:gpu_cache_usage_perc
      vllm:cpu_cache_usage_perc
    Return dict with these keys. If any missing => 0.0
    """
    model_tag = f'model_name="{model_hf_name}"'
    vals = {
        "avg_prompt_throughput_toks_per_s": 0.0,
        "avg_generation_throughput_toks_per_s": 0.0,
        "num_requests_running": 0.0,
        "num_requests_swapped": 0.0,
        "num_requests_waiting": 0.0,
        "gpu_cache_usage_perc": 0.0,
        "cpu_cache_usage_perc": 0.0,
    }
    for line in metrics_text.splitlines():
        line = line.strip()
        if not model_tag in line:
            continue

        # example parse
        # vllm:avg_prompt_throughput_toks_per_s{model_name="foo/bar"} 123.4
        for k in vals.keys():
            metric_line_start = "vllm:"+k+"{"
            if line.startswith(metric_line_start) and model_tag in line:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        vals[k] = float(parts[1])
                    except ValueError:
                        pass
    return vals

def fetch_realtime_metrics(preset: str, model_hf_name: str) -> Dict[str,float]:
    """
    GET /metrics, parse relevant metrics for this model
    """
    base_url = DOMAIN_FORMAT.format(preset.lower())
    url = base_url + "/metrics"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return parse_realtime_metrics(r.text, model_hf_name)
        else:
            print(f"[WARN] /metrics => {r.status_code}")
            return {k:0.0 for k in METRIC_NAMES}
    except requests.RequestException as e:
        print(f"[ERROR] /metrics => {e}")
        return {k:0.0 for k in METRIC_NAMES}

collected_samples : List[Dict[str,float]] = []
metric_stop_event = threading.Event()

def background_collector_thread(preset: str, model_name: str, poll_interval: float):
    global collected_samples
    while not metric_stop_event.is_set():
        data = fetch_realtime_metrics(preset, model_name)
        collected_samples.append(data)
        # Example log
        print("[METRICS]", 
              f"Avg prompt throughput: {data['avg_prompt_throughput_toks_per_s']:.1f} tokens/s, "
              f"Avg generation throughput: {data['avg_generation_throughput_toks_per_s']:.1f} tokens/s, "
              f"Running: {data['num_requests_running']:.0f} reqs, "
              f"Swapped: {data['num_requests_swapped']:.0f} reqs, "
              f"Pending: {data['num_requests_waiting']:.0f} reqs, "
              f"GPU KV cache usage: {data['gpu_cache_usage_perc']:.1f}%, "
              f"CPU KV cache usage: {data['cpu_cache_usage_perc']:.1f}%.")

        time.sleep(poll_interval)

################################################################################
# Parallel requests
################################################################################

def single_request_blocking(completions_url: str, model_name: str):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": "Lets explore some architecture patterns for microservices",
        "max_tokens": 64,
        "temperature": 0.7
    }
    try:
        r = requests.post(completions_url, json=payload, headers=headers, timeout=300)
        if r.status_code != 200:
            print(f"[WARN] {completions_url} => {r.status_code}, {r.text[:200]}")
    except requests.RequestException as e:
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
                        default=500,
                        help="Total requests to send in parallel.")
    parser.add_argument("--concurrency",
                        type=int,
                        default=10,
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

    existing_combos = set()
    if args.resume and os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, "r", encoding="utf-8") as f:
            rd = csv.reader(f)
            hdr = next(rd, None)
            if hdr and len(hdr) >= 9:  # check columns
                for row in rd:
                    if len(row) < 9:
                        continue
                    p, m = row[0], row[1]
                    existing_combos.add((p,m))

    for preset in selected_presets:
        ng = PRESET_GPU_COUNT.get(preset, 1)
        vram = PRESET_VRAM_PER_GPU.get(preset, 0)
        total_vram = ng * vram

        for model_name in selected_models:
            if (preset, model_name) in existing_combos:
                print(f"[SKIP] CSV => {preset}/{model_name}")
                continue

            req_vram = MODEL_VRAM_REQ.get(model_name, 9999)
            if total_vram < req_vram:
                print(f"[SKIP] {preset} has {total_vram}GB, model needs {req_vram}GB")
                continue

            # Deploy
            print(f"\n=== Deploying {model_name} on {preset} ===")
            job_id = deploy_model_on_preset(preset, model_name)
            if not job_id:
                print("[ERROR] Deployment failed, skipping.")
                continue

            # Wait
            ready = wait_for_endpoint(preset, max_wait_seconds=1800)  # up to 30 min
            if not ready:
                print(f"[{preset}/{model_name}] Not online => Cleanup.")
                delete_namespace_for_preset(preset)
                continue

            # Start background collector
            global collected_samples
            collected_samples = []
            global metric_stop_event
            metric_stop_event = threading.Event()

            collector = threading.Thread(
                target=background_collector_thread,
                args=(preset, model_name, args.poll_interval),
                daemon=True
            )
            collector.start()

            # Run load test
            run_load_test(preset, model_name, args.num_requests, args.concurrency)

            # Stop collector
            metric_stop_event.set()
            collector.join(timeout=10.0)

            # Calculate average from all samples
            # we'll just do a simple mean for each metric
            if len(collected_samples) == 0:
                print("[WARN] No samples collected => might be 0 for everything.")
                avg_result = {mn:0.0 for mn in METRIC_NAMES}
            else:
                sum_vals = {mn:0.0 for mn in METRIC_NAMES}
                for sample in collected_samples:
                    for mn in METRIC_NAMES:
                        sum_vals[mn] += sample[mn]
                n = len(collected_samples)
                avg_result = {mn: (sum_vals[mn]/n) for mn in METRIC_NAMES}

            # Write CSV
            append_csv_row(preset, model_name, avg_result)

            # Cleanup
            delete_namespace_for_preset(preset)

    # After all combos, produce 7 bar charts (one per metric).
    if not os.path.exists(CSV_FILENAME):
        print("\nNo CSV => no charts.")
        return

    # parse CSV
    results = []
    with open(CSV_FILENAME, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd, None)
        if not hdr or len(hdr) < 9:
            print("[WARN] CSV missing columns, skipping chart.")
            return
        # columns => [preset, model, prompt_tps, gen_tps, running, swapped, pending, gpu_cache%, cpu_cache%]
        for row in rd:
            if len(row) < 9:
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
            except ValueError:
                continue
            results.append((p,m,prompt_tps,gen_tps,running,swapped,pending,gpu_cache,cpu_cache))

    if not results:
        print("No rows, skipping chart.")
        return

    # We'll produce 7 separate bar charts:
    #  1) prompt_tps
    #  2) gen_tps
    #  3) running
    #  4) swapped
    #  5) pending
    #  6) gpu_cache
    #  7) cpu_cache

    seen_presets = []
    seen_models = []
    for row in results:
        if row[0] not in seen_presets:
            seen_presets.append(row[0])
        if row[1] not in seen_models:
            seen_models.append(row[1])

    # We'll store them as metric -> dict[(preset, model)] = float
    # row => p,m, pr, gn, run, sw, pend, gpu, cpu
    # indexes =>    2    3   4    5    6    7    8
    data_prompt = {}
    data_gen = {}
    data_run = {}
    data_swap = {}
    data_pending = {}
    data_gpu = {}
    data_cpu = {}

    for (p,m,pr,gn,run,sw,pend,gpu,cp) in results:
        data_prompt[(p,m)] = pr
        data_gen[(p,m)] = gn
        data_run[(p,m)] = run
        data_swap[(p,m)] = sw
        data_pending[(p,m)] = pend
        data_gpu[(p,m)] = gpu
        data_cpu[(p,m)] = cp

    import numpy as np

    # We'll define a helper function to produce the bar chart for a single metric.
    def make_bar_chart(metric_key: str, data_dict, metric_title: str, y_label: str, out_filename: str):
        # data_dict => dict[(preset,model)] => float
        # We'll do a grouped bar approach
        x_presets = seen_presets[:]  # order as discovered
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

    # We'll produce 7 charts:
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

    print("\n=== Final Results from CSV ===")
    for row in results:
        p, m, pr, gn, run, sw, pend, gpu, cp = row
        print(f"{p}/{m}: promptTPS={pr:.2f}, genTPS={gn:.2f}, run={run:.2f}, swap={sw:.2f}, pend={pend:.2f}, GPU={gpu:.2f}%, CPU={cp:.2f}%")

if __name__ == "__main__":
    main()
