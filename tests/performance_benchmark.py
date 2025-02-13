#!/usr/bin/env python3

import subprocess
import time
import requests
import csv
import os
import re
import argparse
import threading
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple
import numpy as np

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

# Default models to test
TEST_MODELS = [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "meta-llama/Llama-3.2-3B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
]

MODEL_VRAM_REQ = {
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": 15,
    "meta-llama/Llama-3.2-3B-Instruct": 15,
    # "meta-llama/Llama-3.1-8B-Instruct": 17,
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

DOMAIN_FORMAT = "https://app-apolo--taddeus--{}.apps.novoserve.org.neu.ro"
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "YOUR_HF_TOKEN_HERE")

METRIC_NAMES = [
    "avg_prompt_throughput_toks_per_s",
    "avg_generation_throughput_toks_per_s",
    "num_requests_running",
    "num_requests_swapped",
    "num_requests_waiting",
    "gpu_cache_usage_perc",
    "cpu_cache_usage_perc",
    "avg_response_tokens_per_s",   # from the sum of actual completion tokens in all requests
]


class VLLMBenchmark:
    """
    This class supports two load-testing modes:
      1) 'sequential': Keep concurrency requests in flight; only start a new one when an old one finishes.
      2) 'parallel': Launch a batch of concurrency requests every batch_interval seconds (possibly overlapping).
    """
    def __init__(self):
        # Metric collection
        self.collected_samples: List[Dict[str, float]] = []
        self.metric_stop_event = threading.Event()

        # Per-request aggregates
        self.all_request_latencies: List[float] = []
        self.all_request_tps: List[float] = []
        self.error_count = 0

        # vLLM counters
        self.last_counters: Dict[Tuple[str, str], Dict[str, float]] = {}
        self.response_tokens_total = 0
        self.last_response_tokens_data = {"time": time.time(), "count": 0}

        # Thread-safety
        self.lock = threading.Lock()

        # We'll store references to threads if needed
        self.active_threads: List[threading.Thread] = []

    ############################################################################
    # CSV
    ############################################################################
    def append_csv_row(
        self,
        preset: str,
        model_name: str,
        avg_metrics: Dict[str, float],
        avg_latency: float,
        error_count: int,
        avg_request_tps: float
    ) -> None:
        """
        Appends a row to vllm_benchmark_results.csv:
         [preset, model, promptTPS, genTPS, running, swapped, pending,
          gpuCache, cpuCache, avg_latency, errors, resp_tps, request_level_TPS]
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
            f"{avg_metrics['avg_response_tokens_per_s']:.2f}",
            f"{avg_request_tps:.2f}",
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
                    "resp_tps",
                    "request_level_TPS",
                ])
            writer.writerow(row_data)

    ############################################################################
    # Deploy
    ############################################################################
    def build_apolo_deploy_command(self, preset: str, model_hf_name: str) -> List[str]:
        server_extra_args = ['--max-model-len=2048', "--dtype=half", "--enforce-eager", "--trust-remote-code"]
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
            "--set branch=benchmarks",
            f'--set "preset_name={preset}"',
            f'--set "model.modelHFName={model_hf_name}"',
            '--set "model.modelRevision=main"',
            f'--set "env.HUGGING_FACE_HUB_TOKEN={HUGGING_FACE_TOKEN}"',
            '--set "ingress.enabled=true"',
            '--set "ingress.clusterName=novoserve"',
        ]
        base_cmd.extend(server_arg_sets)
        return base_cmd

    def deploy_model_on_preset(self, preset: str, model_hf_name: str) -> Optional[str]:
        cmd_list = self.build_apolo_deploy_command(preset, model_hf_name)
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

    def delete_namespace_for_preset(self, preset: str) -> None:
        ns_name = f"app-apolo--taddeus--{preset.lower()}"
        cmd = f"kubectl delete namespace {ns_name}"
        time.sleep(5 * 60)
        print(f"[CLEANUP] {cmd}")
        subprocess.run(cmd, shell=True, check=False)

    ############################################################################
    # Readiness
    ############################################################################
    def wait_for_endpoint_recursively(
        self,
        preset: str,
        max_wait_seconds: int = 300,
        interval_seconds: int = 5,
        elapsed: int = 0
    ) -> bool:
        if elapsed >= max_wait_seconds:
            print(f"[TIMEOUT] {preset} not online in {max_wait_seconds} seconds.")
            return False

        base_url = DOMAIN_FORMAT.format(preset.lower())
        url = base_url + "/v1/models"
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"[READY] {url} responded 200 => online.")
                return True
        except requests.RequestException:
            pass

        print(f"[WAIT] Checking {url} again in {interval_seconds}s...")
        time.sleep(interval_seconds)
        return self.wait_for_endpoint_recursively(preset, max_wait_seconds, interval_seconds, elapsed + interval_seconds)

    ############################################################################
    # Metric Polling
    ############################################################################
    def parse_realtime_metrics(self, metrics_text: str, preset: str, model_hf_name: str) -> Dict[str, float]:
        model_tag = f'model_name="{model_hf_name}"'
        now = time.time()
        result = {
            "avg_prompt_throughput_toks_per_s": 0.0,
            "avg_generation_throughput_toks_per_s": 0.0,
            "num_requests_running": 0.0,
            "num_requests_swapped": 0.0,
            "num_requests_waiting": 0.0,
            "gpu_cache_usage_perc": 0.0,
            "cpu_cache_usage_perc": 0.0,
            "avg_response_tokens_per_s": 0.0,
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
                        val = float(parts[1]) * 100.0
                        result["gpu_cache_usage_perc"] = val
                    except ValueError:
                        pass

            elif line.startswith('vllm:cpu_cache_usage_perc{'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        val = float(parts[1]) * 100.0
                        result["cpu_cache_usage_perc"] = val
                    except ValueError:
                        pass

        key = (preset, model_hf_name)
        old_data = self.last_counters.get(key, None)

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

            self.last_counters[key] = {
                "time": now,
                "prompt_total": prompt_total,
                "generation_total": generation_total
            }

        return result

    def fetch_realtime_metrics(self, preset: str, model_hf_name: str) -> Dict[str, float]:
        base_url = DOMAIN_FORMAT.format(preset.lower())
        url = base_url + "/metrics"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return self.parse_realtime_metrics(r.text, preset, model_hf_name)
            else:
                print(f"[WARN] /metrics => {r.status_code}")
                return {k: 0.0 for k in METRIC_NAMES}
        except requests.RequestException as e:
            print(f"[ERROR] /metrics => {e}")
            return {k: 0.0 for k in METRIC_NAMES}

    def background_collector_thread(self, preset: str, model_name: str, poll_interval: float) -> None:
        while not self.metric_stop_event.is_set():
            data = self.fetch_realtime_metrics(preset, model_name)

            # measure average_response_tokens_per_s from the difference
            now = time.time()
            with self.lock:
                current_count = self.response_tokens_total
            delta_t = now - self.last_response_tokens_data["time"]
            if delta_t > 0:
                delta_tokens = current_count - self.last_response_tokens_data["count"]
                data["avg_response_tokens_per_s"] = delta_tokens / delta_t

            self.last_response_tokens_data["time"] = now
            self.last_response_tokens_data["count"] = current_count

            self.collected_samples.append(data)
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

    ############################################################################
    # Request Logic
    ############################################################################
    def process_one_request(self, completions_url: str, model_name: str) -> None:
        """
        Actually does the synchronous HTTP call,
        capturing tokens, latency, etc.
        """
        start_t = time.time()
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "prompt": """You are a helpful AI assistant.
            The user says: 'Explain the significance of Einstein's theory of relativity in simple terms.'
            Provide a concise but thorough answer.""",
            "max_tokens": 512,
            "temperature": 0.7
        }

        try:
            resp = requests.post(completions_url, json=payload, headers=headers, timeout=300)
            elapsed = time.time() - start_t
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    usage = data.get("usage", {})
                    if "completion_tokens" in usage:
                        resp_count = usage["completion_tokens"]
                    else:
                        choices = data.get("choices", [])
                        if choices:
                            text = choices[0].get("text", "")
                            resp_count = len(text.split())
                        else:
                            resp_count = 0

                    with self.lock:
                        self.response_tokens_total += resp_count

                    request_tps = (resp_count / elapsed) if elapsed > 0 else 0.0
                    with self.lock:
                        self.all_request_tps.append(request_tps)
                        self.all_request_latencies.append(elapsed)

                except Exception as ex:
                    print(f"[WARN] Could not parse JSON usage => {ex}")
                    with self.lock:
                        self.all_request_tps.append(0.0)
                        self.all_request_latencies.append(elapsed)
            else:
                with self.lock:
                    self.error_count += 1
                print(f"[WARN] => {resp.status_code}, {resp.text[:200]}")

        except requests.RequestException as e:
            with self.lock:
                self.error_count += 1
            print(f"[ERROR] => {e}")

    def run_load_test_sequential(
        self,
        completions_url: str,
        model_name: str,
        num_requests: int,
        concurrency: int
    ) -> None:
        """
        'Sequential mode': We maintain up to `concurrency` requests in flight.
        Once a request finishes, we start a new one, until we have launched `num_requests`.
        This effectively is a standard concurrency-limited approach.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"[LOADTEST:Sequential] Launching {num_requests} requests with concurrency={concurrency}.")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for _ in range(num_requests):
                futures.append(pool.submit(self.process_one_request, completions_url, model_name))

            # Wait for them all to finish
            for f in as_completed(futures):
                pass

        elapsed = time.time() - start_time
        print(f"[LOADTEST:Sequential] All {num_requests} requests done in {elapsed:.2f}s.")

    def run_load_test_parallel(
        self,
        completions_url: str,
        model_name: str,
        num_requests: int,
        concurrency: int,
        batch_interval: float
    ) -> None:
        """
        'Parallel mode': Launch a batch of `concurrency` requests every `batch_interval` seconds,
        potentially overlapping, until we've scheduled `num_requests`.
        """
        print(f"[LOADTEST:Parallel] Launching {num_requests} total requests in batches of {concurrency} "
              f"every {batch_interval}s, not waiting for them to finish.")
        requests_launched = 0
        start_time = time.time()

        while requests_launched < num_requests:
            remaining = num_requests - requests_launched
            batch_size = min(remaining, concurrency)

            # spawn batch_size threads at once
            threads = []
            for _ in range(batch_size):
                t = threading.Thread(
                    target=self.process_one_request,
                    args=(completions_url, model_name),
                    daemon=False
                )
                t.start()
                threads.append(t)

            with self.lock:
                self.active_threads.extend(threads)

            requests_launched += batch_size
            if requests_launched >= num_requests:
                break

            time.sleep(batch_interval)

        elapsed = time.time() - start_time
        print(f"[LOADTEST:Parallel] Done scheduling {requests_launched} requests in {elapsed:.2f}s. "
              "They may still be running in background threads.")

    ############################################################################
    # Main
    ############################################################################
    def main(self) -> None:
        parser = argparse.ArgumentParser(
            description="Benchmark vLLM with either 'sequential' concurrency-limited or 'parallel' batch mode."
        )
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
                            default=50,
                            help="Total requests to launch.")
        parser.add_argument("--concurrency",
                            type=int,
                            default=1,
                            help="Number of requests to keep in flight (sequential mode) or per batch (parallel mode).")
        parser.add_argument("--batch-interval",
                            type=float,
                            default=1.0,
                            help="Seconds to wait between each batch (only used in parallel mode).")
        parser.add_argument("--mode",
                            type=str,
                            default="sequential",
                            choices=["sequential", "parallel"],
                            help="Which load test mode to use.")
        parser.add_argument("--poll-interval",
                            type=float,
                            default=1.0,
                            help="How often to poll /metrics in seconds.")
        args = parser.parse_args()

        if args.preset:
            if args.preset not in GPU_PRESETS:
                raise ValueError(f"Unknown preset {args.preset}. Must be in {GPU_PRESETS}")
            selected_presets = [args.preset]
        else:
            selected_presets = GPU_PRESETS[:]

        if args.model:
            if args.model not in TEST_MODELS:
                raise ValueError(f"Unknown model {args.model}, must be in {TEST_MODELS}")
            selected_models = [args.model]
        else:
            selected_models = TEST_MODELS[:]

        existing_combos = set()
        if args.resume and os.path.exists(CSV_FILENAME):
            with open(CSV_FILENAME, "r", encoding="utf-8") as f:
                rd = csv.reader(f)
                hdr = next(rd, None)
                if hdr and len(hdr) >= 12:
                    for row in rd:
                        if len(row) < 12:
                            continue
                        p, m = row[0], row[1]
                        existing_combos.add((p, m))

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
                job_id = self.deploy_model_on_preset(preset, model_name)
                if not job_id:
                    print("[ERROR] Deployment failed => skipping.")
                    continue

                # Recursive readiness check
                ready = self.wait_for_endpoint_recursively(preset, max_wait_seconds=300, interval_seconds=5)
                if not ready:
                    print(f"[ERROR] {preset}/{model_name} not online => Cleanup and skip.")
                    self.delete_namespace_for_preset(preset)
                    continue

                # reset collectors
                self.collected_samples.clear()
                self.all_request_latencies.clear()
                self.all_request_tps.clear()
                self.error_count = 0
                self.metric_stop_event.clear()

                self.last_counters[(preset, model_name)] = {
                    "time": time.time(),
                    "prompt_total": 0.0,
                    "generation_total": 0.0,
                }
                self.response_tokens_total = 0
                self.last_response_tokens_data = {"time": time.time(), "count": 0}
                self.active_threads.clear()

                # Start background collector
                collector_thread = threading.Thread(
                    target=self.background_collector_thread,
                    args=(preset, model_name, args.poll_interval),
                    daemon=True
                )
                collector_thread.start()

                base_url = DOMAIN_FORMAT.format(preset.lower())
                completions_url = base_url + "/v1/completions"

                # MAIN LOAD TEST
                if args.mode == "sequential":
                    # Keep concurrency requests in flight until done
                    self.run_load_test_sequential(
                        completions_url=completions_url,
                        model_name=model_name,
                        num_requests=args.num_requests,
                        concurrency=args.concurrency
                    )
                else:
                    # "parallel": spawn concurrency requests every batch_interval
                    self.run_load_test_parallel(
                        completions_url=completions_url,
                        model_name=model_name,
                        num_requests=args.num_requests,
                        concurrency=args.concurrency,
                        batch_interval=args.batch_interval
                    )

                # optionally, if in parallel mode you do NOT want to wait => do nothing
                # or if in sequential we have already waited for them. Either way, let's
                # let the metric collector gather some data:
                time.sleep(5.0)

                # Stop collector
                self.metric_stop_event.set()
                collector_thread.join(timeout=10.0)

                # Summarize from collected samples
                if len(self.collected_samples) == 0:
                    print("[WARN] No metric samples collected => zero for everything.")
                    avg_result = {mn: 0.0 for mn in METRIC_NAMES}
                else:
                    sum_vals = {mn: 0.0 for mn in METRIC_NAMES}
                    for sample in self.collected_samples:
                        for mn in METRIC_NAMES:
                            sum_vals[mn] += sample[mn]
                    n = len(self.collected_samples)
                    avg_result = {mn: (sum_vals[mn] / n) for mn in METRIC_NAMES}

                # compute avg latency & request-level TPS
                with self.lock:
                    if self.all_request_latencies:
                        avg_latency = sum(self.all_request_latencies) / len(self.all_request_latencies)
                    else:
                        avg_latency = 0.0

                    if self.all_request_tps:
                        avg_request_tps = sum(self.all_request_tps) / len(self.all_request_tps)
                    else:
                        avg_request_tps = 0.0

                    local_err_count = self.error_count

                # Write CSV row
                self.append_csv_row(
                    preset=preset,
                    model_name=model_name,
                    avg_metrics=avg_result,
                    avg_latency=avg_latency,
                    error_count=local_err_count,
                    avg_request_tps=avg_request_tps
                )

                # Cleanup
                self.delete_namespace_for_preset(preset)

        # Chart generation if CSV is present
        if not os.path.exists(CSV_FILENAME):
            print("\nNo CSV => no charts.")
            return
        self.generate_charts()

    ############################################################################
    # Charting
    ############################################################################
    def generate_charts(self) -> None:
        if not os.path.exists(CSV_FILENAME):
            print("[WARN] CSV not found => skipping charts.")
            return

        rows = []
        with open(CSV_FILENAME, "r", encoding="utf-8") as f:
            rd = csv.reader(f)
            hdr = next(rd, None)
            if not hdr or len(hdr) < 12:
                print("[WARN] CSV missing columns, skipping chart generation.")
                return

            for row in rd:
                if len(row) < 12:
                    continue
                rows.append(row)

        if not rows:
            print("[WARN] No CSV data => skipping charts.")
            return

        # parse them
        data_parsed = []
        for row in rows:
            # row => [preset, model, promptTPS, genTPS, running, swapped, pending,
            #         gpu_cache_percent, cpu_cache_percent, avg_latency_s, errors, resp_tps, request_tps]
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
                resp_tps = float(row[11])
                req_tps = float(row[12]) if len(row) >= 13 else 0.0
            except ValueError:
                continue
            data_parsed.append((p, m, prompt_tps, gen_tps, running, swapped,
                                pending, gpu_cache, cpu_cache, avg_lat, errs, resp_tps, req_tps))

        # group by
        seen_presets = []
        seen_models = []
        for (p, m, *_rest) in data_parsed:
            if p not in seen_presets:
                seen_presets.append(p)
            if m not in seen_models:
                seen_models.append(m)

        # separate dict
        data_prompt = {}
        data_gen = {}
        data_run = {}
        data_swap = {}
        data_pending = {}
        data_gpu = {}
        data_cpu = {}
        data_latency = {}
        data_errors = {}
        data_resp_tps = {}
        data_request_tps = {}

        for (p, m, pr, gn, run_, sw, pend, gpu, cp, lat, errs, resp, rtps) in data_parsed:
            data_prompt[(p, m)] = pr
            data_gen[(p, m)] = gn
            data_run[(p, m)] = run_
            data_swap[(p, m)] = sw
            data_pending[(p, m)] = pend
            data_gpu[(p, m)] = gpu
            data_cpu[(p, m)] = cp
            data_latency[(p, m)] = lat
            data_errors[(p, m)] = errs
            data_resp_tps[(p, m)] = resp
            data_request_tps[(p, m)] = rtps

        def make_bar_chart(
            data_dict: Dict[Tuple[str, str], float],
            metric_title: str,
            y_label: str,
            out_filename: str
        ):
            x_presets = seen_presets[:]
            x_indices = np.arange(len(x_presets))
            if len(seen_models) == 1:
                bar_width = 0.4
            else:
                bar_width = 0.15

            fig, ax = plt.subplots(figsize=(12, 6))
            for i, modn in enumerate(seen_models):
                offsets = (i - (len(seen_models) - 1) / 2.0) * bar_width
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

        # produce charts
        make_bar_chart(data_prompt, "Avg Prompt Throughput", "prompt tokens/s",
                       "chart_avg_prompt_throughput.png")
        make_bar_chart(data_gen, "Avg Generation Throughput", "gen tokens/s",
                       "chart_avg_generation_throughput.png")
        make_bar_chart(data_run, "Avg #Requests Running on GPU", "requests",
                       "chart_avg_requests_running.png")
        make_bar_chart(data_swap, "Avg #Requests Swapped to CPU", "requests",
                       "chart_avg_requests_swapped.png")
        make_bar_chart(data_pending, "Avg #Requests Pending in Queue", "requests",
                       "chart_avg_requests_pending.png")
        make_bar_chart(data_gpu, "Avg GPU KV-cache usage", "percent",
                       "chart_gpu_cache_usage.png")
        make_bar_chart(data_cpu, "Avg CPU KV-cache usage", "percent",
                       "chart_cpu_cache_usage.png")
        make_bar_chart(data_latency, "Average Latency per Request", "seconds",
                       "chart_avg_latency.png")
        make_bar_chart(data_errors, "Total Errors", "count",
                       "chart_errors.png")
        make_bar_chart(data_resp_tps, "Avg Response TPS (server-level)", "tokens/s",
                       "chart_avg_resp_tokens_s.png")
        make_bar_chart(data_request_tps, "Per-Request TPS (averaged)", "tokens/s",
                       "chart_avg_request_tokens_s.png")

        print("\n=== Final Results from CSV ===")
        for row in data_parsed:
            (p, m, pr, gn, run_, sw, pend, gpu, cp, lat, errs, resp, rtps) = row
            print(f"{p}/{m}: promptTPS={pr:.2f}, genTPS={gn:.2f}, "
                  f"run={run_:.2f}, swap={sw:.2f}, pend={pend:.2f}, "
                  f"GPU={gpu:.2f}%, CPU={cp:.2f}%, avg_latency={lat:.4f}s, errors={errs}, "
                  f"respTPS={resp:.2f}, requestTPS={rtps:.2f}")


################################################################################
# Entry Point
################################################################################

if __name__ == "__main__":
    VLLMBenchmark().main()
