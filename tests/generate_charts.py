#!/usr/bin/env python3

import csv
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

################################################################################
# Configuration
################################################################################

# By default, we'll read the same CSV filename that the original script writes.
CSV_FILENAME = "vllm_benchmark_results.csv"

# We measure these columns:
#   [preset, model, prompt_tps, gen_tps, running, swapped, pending,
#    gpu_cache_percent, cpu_cache_percent, avg_latency_s, errors]
# We'll produce 9 bar charts:
#   1) prompt_tps
#   2) gen_tps
#   3) running
#   4) swapped
#   5) pending
#   6) gpu_cache_percent
#   7) cpu_cache_percent
#   8) avg_latency_s
#   9) errors

################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Read vllm_benchmark_results.csv and produce bar charts.")
    parser.add_argument("--csv-file",
                        type=str,
                        default=CSV_FILENAME,
                        help="CSV file to read (default: vllm_benchmark_results.csv)")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"[ERROR] CSV file not found: {args.csv_file}")
        return

    results = []
    with open(args.csv_file, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        hdr = next(rd, None)
        if not hdr or len(hdr) < 11:
            print("[WARN] CSV missing columns, cannot produce charts.")
            return
        # columns => 
        # [preset, model, prompt_tps, gen_tps, running, swapped, pending,
        #  gpu_cache_percent, cpu_cache_percent, avg_latency_s, errors]
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
            except ValueError:
                continue
            results.append((p, m, prompt_tps, gen_tps, running, swapped, pending,
                            gpu_cache, cpu_cache, avg_lat, errs))

    if not results:
        print("No valid rows in the CSV, skipping chart.")
        return

    # We'll separate them into 7 data groups + 2 new columns
    seen_presets: List[str] = []
    seen_models: List[str] = []
    for row in results:
        if row[0] not in seen_presets:
            seen_presets.append(row[0])
        if row[1] not in seen_models:
            seen_models.append(row[1])

    # Prepare dictionaries: data[(preset, model)] = value
    data_prompt: Dict[Tuple[str, str], float] = {}
    data_gen: Dict[Tuple[str, str], float] = {}
    data_run: Dict[Tuple[str, str], float] = {}
    data_swap: Dict[Tuple[str, str], float] = {}
    data_pending: Dict[Tuple[str, str], float] = {}
    data_gpu: Dict[Tuple[str, str], float] = {}
    data_cpu: Dict[Tuple[str, str], float] = {}
    data_latency: Dict[Tuple[str, str], float] = {}
    data_errors: Dict[Tuple[str, str], float] = {}

    for (p, m, pr, gn, run_, sw, pend, gpu, cp, lat, errs) in results:
        data_prompt[(p,m)] = pr
        data_gen[(p,m)] = gn
        data_run[(p,m)] = run_
        data_swap[(p,m)] = sw
        data_pending[(p,m)] = pend
        data_gpu[(p,m)] = gpu
        data_cpu[(p,m)] = cp
        data_latency[(p,m)] = lat
        data_errors[(p,m)] = errs

    # We'll define a helper function to produce bar charts.
    def make_bar_chart(metric_key: str,
                       data_dict: Dict[Tuple[str, str], float],
                       metric_title: str,
                       y_label: str,
                       out_filename: str):
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

    # 2 extra columns
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
