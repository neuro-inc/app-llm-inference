#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------------
# Hard-coded pipeline vs. tensor data (PromptTPS, GenTPS) for each model.
# Format: {preset: (promptTPS, genTPS)}.
# ------------------------------------------------------------------------

# Qwen-1.5B
qwen15b_pipeline = {
    "gpu-small":  (86.59, 619.12),
    "mi210x1":    (66.04, 454.55),
    "mi210x2":    (53.57, 380.14),
    "H100X1":     (104.72, 741.28),
}
qwen15b_tensor = {
    "gpu-small":  (90.31, 628.37),
    "mi210x1":    (75.15, 526.83),
    "mi210x2":    (62.67, 443.39),
    "H100X1":     (104.06, 737.20),
}

# Llama-8B
llama8b_pipeline = {
    "gpu-medium": (34.65, 246.05),
    "gpu-large":  (29.50, 205.78),
    "gpu-xlarge": (20.78, 147.21),
    "mi210x1":    (9.55, 66.07),
    "mi210x2":    (41.56, 292.45),
    "H100X1":     (10.38, 72.66),
    "H100X2":     (54.30, 383.45),
}
llama8b_tensor = {
    "gpu-medium": (55.06, 389.03),
    "gpu-large":  (58.38, 413.86),
    "gpu-xlarge": (74.20, 515.99),
    "mi210x1":    (52.97, 368.85),
    "mi210x2":    (53.25, 375.32),
    "H100X1":     (73.00, 512.25),
    "H100X2":     (81.51, 566.11),
}

# Qwen-32B
qwen32b_pipeline = {
    "mi210x2": (14.98, 106.93),
    "H100X1":  (28.96, 203.87),
}
qwen32b_tensor = {
    "mi210x2": (24.78, 174.72),
    "H100X1":  (28.78, 204.40),
}

# ------------------------------------------------------------------------
# A helper function to produce a bar chart for either prompt or gen metric
# *and* compute & log the ratio = (tensor / pipeline) - 1.
# ------------------------------------------------------------------------

def plot_one_metric(
    model_name: str,
    pipeline_data: dict,
    tensor_data: dict,
    metric: str,  # "prompt" or "gen"
    out_file: str
):
    """
    pipeline_data[preset] = (promptTPS, genTPS)
    tensor_data[preset]   = (promptTPS, genTPS)

    We'll produce a bar chart with 2 bars (pipeline, tensor) per preset.
    We'll also compute ratio = (tensor_val / pipeline_val) - 1
    to indicate how much faster tensor is vs. pipeline, in decimal form.
    E.g. 0.25 => 25% faster.
    """
    # metric_idx: 0 => prompt, 1 => gen
    if metric == "prompt":
        metric_idx = 0
    elif metric == "gen":
        metric_idx = 1
    else:
        raise ValueError("metric must be 'prompt' or 'gen'")

    # Gather all unique presets
    all_presets = sorted(set(pipeline_data.keys()) | set(tensor_data.keys()))

    pipeline_vals = []
    tensor_vals = []
    # We'll store differences for logging
    ratio_list = []  # array of (preset, pipeline_val, tensor_val, ratio)

    for preset in all_presets:
        # pipeline val
        if preset in pipeline_data:
            pipe_tuple = pipeline_data[preset]
            pipeline_val = pipe_tuple[metric_idx]
        else:
            pipeline_val = 0.0

        # tensor val
        if preset in tensor_data:
            tens_tuple = tensor_data[preset]
            tensor_val = tens_tuple[metric_idx]
        else:
            tensor_val = 0.0

        pipeline_vals.append(pipeline_val)
        tensor_vals.append(tensor_val)

        # compute ratio if pipeline_val > 0
        if pipeline_val > 0:
            ratio_dec = (tensor_val / pipeline_val) - 1.0
        else:
            ratio_dec = None

        ratio_list.append((preset, pipeline_val, tensor_val, ratio_dec))

    # Log differences
    print(f"\n=== {model_name} {metric.capitalize()} - Pipeline vs. Tensor ===")
    sum_ratio = 0.0
    count_valid = 0
    for (preset, pv, tv, ratio_dec) in ratio_list:
        if ratio_dec is not None:
            sum_ratio += ratio_dec
            count_valid += 1
            # E.g. ratio=0.25 => 25% faster
            print(f"  {preset}: pipeline={pv:.2f}, tensor={tv:.2f}, ratio={ratio_dec:.2f}")
        else:
            print(f"  {preset}: pipeline={pv:.2f}, tensor={tv:.2f}, ratio=N/A (pipeline=0)")

    if count_valid > 0:
        avg_ratio = sum_ratio / count_valid
        print(f"  -> Average ratio across presets: {avg_ratio:.2f}")
        print(f"     (e.g. {avg_ratio:.2f} => {avg_ratio*100:.0f}% faster than pipeline on average.)")
    else:
        print("  -> No valid pipeline data to compare")

    # Build chart
    x_indices = np.arange(len(all_presets))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(x_indices - bar_width/2, pipeline_vals, width=bar_width,
           label="Pipeline", color="tab:blue", alpha=0.8)
    ax.bar(x_indices + bar_width/2, tensor_vals,   width=bar_width,
           label="Tensor", color="tab:green", alpha=0.8)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(all_presets, rotation=45, ha="right")
    ax.set_ylabel("Tokens/s")
    chart_title = f"{model_name} - {metric.capitalize()} Throughput\n(Pipeline vs Tensor)"
    ax.set_title(chart_title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved chart => {out_file}")
    plt.close(fig)

# ------------------------------------------------------------------------
# We'll produce 2 charts per model => 6 total. Also logs the ratio.
# ------------------------------------------------------------------------

def main():
    # Qwen-1.5B
    plot_one_metric("Qwen-1.5B",
                    qwen15b_pipeline,
                    qwen15b_tensor,
                    "prompt",
                    "qwen15b_prompt_pipeline_vs_tensor.png")
    plot_one_metric("Qwen-1.5B",
                    qwen15b_pipeline,
                    qwen15b_tensor,
                    "gen",
                    "qwen15b_gen_pipeline_vs_tensor.png")

    # Llama-8B
    plot_one_metric("Llama-8B",
                    llama8b_pipeline,
                    llama8b_tensor,
                    "prompt",
                    "llama8b_prompt_pipeline_vs_tensor.png")
    plot_one_metric("Llama-8B",
                    llama8b_pipeline,
                    llama8b_tensor,
                    "gen",
                    "llama8b_gen_pipeline_vs_tensor.png")

    # Qwen-32B
    plot_one_metric("Qwen-32B",
                    qwen32b_pipeline,
                    qwen32b_tensor,
                    "prompt",
                    "qwen32b_prompt_pipeline_vs_tensor.png")
    plot_one_metric("Qwen-32B",
                    qwen32b_pipeline,
                    qwen32b_tensor,
                    "gen",
                    "qwen32b_gen_pipeline_vs_tensor.png")

if __name__ == "__main__":
    main()
