#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

"""
This script produces bar charts for each of your 7 scenario tables:

1) Qwen-1.5B @2048
2) Llama-8B @2048
3) Qwen-32B @2048
4) Llama-8B @8192
5) Qwen-32B @8192
6) Qwen-1.5B @128k
7) Llama-8B @128k

Each scenario is stored as a dictionary:
  scenario_name = {
    "preset": (prompt_tps, gen_tps),
    ...
  }

We then plot one chart per scenario with X-axis = all presets, and two bars:
  - prompt
  - gen

We ignore latencies, errors, running, swapped, etc. in these specific bar charts.
You can adapt the code if you want multiple metrics per chart.
"""

# 1) Qwen-1.5B at 2048
qwen15b_2048 = {
    "gpu-small":  (90.31, 628.37),
    "gpu-medium": (91.82, 648.49),
    "gpu-large":  (89.11, 623.81),
    "mi210x1":    (75.15, 526.83),
    "mi210x2":    (62.67, 443.39),
    "H100X1":     (104.06, 737.20),
    "H100X2":     (98.97, 703.66),
}

# 2) Llama-8B at 2048
llama8b_2048 = {
    "gpu-medium": (55.06, 389.03),
    "gpu-large":  (58.38, 413.86),
    "gpu-xlarge": (74.20, 515.99),
    "mi210x1":    (52.97, 368.85),
    "mi210x2":    (53.25, 375.32),
    "H100X1":     (73.00, 512.25),
    "H100X2":     (81.51, 566.11),
}

# 3) Qwen-32B at 2048
qwen32b_2048 = {
    "gpu-xlarge": (31.26, 221.26),
    "mi210x2":    (24.78, 174.72),
    "H100X1":     (28.78, 204.40),
    "H100X2":     (38.23, 271.05),
}

# 4) Llama-8B at 8192
llama8b_8192 = {
    "gpu-medium": (60.01, 426.97),
    "gpu-large":  (57.56, 406.41),
    "gpu-xlarge": (55.92, 397.59),
    "mi210x1":    (55.12, 392.45),
    "mi210x2":    (60.57, 429.44),
    "H100X1":     (69.31, 492.02),
    "H100X2":     (63.93, 454.61),
}

# 5) Qwen-32B at 8192
qwen32b_8192 = {
    "gpu-xlarge": (10.56, 74.43),
    "mi210x2":    (13.87, 98.45),
    "H100X1":     (29.54, 209.36),
    "H100X2":     (37.38, 265.45),
}

# 6) Qwen-1.5B at 128k
qwen15b_128k = {
    # first 3 fail => 0.0 => 1000 errors
    "mi210x1":    (83.28, 590.77),
    "mi210x2":    (67.62, 478.06),
    "H100X1":     (82.16, 578.37),
    "H100X2":     (71.55, 504.43),
}

# 7) Llama-8B at 128k
# for simplicity, we unify the data from multiple runs; if you'd like separate charts, replicate.
llama8b_128k = {
    # from run #1:
    # "gpu-large":  (0.00, 0.00), # 1000 errors
    # "gpu-xlarge": (0.00, 0.00), # 1000 errors
    # "mi210x1":    (55.23, 390.70),
    # "mi210x2":    (60.28, 426.01),
    # "H100X1":     (71.85, 507.71),
    # "H100X2":     (65.37, 465.19),

    # from run #2:
    "mi210x1":    (55.14, 391.87),
    "mi210x2":    (60.84, 431.31),
    "H100X1":     (71.52, 506.32),
    "H100X2":     (64.54, 457.27),

    
}


def plot_scenario(
    scenario_name: str,
    scenario_data: dict,
    chart_title: str,
    out_filename: str
):
    """
    scenario_data: dict[preset] = (prompt_tps, gen_tps)
    We'll produce a bar chart with x-axis = presets,
    and 2 bars per preset: prompt TPS (blue), gen TPS (orange).
    """
    # Gather sorted presets
    presets = sorted(scenario_data.keys())
    # Build arrays
    prompt_vals = []
    gen_vals = []
    for p in presets:
        pr, gn = scenario_data[p]
        prompt_vals.append(pr)
        gen_vals.append(gn)

    x_indices = np.arange(len(presets))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x_indices - bar_width/2, prompt_vals, width=bar_width,
           color="tab:blue", alpha=0.8, label="Prompt TPS")
    ax.bar(x_indices + bar_width/2, gen_vals,    width=bar_width,
           color="tab:green", alpha=0.8, label="Gen TPS")

    ax.set_xticks(x_indices)
    ax.set_xticklabels(presets, rotation=45, ha="right")
    ax.set_ylabel("Tokens/s")
    ax.set_title(chart_title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_filename)
    print(f"{scenario_name}: saved chart => {out_filename}")
    plt.close(fig)


def main():
    # We'll produce 7 charts, one for each scenario:

    # 1) Qwen-1.5B at 2048
    plot_scenario(
        scenario_name = "qwen15b_2048",
        scenario_data = qwen15b_2048,
        chart_title   = "Qwen-1.5B @2048 tokens, 10 req/s for 1000 requests",
        out_filename  = "chart_qwen15b_2048.png"
    )

    # 2) Llama-8B at 2048
    plot_scenario(
        scenario_name = "llama8b_2048",
        scenario_data = llama8b_2048,
        chart_title   = "Llama-8B @2048 tokens, 10 req/s for 1000 requests",
        out_filename  = "chart_llama8b_2048.png"
    )

    # 3) Qwen-32B at 2048
    plot_scenario(
        scenario_name = "qwen32b_2048",
        scenario_data = qwen32b_2048,
        chart_title   = "Qwen-32B @2048 tokens, 10 req/s for 1000 requests",
        out_filename  = "chart_qwen32b_2048.png"
    )

    # 4) Llama-8B at 8192
    plot_scenario(
        scenario_name = "llama8b_8192",
        scenario_data = llama8b_8192,
        chart_title   = "Llama-8B @8192 tokens, 10 req/s for 1000 requests",
        out_filename  = "chart_llama8b_8192.png"
    )

    # 5) Qwen-32B at 8192
    plot_scenario(
        scenario_name = "qwen32b_8192",
        scenario_data = qwen32b_8192,
        chart_title   = "Qwen-32B @8192 tokens, 10 req/s for 1000 requests",
        out_filename  = "chart_qwen32b_8192.png"
    )

    # 6) Qwen-1.5B at 128k
    plot_scenario(
        scenario_name = "qwen15b_128k",
        scenario_data = qwen15b_128k,
        chart_title   = "Qwen-1.5B @128k tokens, 10 req/s for 1000 requests",
        out_filename  = "chart_qwen15b_128k.png"
    )

    # 7) Llama-8B at 128k
    plot_scenario(
        scenario_name = "llama8b_128k",
        scenario_data = llama8b_128k,
        chart_title   = "Llama-8B @128k tokens, 10 req/s for 1000 requests",
        out_filename  = "chart_llama8b_128k.png"
    )

if __name__ == "__main__":
    main()
