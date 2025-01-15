import re
import csv
from collections import defaultdict

def parse_task_info(line):
    """
    Extracts model name (with underscores), environment variable values, and status from a log line.
    Example line:

        [10:02:31 +0000] Task amd_crash_test_model_Qwen_Qwen2_7B_Instruct_flash_1_fp8_0_hipDsa_1_finePCIE_1_sdma_0_cuMask_0 job-bb57dc08-2b36-4311-a9f2-5ae45f0ac3c8 is succeeded (no outputs)

    Returns a dict like:
        {
          "model": "Qwen_Qwen2_7B_Instruct",
          "flash": "1",
          "fp8": "0",
          "hipDsa": "1",
          "finePCIE": "1",
          "sdma": "0",
          "cuMask": "0",
          "status": "succeeded"
        }

    or None if the line doesn't match the pattern.
    """
    pattern = re.compile(
        r"Task\s+amd_crash_test_model_([^ ]+)\s+job-[a-z0-9\-]+\s+is\s+(succeeded|failed)"
    )
    match = pattern.search(line)
    if not match:
        return None

    chunk = match.group(1)  # e.g., "Qwen_Qwen2_7B_Instruct_flash_1_fp8_0_hipDsa_1_finePCIE_1_sdma_0_cuMask_0"
    status = match.group(2) # "succeeded" or "failed"

    # The chunk typically looks like:
    #   {model_name_with_underscores}_flash_{0or1}_fp8_{0or1}_hipDsa_{0or1}_finePCIE_{0or1}_sdma_{0or1}_cuMask_{0or1}
    #
    # Example: Qwen_Qwen2_7B_Instruct_flash_1_fp8_0_hipDsa_1_finePCIE_1_sdma_0_cuMask_0

    parts = chunk.split("_flash_")
    if len(parts) < 2:
        return None

    model_part = parts[0]  # e.g., "Qwen_Qwen2_7B_Instruct"
    remainder = parts[1]   # e.g., "1_fp8_0_hipDsa_1_finePCIE_1_sdma_0_cuMask_0"

    re2 = re.compile(r"^(\d+)_fp8_(\d+)_hipDsa_(\d+)_finePCIE_(\d+)_sdma_(\d+)_cuMask_(\d+)$")
    match2 = re2.search(remainder)
    if not match2:
        return None

    flash_value = match2.group(1)
    fp8_value   = match2.group(2)
    hipdsa_val  = match2.group(3)
    finepcie    = match2.group(4)
    sdma_val    = match2.group(5)
    cumask_val  = match2.group(6)

    return {
        "model": model_part,
        "flash": flash_value,
        "fp8": fp8_value,
        "hipDsa": hipdsa_val,
        "finePCIE": finepcie,
        "sdma": sdma_val,
        "cuMask": cumask_val,
        "status": status
    }

def parse_logs_and_create_csv(
    logfile_path: str,
    models: list[str],
    csv_output_path: str = "vllm_results.csv",
    popularity_csv: str = "vllm_config_popularity.csv"
):
    """
    1) Reads the text file line by line to find lines matching "Task amd_crash_test_model_..."
       and extracts model, env vars, and status.
    2) Outputs a CSV with columns:
       [model, flash, fp8, hipDsa, finePCIE, sdma, cuMask, status]
    3) Ranks each config by how many models it succeeded on, writing a second CSV with
       [flash, fp8, hipDsa, finePCIE, sdma, cuMask, success_count, models_succeeded].
    4) Prints the same popularity info to console, plus any config that succeeds for all models.
    5) Reports models for which *no* configuration succeeded.
    """
    results = []  # List[dict(...)] of per-task info.

    # --- 1) Parse the log file ---
    with open(logfile_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_task_info(line)
            if parsed:
                results.append(parsed)

    # --- 2) Write the raw results to a CSV ---
    with open(csv_output_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "model",
            "vllm_use_triton_flash_attn",
            "vllm_fp8_padding",
            "torch_use_hip_dsa",
            "hsa_force_fine_grain_pcie",
            "hsa_enable_sdma",
            "rocm_disable_cu_mask",
            "status"
        ])
        for r in results:
            writer.writerow([
                r["model"],
                r["flash"],
                r["fp8"],
                r["hipDsa"],
                r["finePCIE"],
                r["sdma"],
                r["cuMask"],
                r["status"]
            ])

    print(f"Parsed {len(results)} lines matching tasks. CSV written to {csv_output_path}.")

    # --- 3) Build a map: config -> set of models that SUCCEEDED ---
    config_success_map = defaultdict(set)
    # We'll also track a set of all models that appear in logs at all (for cross-check)
    all_logged_models = set()

    for r in results:
        all_logged_models.add(r["model"])
        if r["status"] == "succeeded":
            config_key = (r["flash"], r["fp8"], r["hipDsa"], r["finePCIE"], r["sdma"], r["cuMask"])
            config_success_map[config_key].add(r["model"])

    # If your logs store the model name with underscores, 
    # but your 'models' list has slashes/dots, we unify them via a function:
    def underscoreify(m: str) -> str:
        return m.replace("/", "_").replace("-", "_").replace(".", "_")

    # We'll keep the log-based names (underscore form) for set membership checks
    underscore_models = [underscoreify(m) for m in models]

    # --- 4) Count how many models each config succeeded for ---
    popularity_list = []
    for config_key, model_set in config_success_map.items():
        success_count = len(model_set)
        model_list = sorted(model_set)  # e.g. ["Qwen_Qwen2_7B_Instruct", ...]
        popularity_list.append((config_key, success_count, model_list))

    # Sort descending by success_count
    popularity_list.sort(key=lambda x: x[1], reverse=True)

    # --- 5) Write the popularity CSV ---
    with open(popularity_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "vllm_use_triton_flash_attn",
            "vllm_fp8_padding",
            "torch_use_hip_dsa",
            "hsa_force_fine_grain_pcie",
            "hsa_enable_sdma",
            "rocm_disable_cu_mask",
            "success_count",
            "models_succeeded"
        ])
        for (flash, fp8, hipDsa, finePCIE, sdma, cuMask), success_count, model_list in popularity_list:
            writer.writerow([
                flash, fp8, hipDsa, finePCIE, sdma, cuMask,
                success_count,
                ";".join(model_list)
            ])

    print(f"Wrote config popularity to {popularity_csv}.")

    # --- 6) Print popularity info ---
    print("\n=== Configuration popularity ranking ===")
    for (flash, fp8, hipDsa, finePCIE, sdma, cuMask), success_count, model_list in popularity_list:
        print(f"- Config flash={flash}, fp8={fp8}, hipDsa={hipDsa}, "
              f"finePCIE={finePCIE}, sdma={sdma}, cuMask={cuMask}: "
              f"{success_count} successes => {model_list}")

    # --- 7) Identify any config that succeeded for ALL tested models ---
    best_configs = []
    for (config_key, model_set) in config_success_map.items():
        # Check if all 'underscore_models' are in model_set
        if all(m in model_set for m in underscore_models):
            best_configs.append(config_key)

    if not best_configs:
        print("\nNo single config was successful for *ALL* models.\n")
    else:
        print("\nConfiguration(s) that succeeded for ALL models:\n")
        for bc in best_configs:
            flash, fp8, hipdsa, finepcie, sdma, cumask = bc
            print(f"  - flash={flash}, fp8={fp8}, hipDsa={hipdsa}, "
                  f"finePCIE={finepcie}, sdma={sdma}, cuMask={cumask}")

    # --- 8) Find models for which NO config worked ---
    # We'll gather all models that appear in any success set
    models_that_succeeded = set()
    for model_set in config_success_map.values():
        models_that_succeeded.update(model_set)

    # Now check which underscore_models never appear in that union
    missing = [m for m in underscore_models if m not in models_that_succeeded]

    if missing:
        print("\nNo configuration worked for these models:\n")
        for mm in missing:
            print(f"  - {mm}")
    else:
        print("\nAll tested models succeeded in at least one configuration.\n")


# ----------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    models_tested = [
        "lmsys/vicuna-7b-v1.3",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "meta-llama/Llama-2-7b-hf",
        "tiiuae/falcon-7b-instruct",
        "mistralai/Mistral-7B-v0.1",
        "baichuan-inc/Baichuan2-13B-Chat",
        "THUDM/chatglm2-6b",
        "bigscience/bloomz",
    ]

    parse_logs_and_create_csv(
        logfile_path="apolo_output.log",              # Your raw log file
        models=models_tested,
        csv_output_path="vllm_results.csv",
        popularity_csv="vllm_config_popularity.csv"  
    )
