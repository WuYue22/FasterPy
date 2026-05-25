import json
import numpy as np

def load_jsonl_np(path):
    input_tokens = []
    output_tokens = []
    total_tokens = []

    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            input_tokens.append(data["input_tokens"])
            output_tokens.append(data["output_tokens"])
            total_tokens.append(data["total_tokens"])

    return {
        "input": np.array(input_tokens),
        "output": np.array(output_tokens),
        "total": np.array(total_tokens),
    }
def compute_metrics_np(data):
    metrics = {}

    metrics["avg_input"] = np.mean(data["input"])
    metrics["avg_output"] = np.mean(data["output"])
    metrics["avg_total"] = np.mean(data["total"])

    metrics["sum_total"] = np.sum(data["total"])

    metrics["median_total"] = np.median(data["total"])
    metrics["p95_total"] = np.percentile(data["total"], 95)
    print(metrics)
    return metrics

def compute_reduction(base, ours):
    return {
        "input_reduction_%": (base["avg_input"] - ours["avg_input"]) / base["avg_input"] * 100,
        "output_reduction_%": (base["avg_output"] - ours["avg_output"]) / base["avg_output"] * 100,
        "total_reduction_%": (base["avg_total"] - ours["avg_total"]) / base["avg_total"] * 100,
    }

files = {
    "qwen_base": r"\eval\cost\qw-with-cost.jsonl",
    "qwen_ours": r"\eval\cost\qw-fp-cost.jsonl",
    "deepseek_base": r"\eval\cost\ds-non-cost.jsonl",
    "deepseek_ours": r"\eval\cost\ds-fp-cost.jsonl",
    "codellama_base": r"\eval\cost\cl-non-cost.jsonl",
    "codellama_ours": r"\eval\cost\cl-fp-cost.jsonl",
}

# 加载
data = {k: load_jsonl_np(v) for k, v in files.items()}

# 计算指标
metrics = {k: compute_metrics_np(v) for k, v in data.items()}

# 对比
pairs = [
    ("qwen_base", "qwen_ours"),
    ("deepseek_base", "deepseek_ours"),
    ("codellama_base", "codellama_ours"),
]

for base_name, ours_name in pairs:
    base = metrics[base_name]
    ours = metrics[ours_name]

    reduction = compute_reduction(base, ours)

    print(f"\n=== {base_name} vs {ours_name} ===")
    print(f"Avg total tokens: {base['avg_total']:.1f} -> {ours['avg_total']:.1f}")
    print(f"Reduction: {reduction['total_reduction_%']:.2f}%")
