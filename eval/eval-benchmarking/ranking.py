import json
import numpy as np
from scipy.stats import kendalltau

# 1. 读取 jsonl 文件
def load_reference_times(file_path):
    times = []
    ids = []
    with open(file_path, "r") as f:
        for line in f:
            d = json.loads(line)
            times.append(d["reference_time_mean"])
            ids.append(d["problem_id"])
    return np.array(times), ids


# 2. 确保三个文件顺序一致
def align_by_id(times_list, ids_list):
    # 以第一个文件为基准排序
    base_ids = ids_list[0]
    id_to_index_list = [{id_: i for i, id_ in enumerate(ids)} for ids in ids_list]

    aligned_times = []
    for times, id_map in zip(times_list, id_to_index_list):
        aligned = [times[id_map[id_]] for id_ in base_ids]
        aligned_times.append(np.array(aligned))

    return aligned_times


# 3. Kendall Tau
def compute_kendall_tau(run1, run2, run3):
    tau_12, _ = kendalltau(run1, run2)
    tau_13, _ = kendalltau(run1, run3)
    tau_23, _ = kendalltau(run2, run3)

    avg_tau = (tau_12 + tau_13 + tau_23) / 3

    return {
        "tau_12": tau_12,
        "tau_13": tau_13,
        "tau_23": tau_23,
        "avg_tau": avg_tau
    }


# 4. Pairwise Ranking Consistency
def ranking_consistency(a, b):
    n = len(a)
    agree = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (a[i] < a[j]) == (b[i] < b[j]):
                agree += 1

    return agree / total


def compute_pairwise_consistency(run1, run2, run3):
    c12 = ranking_consistency(run1, run2)
    c13 = ranking_consistency(run1, run3)
    c23 = ranking_consistency(run2, run3)

    avg_c = (c12 + c13 + c23) / 3

    return {
        "c12": c12,
        "c13": c13,
        "c23": c23,
        "avg_consistency": avg_c
    }

def main(file1, file2, file3):
    t1, id1 = load_reference_times(file1)
    t2, id2 = load_reference_times(file2)
    t3, id3 = load_reference_times(file3)

    # 对齐 problem_id（防止顺序不同）
    t1, t2, t3 = align_by_id([t1, t2, t3], [id1, id2, id3])

    print("Loaded samples:", len(t1))

    # Kendall Tau
    tau_results = compute_kendall_tau(t1, t2, t3)
    print("\n=== Kendall Tau ===")
    for k, v in tau_results.items():
        print(f"{k}: {v:.4f}")

    # Pairwise Consistency
    cons_results = compute_pairwise_consistency(t1, t2, t3)
    print("\n=== Pairwise Ranking Consistency ===")
    for k, v in cons_results.items():
        print(f"{k}: {v:.4f}")



if __name__ == "__main__":
    main(
        "\eval-results_for_benchmarking\pie-report-1.jsonl",
        "\eval-results_for_benchmarking\pie-report-2.jsonl",
        "\eval-results_for_benchmarking\pie-report-3.jsonl"
    )