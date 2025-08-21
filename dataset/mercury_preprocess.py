import difflib

import datasets
from datasets import load_dataset,load_from_disk
import json

dataset_name = "Elfsong/Mercury"
# dataset = load_dataset(dataset_name)
# dataset.save_to_disk('E:\\code generation\\mercury')
dataset=datasets.load_from_disk("E:\\code generation\\mercury")
data = dataset['eval']
samples = []

def parse_runtime_ms(runtime_str):
    """
    去除字符串中的 'ms'，并转换为 float 毫秒值
    如 '14ms' -> 14.0
    """
    try:
        return float(runtime_str.strip().lower().replace("ms", ""))
    except (ValueError, AttributeError):
        return float('inf')  # 若格式错误就跳过

for item in data:
    solutions = item.get("solutions", [])

    # 要求至少有 2 个提交才能配对
    if len(solutions) < 2:
        continue

    # 按 runtime 升序排序
    sorted_solutions = sorted(solutions, key=lambda s: parse_runtime_ms(s.get("runtime", "inf")))
    #print(len(sorted_solutions))
    # 取最快实现作为 target
    fastest = sorted_solutions[0]
    fastest_runtime = parse_runtime_ms(fastest.get("runtime",1))  # 避免除0
    #print("fastest_runtime:",fastest_runtime)
    fastest_solution = fastest.get("solution", "")

    # 对其余提交，构造样本
    for submission in sorted_solutions[1:]:
        input_code = submission.get("solution", "")
        input_runtime = parse_runtime_ms(submission.get("runtime",float('inf')))

        # 过滤异常情况
        if fastest_runtime == 0 or input_runtime == float('inf'):
            continue

        # 构造 diff 补丁
        input_lines = input_code.splitlines(keepends=True)
        target_lines = fastest_solution.splitlines(keepends=True)

        diff_lines = list(difflib.unified_diff(
                input_lines,
                target_lines,
                fromfile="input.py",
                tofile="target.py",
                lineterm=""  # 防止结尾多出换行
            ))

        diff_text = ''.join(diff_lines)

        sample = {
            "input": input_code,
            "target": fastest_solution,
            "rate": round(input_runtime / fastest_runtime, 4),
            "diff": diff_text
        }
        samples.append(sample)

with open("Mercuryv2.jsonl", "w", encoding="utf-8") as f:
    for item in samples:
        f.write(json.dumps(item, ensure_ascii=True) + "\n")