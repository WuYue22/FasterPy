import json
import numpy as np

def cal_speedup(file_path):
    speedups = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            input_time = data.get("input_time_mean")
            gen_time = data.get("model_generated_potentially_faster_code_col_time_mean")
            acc = data.get("model_generated_potentially_faster_code_col_acc")
            if acc == 1:
                # 避免除以 0 或缺失值
                if input_time is not None and gen_time not in (None, 0):
                    speedup = input_time / gen_time
                    if speedup>5:
                        speedup = 5

                    speedups.append(speedup)

    # 计算平均 speedup
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        # print(f"平均 Speedup: {avg_speedup:.6f}")
    else:
        print("没有有效数据可计算。")
    arr = np.array(speedups)
    print("------",file_path,"------")
    print("mean speedup:", np.mean(arr))
    print("median:", np.median(arr))
    print("max:", np.max(arr))
    print("P99:", np.percentile(arr, 99))

def cal_ref_speedup(file_path):
    speedups = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            input_time = data.get("input_time_mean")
            ref_time = data.get("reference_time_mean")
           
            # 避免除以 0 或缺失值
            if input_time is not None and ref_time not in (None, 0):
                speedup = input_time / ref_time
                if speedup>5:
                    speedup = 5

                speedups.append(speedup)


    # 计算平均 speedup
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        # print(f"平均 Speedup: {avg_speedup:.6f}")
    else:
        print("没有有效数据可计算。")
    arr = np.array(speedups)
    print("------",file_path,"------")
    print("mean speedup:", np.mean(arr))
    print("median:", np.median(arr))
    print("max:", np.max(arr))
    print("P99:", np.percentile(arr, 99))


file_path = r"\eval\eval-results\test-codex-report.jsonl"
cal_speedup(file_path)
cal_ref_speedup(file_path)
