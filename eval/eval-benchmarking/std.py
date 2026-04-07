import json
import numpy as np

def load(file):
    times = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)

            t = d.get("reference_time_mean")

            if t is not None:
                times.append(float(t))

    return np.array(times)
file_path=r"\eval-results_for_benchmarking\pie-report-1.jsonl"
A1 = load(file_path)
A2 = load(file_path.replace("1","2"))
A3 = load(file_path.replace("1","3"))

mean_A = np.mean([A1,A2,A3],axis=0)
std_A = np.std([A1,A2,A3],axis=0)
cv_A = std_A / mean_A
print("Mean Time:", np.mean(mean_A))
print("Std Time:", np.mean(std_A))
print("Mean CV:", np.mean(cv_A))
print("Median CV:", np.median(cv_A))