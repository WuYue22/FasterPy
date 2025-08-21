
import math
from typing import List
import numpy as np

def flatten_list(l:list):      
    return [item for sublist in l for item in sublist]


def PassAtK(acc_list:List[List[float]],N:int=1,k:int=1) -> float:
    """
    Pass@k for num_return_sequences=N
    """
    acc_list = [[0 if item != 1 else 1 for item in sub_list]for sub_list in acc_list]
    inners=[]
    for sub_list in acc_list:
        inner=(1    -   ( math.comb(N-sum(sub_list), k)/math.comb(N,k) ) )
        inners.append(inner)
    return np.mean(inners)
def PassAt1(acc_list:List[float]) -> float:
    """
    Pass@1 for num_return_sequences=1
    """
    acc_list = [0 if item != 1 else 1 for item in acc_list]
    return np.mean(acc_list)

def SpeedupAt1(baseline_runtime:List[float],model_generated_runtime:List[float]) -> float:
    """
    Speedupn@k for N=num_return_sequences=1,k=1
    """
    assert len(baseline_runtime)==len(model_generated_runtime)
    speed_up_per_trail=[baseline_runtime[i]/model_generated_runtime[i] for i in range(len(baseline_runtime))]
    speed_up_per_trail = [speedup if speedup > 1 else 1 for speedup in speed_up_per_trail]
    return np.mean(speed_up_per_trail)
def OPT(baseline_runtime:List[float],model_generated_runtime:List[float])->float:
    assert len(baseline_runtime)==len(model_generated_runtime)
    total_count=len(baseline_runtime)
    faster_count=0
    for br,mgr in zip(baseline_runtime,model_generated_runtime):
        speedup=(br/mgr)-1
        # A program must be at least 10% faster and correct to contribute.
        if speedup>0.1:
            faster_count+=1
    return faster_count/total_count