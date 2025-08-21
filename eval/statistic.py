import pandas as pd
import sys

from metrics import PassAt1,OPT
if __name__=='__main__':
    path= sys.argv[1]
    df=pd.read_json(path,lines=True)

    df['su']=df['input_time_mean']/df['model_generated_potentially_faster_code_col_time_mean']
    cor=df[df['model_generated_potentially_faster_code_col_acc']==1]
    # print(f"%OPT = {OPT(cor['input_time_mean'].tolist(),cor['model_generated_potentially_faster_code_col_time_mean'].tolist()):.4f}")
    print(f"Pass@1 = {PassAt1(df['model_generated_potentially_faster_code_col_acc'].tolist()):.4f}")
    # cor['su'].clip(lower=1)
    print("Speedup@1")
    print(cor['su'].describe())
    print("Speedup@1 1~10k")
    cor=cor[(cor['su']>=1)&(cor['su']<=10000)]
    print(cor['su'].describe())

    print("Speedup@1 with incorrect")
    print(df['su'].describe())