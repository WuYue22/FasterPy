import pandas as pd
from tqdm import tqdm
import json
df=pd.read_json('/Users/hanminghao/Desktop/study/毕业设计/eval/src/codex_greedy_outputs.jsonl.report',lines=True)
print(df.shape)
print(df.columns)
# tqdm.pandas()
# def to_json_line(row):
#     return {'slow_code_col':row['input'],'model_generated_potentially_faster_code_col':row['target']}
# df['json_line']=df.progress_apply(to_json_line,axis=1)
# df['json_line'].to_json('./test.jsonl',orient='records',lines=True)