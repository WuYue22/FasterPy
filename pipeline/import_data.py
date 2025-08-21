from knowledge.knowledge_base import KnowledgeBase
import pandas as pd
import ast
from tqdm import tqdm
tqdm.pandas()
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
df = pd.read_json("../dataset/knowlegde-base/OD-base.jsonl",lines=True)
print(df.shape)
print(df.loc[1,:])
df['src_code_len'] = df.parallel_apply(lambda row: len(row['input'].splitlines()), axis=1)
df['vector']=df.parallel_apply(lambda row: ast.literal_eval(row['vector']), axis=1)

kb  = KnowledgeBase(db_name='CKB')
def insert(row):
    data={
        "vector":row["vector"],
        "src_code_len":row["src_code_len"],
        "summary":row["summary"],
        "rate":row["rate"]
    }
    kb.insert_single_with_vector(data)
df.progress_apply(insert, axis=1)

