import json
from tqdm import tqdm
from knowledge.code_embedder import CodeEmbedder

input_file = "OD-merged.jsonl"          # 原数据集
output_file = "OD-merged-with-vec.jsonl" # 输出数据集

# 初始化向量生成器
ce = CodeEmbedder()

# 读取数据集
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 批量生成向量
for row in tqdm(data, desc="Generating embeddings"):
    if "input" in row:
        try:
            vec = ce(row["input"], keep_tensor=False)  # 返回 list
            row["vector"] = vec
        except Exception as e:
            row["vector"] = None
            print(f"编码失败: {e}")
    else:
        print(f"缺少 input 字段: {row.idx}")

# 保存
with open(output_file, "w", encoding="utf-8") as f:
    for row in data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"已处理 {len(data)} 条数据，结果保存到 {output_file}")
