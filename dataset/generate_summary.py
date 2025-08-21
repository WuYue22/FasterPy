import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import torch
import gc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "../model/Qwen-Max-Latest"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map={"": device},
    torch_dtype="auto"
)
p_model = base_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

df = pd.read_json("OD-merged-packaged-example.jsonl", lines=True)
print(f"df.shape: {df.shape}")

torch.cuda.empty_cache()

def generate_batch(batch_messages, max_new_tokens=192):
    batch_inputs = [
        tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    model_inputs = tokenizer(
        batch_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(p_model.device)
    with torch.inference_mode():
        outputs = p_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
    summaries = []
    for j in range(len(batch_inputs)):
        generated_tokens = outputs[j][model_inputs["input_ids"].shape[1]:]
        summary = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        summaries.append(summary)
    del model_inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return summaries

def batch_generate_summaries_dynamic(messages_list, default_batch_size=16, small_batch_size=2, max_new_tokens=192):
    summaries = []
    i = 0
    n = len(messages_list)
    batch_size = default_batch_size

    pbar = tqdm(total=n, desc="批量生成")
    while i < n:
        try:
            current_batch = messages_list[i:i+batch_size]
            batch_summaries = generate_batch(current_batch, max_new_tokens=max_new_tokens)
            summaries.extend(batch_summaries)
            i += batch_size
            pbar.update(batch_size)
            batch_size = default_batch_size  # 成功后恢复默认批量大小
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM occurred at index {i} with batch_size={batch_size}. Switching to small batch size {small_batch_size}...")
                torch.cuda.empty_cache()
                gc.collect()
                # 逐条或小批量生成，避免大批次OOM
                for j in range(i, min(i + batch_size, n), small_batch_size):
                    small_batch = messages_list[j:j+small_batch_size]
                    try:
                        small_summaries = generate_batch(small_batch, max_new_tokens=max_new_tokens)
                        summaries.extend(small_summaries)
                        pbar.update(len(small_batch))
                    except RuntimeError as e2:
                        print(f"Even small batch OOM at index {j}. Skipping these inputs.")
                        summaries.extend([''] * len(small_batch))  # 用空字符串占位
                        torch.cuda.empty_cache()
                        gc.collect()
                i += batch_size
                batch_size = default_batch_size  # 处理完小批量后恢复默认批量大小
            else:
                raise e
    pbar.close()
    return summaries


messages_list = df["body"].apply(lambda x: x["messages"]).tolist()
df["summary"] = batch_generate_summaries_dynamic(messages_list, default_batch_size=16, small_batch_size=2, max_new_tokens=192)

df.to_json("OD-merged-summary-example.jsonl", lines=True, orient="records")
empty_rows = df[df['summary'] == '']
print(f"empty rows: {empty_rows.shape[0]}")
