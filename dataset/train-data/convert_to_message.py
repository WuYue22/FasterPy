import json

input_path = "train_data_sug.jsonl"
output_path = "train_data_msg.jsonl"

system_prompt = """You are an AI programming assistant. Try your best to optimize the Input code with focusing on reducing its runtime while maintaining correctness. Output the optimized code. Feel free to refer to the suggestions below for potential optimizations, but you’re not restricted to them. Applicability represents the degree to which a suggestion fits the input code. Rate represents the degree to which a suggestion improves the input code."""

def build_user_prompt(input_code, suggestions):
    suggestion_texts = [f"{i+1}.Applicability:{s['distance']},Rate:{s['rate']},{s['suggestion']}" for i,s in enumerate(suggestions)]
    joined_suggestions = "\n".join(suggestion_texts)
    return (
        f"""Input code:
 ```
 {input_code}
 ```
"""
        f"Suggestions:{joined_suggestions}"
    )


with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        item = json.loads(line)
        input_code = item.get("input", "").strip()
        target_code = item.get("target", "").strip()
        suggestions = item.get("suggestions", [])[:2]

        if not input_code or not target_code or len(suggestions) < 1:
            continue  # 跳过不完整数据

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(input_code, suggestions)},
            {"role": "assistant", "content": target_code}
        ]
        json.dump({"messages": messages}, outfile, ensure_ascii=False)
        outfile.write("\n")

print(f"✅ completed，saved to {output_path}")
