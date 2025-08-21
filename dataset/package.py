import json

input_path = 'OD-merged.jsonl'
output_path = 'OD-merged-packaged.jsonl'

output_data = []

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for idx, line in enumerate(infile, start=1):
        item = json.loads(line)
        diff = item.get('diff', '').strip()

        formatted_item = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "qwen-max-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert in the field of code execution efficiency optimization."
                    },
                    {
                        "role": "user",
                        "content": f"Based on this unified-style patch: {diff}, summarize how it optimizes code execution efficiency. Provide up to two key points. Your output format should be: (If there is one summary) 1.xxx (If there are two summaries) 1.xxx;2.xxx"
                    }
                ]
            }
        }

        json.dump(formatted_item, outfile, ensure_ascii=True)
        outfile.write('\n')
