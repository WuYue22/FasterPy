import json
import difflib
with open('..\config.json', 'r') as file:
    config = json.load(file)
#print(config['pie_v0'])

input_file = config['pie_v0']+"\\train.jsonl" #PIE
output_file = "PIEv2.jsonl"

fields_to_keep = {"input", "target", "improvement_frac"}

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        item = json.loads(line.strip())
        input_lines = item.get('input', '').strip().splitlines(keepends=True)
        target_lines = item.get('target', '').splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            input_lines,
            target_lines,
            fromfile="input.py",
            tofile="target.py",
            lineterm=""  # 防止结尾多出换行
        ))
        diff_text = ''.join(diff_lines)

        new_item = {field: item[field] for field in fields_to_keep if field in item}

        # rate
        if "improvement_frac" in item:
            try:
                new_item["rate"] = float(item["improvement_frac"]) / 100
                new_item["diff"] = diff_text
            except ValueError:
                new_item["rate"] = None  # 无法转为float时设为None

        # 写入新数据
        outfile.write(json.dumps(new_item, ensure_ascii=True) + '\n')


