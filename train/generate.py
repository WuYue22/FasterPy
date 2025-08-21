import pandas as pd
from string import Template
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os
import re
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
import ast
from tqdm import tqdm
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 第一步：加载 LoRA 的配置
peft_model_path = "output-src-qw/checkpoint-1112"
config = PeftConfig.from_pretrained(peft_model_path)
model_id="model/Qwen2.5-7B-Instruct"
# 第二步：加载基础模型（LoRA 是在这个模型上微调的）
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,  # 原始模型
    trust_remote_code=True,
    # device_map="auto",
    device_map={"": device},
    torch_dtype="auto"
)

# 第三步：加载 LoRA 适配器权重
# 消融
# p_model=base_model
p_model = PeftModel.from_pretrained(base_model, peft_model_path).to(device)
p_model.eval()

# 第四步：加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def is_probably_python(text: str) -> bool:
    try:
        lexer = guess_lexer(text)
        return 'Python' in lexer.name
    except ClassNotFound:
        return False

def extract_code(text: list):
    pattern = fr"```python\s*(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    # =================================
    pattern = fr"```\s*(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    # =================================
    text.strip()
    possible_codes = text.split("```")
    # =================================
    if is_probably_python(possible_codes[0]):
        return possible_codes[0].strip()
    elif is_probably_python(possible_codes[-1]):
        return possible_codes[-1].strip()

    return text


# 读取数据
df = pd.read_json("pie-test-sug.jsonl",lines=True)
print(df.shape)

input_template = Template("Input code:\n$input\nSuggestions:\n$suggestions\nOutput: ```python\n {{optimized code}} \n``` <|EOS|>")
system="""You are an AI programming assistant. Try your best to optimize the Input code with focusing on reducing its runtime while maintaining correctness.
Feel free to refer to the suggestions below for potential optimizations, but you're not restricted to them.
Applicability represents the degree to which a suggestion fits the input code.
Rate represents the degree to which a suggestion improves the input code."""

tqdm.pandas()

torch.cuda.empty_cache()

def infer(row):
    input = row['slow_code_col']
    suggestions = ast.literal_eval(row['suggestion'])
    suggestions = "\n".join(
        [f"{index + 1}. Applicability:{suggestion['distance']} Rate:{suggestion['rate']}\n{suggestion['text']}" for
         index, suggestion in enumerate(suggestions)])
    prompt = input_template.substitute(input=input, suggestions=suggestions)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        # tokenize=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(p_model.device)
    with torch.inference_mode():
        outputs = p_model.generate(
            model_inputs,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
    response = tokenizer.decode(outputs[0][len(model_inputs[0]):], skip_special_tokens=True)
    del model_inputs
    del outputs
    torch.cuda.empty_cache()

    code = extract_code(response)
    return code

df['model_generated_potentially_faster_code_col'] = df.progress_apply(infer, axis=1)
torch.cuda.empty_cache()
df.to_json("test-my-qw.jsonl",lines=True,orient='records')
empty_rows = df[df['model_generated_potentially_faster_code_col'] == '']
print(empty_rows.shape)