from typing import List, Tuple

PROMPT_TEMPLATE = "Try your best to optimize the Input code with focusing on reducing its runtime while maintaining correctness."

# generate prompt
def gen_prompt(original_code: str, func_signature: str, examples: List[Tuple[str, float]]) -> str:
    prompt = PROMPT_TEMPLATE
    prompt += f"Input code:\n```\n{original_code}\n```\n"
    for summary, similarity in examples:
        # src_code,patch =  example
        prompt += f"\Applicability: {similarity}\n Optimization suggestion: \n/* {summary} */\n Give the optimization code for this function and its callees: \n{func_signature}"
    return prompt