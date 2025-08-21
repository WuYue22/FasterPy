from typing import List
from py_formatter import Formatter
from knowledge.knowledge_base import KnowledgeBase
from prompt_generator import gen_prompt
from patch_synthesizer import syn
from fasterpy_generator import FasterPyGenerator
import re
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

NORMAL_EXIT = 0
EXAMPLE_NOT_FOUNDED = 1
 
 
class InferencePipeline:
    def __init__(self, model_path:str = "../model/deepseek-coder-6.7b-Instruct-FT"):
        self.formatter = Formatter()
        self.kb = KnowledgeBase("CKB")
        self.fpg = FasterPyGenerator(model_path)

    def is_probably_python(self,text: str) -> bool:
        try:
            lexer = guess_lexer(text)
            return 'Python' in lexer.name
        except ClassNotFound:
            return False

    def extract_code(self, text: list):
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
        if self.is_probably_python(possible_codes[0]):
            return possible_codes[0].strip()
        elif self.is_probably_python(possible_codes[-1]):
            return possible_codes[-1].strip()
        return text

    # r_thre for rate thredhold, s_thre for similarity threhold
    def inference(self, origin_code :str, target_function_name:str="Solution",example_num:int=1, r_thre:float=0.5,s_thre:float=0.1)->int:
        # 1. 输入要改进的文件，以及指定要改进的函数
        # 2. formatter去除无关函数，只保留指定函数及其callee，并进行格式化
        formatted_code = self.formatter.format(origin_code, target_function_name)
        if not formatted_code:
            print("err when formating code")
            formatted_code = origin_code
            # return 0
        # 3. 从知识库找到相相似的 “被改进代码” 及建议
        res = self.kb.search([formatted_code],top_k=example_num, similarity_thredhold=s_thre,rate_thredhold=r_thre)
        hits = res[0]
        if len(hits)<1:
            print("!Can not find similar improvement case")
            improve_examples = []
            # return EXAMPLE_NOT_FOUNDED
        else:
            improve_examples = [(hit["entity"]["summary"],hit["distance"]) for hit in hits]
        # print(improve_examples)
        # 4. 生成prompt
        func_signature = f"def {target_function_name}("
        prompt = gen_prompt(origin_code, func_signature,improve_examples)
        # 5. 传递给fasterc_generator.py，生成优化代码
        ans = self.fpg([prompt])
        # 6. 提取代码
        optimized_code = self.extract_code(ans)
        # 7. 输出新的代码
        print(f"===optimized_code:===\n{optimized_code}")
        # 8. return 0;
        return 0
    
    def inference_batch(self, origin_codes:List[str], target_function_names:List[str]="Solution",example_num:int=1, r_thre:float=0.5,s_thre:float=0.1)->int:
        # 1. 输入要改进的文件，以及指定要改进的函数
        # 2. formatter去除无关函数，只保留指定函数及其callee，并进行格式化
        formatted_codes = []
        for origin_code, target_function_name in zip(origin_codes, target_function_names):
            formatted_code = self.formatter.format(origin_code, target_function_name)
            if not formatted_code:
                print("err when formating code")
                formatted_code = origin_code
                # return 0
            formatted_codes.append(formatted_code)
        # 3. 从知识库找到相相似的 “被改进代码” 以及 “改进patch”
        res = self.kb.search(formatted_codes,top_k=example_num, similarity_thredhold=s_thre,rate_thredhold=r_thre)
        improve_exampless = []
        for hits in res:
            if len(hits)<1:
                print("!Can not find similar improvement case")
                # return EXAMPLE_NOT_FOUNDED
            else:
                improve_examples = [(hit["entity"]["summary"],hit["distance"]) for hit in hits]
                improve_exampless.append(improve_examples)
        # print(improve_examples)
        # 4. 生成prompt
        # func_signature = f"def {target_function_name}("
        prompts = [gen_prompt(origin_code,target_function_names,improve_examples) for origin_code,target_function_names,improve_examples in zip(origin_codes,target_function_names,improve_exampless)]

        optimized_codes = []
        for prompt in prompts:
            ans = self.fpg([prompt])# 5. 传递给fasterc_generator.py，生成改进代码
            code = self.extract_code(ans)# 6. 代码提取
            optimized_codes.append(code)
        # 7. 输出新的代码
        for i, code in enumerate(optimized_codes):
            print(f"=== Optimized code {i} ===\n{code}")
        # 8. return 0;
        return 0
    
if __name__ == "__main__":
    pipeline = InferencePipeline()
    
    status_code = pipeline.inference()