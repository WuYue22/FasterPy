# This file is responsible for synthesizing the generated patch with original code to output the final fastered code
import subprocess
import os
import difflib
import tempfile

no_new_line_str="\ No newline at end of file"
def diff(old_code:str,new_code:str,context_len:int=3):
    old_code_lines = old_code.splitlines()
    new_code_lines = new_code.splitlines()
    patch_lines = difflib.unified_diff(old_code_lines,new_code_lines,"old_code","new_code","None","None",n=context_len)
    patch_lines = list(patch_lines)
    if not old_code.endswith('\n'):
        patch_lines.insert(-1,no_new_line_str)
    if not new_code.endswith('\n'):
        patch_lines.append(no_new_line_str)
    patch='\n'.join([line.rstrip('\n') for line in patch_lines])
    return patch
    # print(patch_lines)
    


def syn(code:str, patch:str, reverse:bool=False)->str:
    if not patch:
        return code

    retried=False
    while True:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        try:
            if not reverse:
                process = subprocess.run(
                    ["patch", temp_file_path],
                    input=patch,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                process = subprocess.run(
                    ["patch", "-R", temp_file_path],
                    input=patch,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=3,
                    text=True
                )
            if process.returncode != 0:
                if not retried:
                    retried=True
                    code = code.rstrip('\n')
                    continue
                else:
                    print("patch return code:", process.returncode)
                    print("patch stdout:", process.stdout)
                    print("patch stderr:", process.stderr)
                    return ""
            else: break
        
        except subprocess.TimeoutExpired:
            if not retried:
                retried=True
                code = code.rstrip('\n')
                continue
            else: 
                os.remove(temp_file_path)
                return ""
        except Exception as e:
            os.remove(temp_file_path)
            return ""
    patched_code = open(temp_file_path).read()
    os.remove(temp_file_path)
    return patched_code

def style_git2unified(git_patch:str):
    # 将输入字符串分成行
    lines = git_patch.splitlines()
    diff_u = lines[2:]
    
    return '\n'.join(diff_u)



    
if __name__ == "__main__":
    code1=""
    with open("example.py","r") as file:
        code1 = file.read()
    code2=""
    with open("example2.py","r") as file:
        code2 = file.read()
    # patch=patch.strip()
    # print(patch)
    patch_lines = diff(code1,code2)
    # print("\n".join(patch_lines))
    # for line in patch_lines:
    #     print(line)
    if not patch_lines:
        print("没有变化，跳过 patch 过程")
    else: 
        syned = syn(code1,patch_lines)
        print("syned")
        print(syned)
   
