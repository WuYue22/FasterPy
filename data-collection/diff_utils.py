import difflib
import subprocess
import tempfile
import os

def diff(old_code:str,new_code:str,context_len:int=3):
    old_code = old_code.splitlines()
    new_code = new_code.splitlines()
    patch_lines = difflib.unified_diff(old_code,new_code,"old_code","new_code","None","None",n=context_len)
    patch='\n'.join([line.rstrip('\n') for line in patch_lines])+"\n"
    return patch
    # print(patch_lines)
    


def syn(code:str, patch:str, reverse:bool=False)->str:
    if not patch:
        return code
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    retried=False
    try:
        while True:
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
                    text=True
                )
            if process.returncode != 0:
                if not retried:
                    retried=True
                    code = code.rstrip('\n')
                else:
                    print("code"+f"-"*25)
                    print(code)
                    print("patch"+f"-"*25)
                    print(patch)
                    print(f"-"*25)
                    print(process.stdout)
                    raise Exception(f"Patch failed: {process.stderr}")
            break
        patched_code = open(temp_file_path).read()
        return patched_code
    finally:
        os.remove(temp_file_path)