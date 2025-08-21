import os
from multiprocessing import Process, Queue
import time
import queue
from cirron import Collector
import io
import re
import sys
DONE_CODE="<|DONE|>_for_unique_8sadtq4iovbtdmvhruhgfgkgkfyigc6iuhfidio7937qhiorqwn4"

# 重载close方法，防止被exec中执行的代码所关闭，以至于无法获取输出
class UnclosableStringIO(io.StringIO):
    def close(self):
        pass  # 阻止被关闭

WRAP_FUNC_HEAD="""
import sys
import traceback
try:
"""
     
# use complex variable name cirron_collected_instruction_count to avoid possible name conflict
WRAP_FUNC_TAIL="""
except Exception as e:
    traceback.print_exc(file=sys.stderr)
    print(f'exception={e}', file=sys.stderr)
    
"""

def inject_wrapper(code):
    lines=code.strip().splitlines()
    lines=[f" "*4+line for line in lines]
    injected= "\n".join([WRAP_FUNC_HEAD,"\n".join(lines),WRAP_FUNC_TAIL])
    # 将open(0)替换为 sys.stdin，以获取重定向的输入
    return re.sub(r'\bopen\(0\)', 'sys.stdin', injected)

    
def terminate_process(p):
    if p.is_alive():
        p.terminate()
        p.join()
        

def exec_in_process(iq, oq, auto_terminate:int):
    import threading
    # 需要在新线程中运行exec(code,args)，以本进程（exec_in_process）被待测试code中的sys.exit(0)意外终止退出。这是大坑。
    def exec_in_thread(code,global_scope):
        t = threading.Thread(target=exec, args=(code,global_scope))
        t.start()
        t.join()
    # 备份stdin out err
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin=sys.stdin
    origin_stdin_fd = os.dup(0)

    # 初始化cirron_collected_instruction_count，-1为异常值
    cirron_collected_instruction_count=-1
    def redirect_stdinouterr(input_path):
        f =open(input_path,'r')
        sys.stdout = UnclosableStringIO()
        sys.stderr = UnclosableStringIO()
        sys.stdin=f
        os.dup2(f.fileno(), 0)
        return f
        
    def restore_stdinouterr(f):
        f.close()
        os.dup2(origin_stdin_fd, 0)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin=old_stdin
    pid=os.getpid()
    while True:
        # prepare
        try:
            # set timeout. If no more jobs are available in 15s, assuming all tasks have donw, exit.
            code,input_path=iq.get(timeout=auto_terminate)
            if code == DONE_CODE:
                break
            injected_code=inject_wrapper(code)
            # 每次运行前都要重置sandbox_global，避免sandbox_global被污染，影响后续的执行
            sandbox_global={'Collector':Collector,'__name__':'__main__'} 
            
            # redirect to simulate a process
            f = redirect_stdinouterr(input_path)
            compiled = compile(injected_code, filename="my_exec_code.py", mode="exec")
            exec_in_thread(compiled,sandbox_global)
            stdout_output = sys.stdout.getvalue().strip()
            stderr_output = sys.stderr.getvalue().strip()
            restore_stdinouterr(f)
            
            # collect output
            if stderr_output=='':
                oq.put_nowait((stderr_output, stdout_output.strip()))
            else:
                oq.put_nowait((stderr_output, None, 0))
            continue
        except queue.Empty:
            # restore_stdinouterr()
            print(f"No more jobs, exit {pid}")
            break
        except queue.Full:
            print(f"Output queue is full {pid}")
            oq.put_nowait(("Output queue is full", None, 0)),
        except Exception as e:
            oq.put_nowait((str(e), None, 0))
            continue
    
    # restore_stdinouterr()
def get_accuracy(output: str, ground_truth: str) -> float:
    """
    Compare the output of the code with the ground truth.
    """
    num_correct = 0
    ground_truth_lines = ground_truth.strip().splitlines()
    output_truth_lines = output.strip().splitlines()
    for gen_output, ground_truth_output in zip(output_truth_lines, ground_truth_lines):
        is_corr = gen_output == ground_truth_output
        if not is_corr:
            try:
                gen_output = float(gen_output)
                ground_truth_output = float(ground_truth_output)
                is_corr = abs(gen_output - ground_truth_output) < 1e-3
            except:
                pass
        num_correct += int(is_corr)
    if len(ground_truth_lines)==0:
        if len(output_truth_lines)==0:
            return 1
        else:
            return 0
    else:
        return num_correct / len(ground_truth_lines)

def exec_with_timeout(iq,oq,p,code:str, input_path:str,timeout:int=2):
    iq.put((code,input_path))
    try:
        # result=oq.get()
        result=oq.get(timeout=timeout)
        return result
    except queue.Empty:
        terminate_process(p)
        return('TimeoutError',None,0)
    
import numpy as np
import pandas as pd
from tqdm import tqdm
if __name__ == "__main__":
    test_cases_path="/Users/hanminghao/Desktop/study/毕业设计/eval/codenet/public_test_cases"
    df=pd.read_json('/Users/hanminghao/Downloads/pie-python_splits/PIE4DatabaseSumIdsPid.jsonl',lines=True)
    # df=df.loc[:100,:]
    # Split the dataframe into 4 equal parts
    # df_split = np.array_split(df, 4)
    iq = Queue(maxsize=1)
    oq = Queue(maxsize=1)
    p=None
    def __get_process(auto_terminate:int=5):
        global p
        try:
            if p is None or not p.is_alive():
                p=Process(target=exec_in_process, args=(iq,oq,auto_terminate))
                p.start()
            return p
        except Exception as e:
            print(f"__get_process err{e}")
            raise e
    def try_run(row):
        problem_id=row['problem_id']
        input_path=os.path.join(test_cases_path,problem_id,"input.0.txt")
        output_path=os.path.join(test_cases_path,problem_id,"output.0.txt")
        global iq,oq
        if oq.full():
            oq=Queue(maxsize=1)
        if iq.full(): 
            iq = Queue(maxsize=1)
        slow_code = row['code_v0_no_empty_lines']
        with open(output_path,'r') as f:
            output=f.read()
        p=__get_process()
        slow_result=exec_with_timeout(iq,oq,p,slow_code,input_path,timeout=1)
        if slow_result=='TimeoutError' or slow_result[1] is None or get_accuracy(slow_result[1],output)!=1:
            slow_pass=False
        else:
            slow_pass=True
        if oq.full():
            oq=Queue(maxsize=1)
        if iq.full(): 
            iq = Queue(maxsize=1)
        p=__get_process()
        reference_code=row['code_v1_no_empty_lines']
        reference_result=exec_with_timeout(iq,oq,p,reference_code,input_path,timeout=1)
        if reference_result=='TimeoutError' or reference_result[1] is None or get_accuracy(reference_result[1],output)!=1:
            reference_pass=False
        else:
            reference_pass=True
        return pd.Series([slow_pass,reference_pass])
    tqdm.pandas()
    df[['slow_pass','reference_pass']]=df.progress_apply(try_run,axis=1)
    df.to_json('/Users/hanminghao/Downloads/pie-python_splits/PIE4DatabaseSumIdsPidPass.jsonl',lines=True,orient='records')
    # print(df['slow_pass'])
    # print(df['reference_pass'])