import os
from multiprocessing import Process, Queue
import time
import queue
from cirron import Collector
import io
import re
import sys
TAKEN_CODE="<|TimeOrInstructionTaken|>_for_unique_8sadtq4iovbtdmvhruhgfgkgkfyigc6iuhfidio7937qhiorqwn4"

# 重载close方法，防止被exec中执行的代码所关闭，以至于无法获取输出
class UnclosableStringIO(io.StringIO):
    def close(self):
        pass  # 阻止被关闭

WRAP_FUNC_HEAD="""
import sys
import traceback
try:
    # with Collector() as collector:
"""
     
# use complex variable name cirron_collected_instruction_count to avoid possible name conflict
WRAP_FUNC_TAIL="""
except Exception as e:
    traceback.print_exc(file=sys.stderr)
    print(f'exception={e}', file=sys.stderr)
 
# finally:
#     instruction_count=collector.counters.instruction_count
"""

def inject_wrapper(code):
    lines=code.strip().splitlines()
    lines=[f" "*4+line for line in lines]
    injected= "\n".join([WRAP_FUNC_HEAD,"\n".join(lines),WRAP_FUNC_TAIL])
    # 将open(0)替换为 sys.stdin，以获取重定向的输入
    return re.sub(r'\bopen\(0\)', 'sys.stdin', injected)


THE_PROCESS=None
THE_INPUT_QUEUE=None
THE_OUTPUT_QUEUE=None

def clear_queue(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass  # 队列清空完毕

def __get_iq():
    global THE_INPUT_QUEUE
    try:
        if THE_INPUT_QUEUE is None:
            THE_INPUT_QUEUE=Queue(maxsize=1)
        return THE_INPUT_QUEUE
    except Exception as e:
        print(f"__get_iq err{e}")
        raise e
def __get_oq():
    global THE_OUTPUT_QUEUE
    try:
        if THE_OUTPUT_QUEUE is None:
            THE_OUTPUT_QUEUE=Queue(maxsize=1)
        return THE_OUTPUT_QUEUE
    except Exception as e:
        print(f"__get_oq err{e}")
        raise e

def __get_process(auto_terminate:int,measurement:str='time'):
    global THE_PROCESS
    try:
        if THE_PROCESS is None or not THE_PROCESS.is_alive():
            THE_PROCESS=Process(target=exec_in_process, args=(__get_iq(),__get_oq(),auto_terminate,measurement))
            THE_PROCESS.start()
        return THE_PROCESS
    except Exception as e:
        print(f"__get_process err{e}")
        raise e
    
def terminate_process(p):
    if p.is_alive():
        p.terminate()
        p.join()
    global THE_INPUT_QUEUE, THE_OUTPUT_QUEUE
    THE_INPUT_QUEUE=None
    THE_OUTPUT_QUEUE=None
        
def my_exec(code, global_scope):
    try:
        with Collector() as collector:
            exec(code,global_scope)
    finally:
        if collector.counters.instruction_count/1000000>10000:
                    print(f"instruction_taken too high: {collector.counters.instruction_count}")
        
        instruction_taken=int(collector.counters.instruction_count)/1000000
        global_scope[TAKEN_CODE]=instruction_taken
        
def my_exec_time(code, global_scope):
    try:
        start=time.time()
        exec(code,global_scope)
    finally:
        elapse_ms=(time.time()-start)*1000
        global_scope[TAKEN_CODE]=elapse_ms
        
def exec_in_process(iq, oq, auto_terminate:int,measurement:str='time'):
    import threading
    # 需要在新线程中运行exec(code,args)，以本进程（exec_in_process）被待测试code中的sys.exit(0)意外终止退出。这是大坑。
    if measurement=='time':
        
        def exec_in_thread(code,global_scope):
            t = threading.Thread(target=my_exec_time, args=(code,global_scope))
            t.start()
            t.join()
    elif measurement=='instruction':
        def exec_in_thread(code,global_scope):
            t = threading.Thread(target=my_exec, args=(code,global_scope))
            t.start()
            t.join()
    else:
        raise Exception(f"unknown measurement {measurement}")
    # 备份stdin out err
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin=sys.stdin
    origin_stdin_fd = os.dup(0)

    # 初始化cirron_collected_instruction_count，-1为异常值
    def redirect_stdinouterr(f):
        sys.stdout = UnclosableStringIO()
        sys.stderr = UnclosableStringIO()
        sys.stdin=f
        os.dup2(f.fileno(), 0)
        
    def restore_stdinouterr():
        os.dup2(origin_stdin_fd, 0)
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin=old_stdin
    pid=os.getpid()
    # os.sched_setaffinity(pid, {0})
    # print(f"pid={pid}")
    while True:
        # prepare
        try:
            # set timeout. If no more jobs are available in 15s, assuming all tasks have donw, exit.
            code,input_path=iq.get(timeout=auto_terminate)
            # if code == DONE_CODE:
            #     break
            injected_code=inject_wrapper(code)
            # 每次运行前都要重置sandbox_global，避免sandbox_global被污染，影响后续的执行
            taken=None
            sandbox_global={'__name__':'__main__',TAKEN_CODE:taken,'Collector':Collector} 
            
            # redirect to simulate a process
            
            with open(input_path,'r') as f:
                redirect_stdinouterr(f)
                compiled = compile(injected_code, filename="my_exec_code.py", mode="exec")
                exec_in_thread(compiled,sandbox_global)
                stdout_output = sys.stdout.getvalue().strip()
                stderr_output = sys.stderr.getvalue().strip()
            restore_stdinouterr()
            if sandbox_global[TAKEN_CODE]==None:
                print('stop hre')
            # collect output
            if stderr_output=='':
                oq.put_nowait((stderr_output, stdout_output.strip(), sandbox_global[TAKEN_CODE]))
            else:
                # restore_stdinouterr()
                # print(f"oq full{oq.full()}")
                oq.put_nowait((stderr_output, None, 0))
            continue
        except queue.Empty:
            # restore_stdinouterr()
            print(f"No more jobs, exit {pid}")
            print()
            break
        except queue.Full:
            print(f"Output queue is full {pid}")
            oq.put_nowait(("Output queue is full", None, 0)),
        except Exception as e:
            # print(f"other error: {e} {pid}")
            oq.put_nowait((str(e), None, 0))
            continue
    
    # restore_stdinouterr()


def exec_with_timeout(code:str, input_path:str,timeout:int=None,measurement:str='time'):
    iq = __get_iq()
    oq = __get_oq()
    p = __get_process(auto_terminate=4,measurement=measurement)
    if oq.full():
        raise RuntimeError("Output queue is full")
    iq.put((code,input_path))
    # start_time=time.time()
    try:
        result=oq.get(timeout=timeout)
        # result=oq.get()
        return result
    except queue.Empty:
        # print(code,flush=True)
        # print(input_path,flush=True)
        # print(f"timeout, time cost {time.time()-start_time}")
        terminate_process(p)
        return('TimeoutError',None,0)


if __name__ == "__main__":
    code="""
import sys
x=int(input())
for i in range(x):
    print(i)
"""
    input_path="src/a.py"
    start_time=time.time()
    for i in range(1):
        result = exec_with_timeout(code,input_path,1)
        print(result)
    print(time.time()-start_time)