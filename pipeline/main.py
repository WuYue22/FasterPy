# The main file for the program
import argparse

from inference_pipeline import InferencePipeline

def optimize_single_function(pipeline):
    code_path = input("Please enter the path to the original code file:").strip()
    func_name = input("Please enter the function name to optimize:").strip()

    with open(code_path, "r", encoding="utf-8") as f:
        origin_code = f.read()

    status_code = pipeline.inference(
        origin_code=origin_code,
        target_function_name=func_name,
        example_num=3,
        r_thre=0.5,
        s_thre=0.1
    )

    if status_code == 0:
        print("Optimization completed!")
    else:
        print("Optimization failed.")

def optimize_multiple_functions(pipeline):
    code_path = input("Please enter the path to the original code file:").strip()
    func_names_str = input("Please enter the list of function names to optimize (comma-separated):").strip()
    func_names = [name.strip() for name in func_names_str.split(",")]

    with open(code_path, "r", encoding="utf-8") as f:
        origin_code = f.read()

    # 所有函数都在同一个文件里
    origin_codes = [origin_code] * len(func_names)

    status_code = pipeline.inference_batch(
        origin_codes=origin_codes,
        target_function_names=func_names,
        example_num=3,
        r_thre=0.5,
        s_thre=0.1
    )

    if status_code == 0:
        print("Batch optimization completed!")
    else:
        print("Optimization failed.")


def main(model_path="../models/Qwen2.5-7B-Instruct-ft"):
    pipeline = InferencePipeline(model_path)

    while True:
        print("\n=== Code Optimization Menu ===")
        print("1. Optimize a single function")
        print("2. Optimize multiple functions")
        print("3. Exit")
        choice = input("Please select an option:").strip()

        if choice == "1":
            try:
                optimize_single_function(pipeline)
            except Exception as e:
                print(f"An error occurred in option 1: {e}")

        elif choice == "2":
            try:
                optimize_multiple_functions(pipeline)
            except Exception as e:
                print(f"An error occurred in option 2: {e}")
        elif choice == "3":
            print("Program exited.")
            break
        else:
            print("Invalid option, please try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FasterPy")
    parser.add_argument("-modelpath", type=str, default="../models/Qwen2.5-7B-Instruct-ft", help="Local path or download URL of the model")
    args = parser.parse_args()
    main(args.modelpath)
    main()
