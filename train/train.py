import os
import torch
import swanlab
from swanlab.integration.transformers import SwanLabCallback
from swift.llm import get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
from swift.utils import get_logger, find_all_linears, get_model_parameter_info, plot_images, seed_everything
from swift.tuners import Swift, LoraConfig
from swift.trainers import Trainer, TrainingArguments
import sys
import json

def print_gpu_status():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory  # 总显存
        allocated_memory = torch.cuda.memory_allocated(gpu_id)  # 已分配的显存
        reserved_memory = torch.cuda.memory_reserved(gpu_id)  # 预留的显存（缓存）

        print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"总显存: {total_memory / 1024**3:.2f} GB")
        print(f"已分配显存: {allocated_memory / 1024**3:.2f} GB")
        print(f"已预留显存: {reserved_memory / 1024**3:.2f} GB")
        print(f"剩余可用显存: {(total_memory - reserved_memory) / 1024**3:.2f} GB")
output_dir="output-src-qw"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = get_logger()
seed_everything(42)
model_id_or_path = "model/Qwen2.5-7B-Instruct"#'Qwen/Qwen2.5-7B-Instruct'  # model_id or model_path
system = 'You are a coder specialized in code efficiency improvement.'

max_length=1024
data_seed = 42
split_dataset_ratio = 0.01  # Split validation set
num_proc = 4  # The number of processes for data loading.
# The following two parameters are used to override the placeholders in the self-cognition dataset.
model_name = ['更快的Py', 'FasterPy']  # The Chinese name and English name of the model
model_author = ['', 'Yue Wu']  # The Chinese name and English name of the model author
# lora
lora_rank = 8
lora_alpha = 32
output_dir = os.path.abspath(os.path.expanduser(output_dir))
logger.info(f'output_dir: {output_dir}')
print_gpu_status()
model, tokenizer = get_model_tokenizer(model_id_or_path)
print_gpu_status()
logger.info(f'model_info: {model.model_info}')
template = get_template(model.model_meta.template, tokenizer, default_system=system, max_length=max_length,truncation_strategy='delete')
template.set_mode('train')

lora_config = LoraConfig(task_type='CAUSAL_LM', r=lora_rank, lora_alpha=lora_alpha,
    target_modules=['up_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj', 'v_proj', 'k_proj'])
model = Swift.prepare_model(model, lora_config)
logger.info(f'model: {model}')
model_parameter_info = get_model_parameter_info(model)
logger.info(f'model_parameter_info: {model_parameter_info}')
# dataset
dataset=["train_data_msg.jsonl"]
# Download and load the dataset, split it into a training set and a validation set,
# and encode the text data into tokens.
train_dataset, val_dataset = load_dataset(dataset, split_dataset_ratio=split_dataset_ratio, num_proc=num_proc,
        model_name=model_name, model_author=model_author, seed=data_seed)
# print(train_dataset.shape)
# print(val_dataset.shape)
# train_data_nums = min(train_dataset.shape[0],6000)
train_dataset = train_dataset.shuffle(seed=42)
val_dataset = val_dataset.shuffle(seed=42)
logger.info(f'dataset name: {dataset}')
logger.info(f'train_dataset: {train_dataset}')
logger.info(f'val_dataset: {val_dataset}')
logger.info(f'train_dataset[0]: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc,batch_size=4)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc,batch_size=4)
logger.info(f'encoded_train_dataset[0]: {train_dataset[0]}')

# Print a sample
template.print_inputs(train_dataset[0])

def get_dataset_size(dataset):
    """统计数据集的总字节数"""
    total_size = 0
    for data in dataset:
        total_size += sys.getsizeof(data)  # 计算每个数据项的大小
        for key, value in data.items():
            total_size += sys.getsizeof(key)  # 计算键的大小
            if isinstance(value, list):
                total_size += sum(sys.getsizeof(i) for i in value)  # 计算列表内部元素的大小
            else:
                total_size += sys.getsizeof(value)
    return total_size

# 计算 `encoded_train_dataset` 的字节数
dataset_size_bytes = get_dataset_size(train_dataset)
logger.info(f"Encoded Train Dataset Size: {dataset_size_bytes / 1024 / 1024:.2f} MB")

learning_rate = 1e-3
num_train_epochs = 2
with open('..\config.json', 'r') as file:
    config = json.load(file)
SWAN_API = config['SWAN_API']
try:
    swanlab.login(api_key=SWAN_API, save=True)
except Exception as e:
    print(e)
swanlab.init(
    # 设置项目名
    project="Swift-Qw-7B-FT-on-ms-hpc",

    # 设置超参数
    config={
        "learning_rate": learning_rate,
        "architecture": "Transformer",
        "dataset": " ".join(dataset),
        "epochs": num_train_epochs
    },
    mode="offline",
    save_dir="swanlab_logs"
)

swanlab_callback = SwanLabCallback(project="Qwen2.5-7B-fintune", experiment_name="Qwen/Qwen2.5-7B-Instruct")
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=learning_rate,
    # resume_from_checkpoint=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_checkpointing=True,
    bf16=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to='none',
    logging_first_step=True,
    save_strategy='steps',
    save_steps=10,
    eval_strategy='steps',
    eval_steps=10,
    gradient_accumulation_steps=4,
    num_train_epochs=num_train_epochs,
    metric_for_best_model='loss',
    save_total_limit=5,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

model.enable_input_require_grads()  # Compatible with gradient checkpointing
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
    callbacks=[swanlab_callback],
)
torch.cuda.empty_cache()

print_gpu_status()
try:
    trainer.train()
    # trainer.train(resume_from_checkpoint="output-src-qw/checkpoint-556")
    last_model_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            print_gpu_status()
            last_model_checkpoint = trainer.state.last_model_checkpoint
            logger.info(f'last_model_checkpoint: {last_model_checkpoint}')
            trainer.train(resume_from_checkpoint=last_model_checkpoint)
        else:
            raise e
except Exception as e:
    print(e)
except KeyboardInterrupt:
    swanlab.finish()
finally:
    swanlab.finish()