%%capture
!pip install unsloth
!pip install --force-reinstall  --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
!pip install wandb
!pip install --force-reinstall  --upgrade torch


import os

os.environ['WANDB_API_KEY'] = ''

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    # Can select any from the below:
    # "unsloth/Qwen2.5-0.5B", "unsloth/Qwen2.5-1.5B", "unsloth/Qwen2.5-3B"
    # "unsloth/Qwen2.5-14B",  "unsloth/Qwen2.5-32B",  "unsloth/Qwen2.5-72B",
    # And also all Instruct versions and Math. Coding verisons!
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset
import os

# Define system prompt template with placeholders for Question, Response, and Reasoning Content
system_prompt = """以下是一个任务描述，配有一个提供进一步背景的提问。请根据问题提供一个适当的回答。
在回答之前，请仔细思考问题，并简洁地创建一个逐步的推理链，确保答案逻辑清晰、准确。

### Instruction:
你是一位经验丰富的心理咨询师，擅长倾听并提供专业建议。请根据以下用户的问题，提供真诚、温暖且专业的回应，避免提供未经验证的信息。

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

# Define the EOS token
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    input = examples["input"]
    reasoning_content = examples["reasoning_content"]
    content = examples["content"]

    # Initialize an empty list to store formatted text
    texts = []

    # Loop through input, reasoning_content, and content to create the formatted text for each sample
    for input_text, reasoning_text, content_text in zip(input, reasoning_content, content):
        # Format the system_prompt with input, reasoning, and content
        text = system_prompt.format(input_text, reasoning_text, content_text) + EOS_TOKEN
        texts.append(text)

    # Return the formatted text as a dictionary
    return {
        "text": texts,
    }

# Load the dataset
dataset = load_dataset("Kedreamix/psychology-10k-Deepseek-R1-zh", split="train[0:40]")

# Split the dataset using datasets' built-in train_test_split method
train_test_split_result = dataset.train_test_split(test_size=0.2, seed=42)  # 80% for training, 20% for temp (validation + test)
train_dataset = train_test_split_result["train"]
temp_dataset = train_test_split_result["test"]

# Split the temp dataset into validation and test datasets
val_test_split_result = temp_dataset.train_test_split(test_size=0.5, seed=42)  # Split remaining 20% equally for validation and test
val_dataset = val_test_split_result["train"]
test_dataset = val_test_split_result["test"]

# Apply formatting to the train, test, and validation datasets
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
test_dataset = test_dataset.map(formatting_prompts_func, batched=True)


# Create a directory to save the formatted datasets
save_path = "data"
os.makedirs(save_path, exist_ok=True)

# Save the datasets as JSON files
train_dataset.to_json(os.path.join(save_path, "train.json"))
val_dataset.to_json(os.path.join(save_path, "val.json"))
test_dataset.to_json(os.path.join(save_path, "test.json"))

# print the data
test_dataset ["text"][0]
test_dataset ["text"][0]
test_dataset ["text"][0]

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb



wandb.login()

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="lucasblack997-none",
    # Set the wandb project where this run will be logged.
    project="deepseek-r1-finetuing-v1",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1.5e-5,
        "architecture": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "dataset": "psychology-10k-zh",
        "epochs": 3,  # Match withnum_train_epochs
        "max_seq_length": max_seq_length,
        "batch_size": 32,
        "gradient_accumulation": 16,
    },
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        output_dir = "outputs",
        per_device_train_batch_size = 2, # 2 is enough
        gradient_accumulation_steps = 16,
        # batch size =  per_device_train_batch_size x  per_device_train_batch_size
        #warmup_steps = 5,
        warmup_ratio= 0.03,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = 2000,
        learning_rate = 1.5e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        save_total_limit = 1,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb", # Use this for WandB etc
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        eval_strategy="epoch",
        save_strategy="epoch",
    ),
)

start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()
trainer.save_model("outputs/final_model")

# Record the final model after training
wandb.finish()

# system_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    system_prompt.format(
        "你是一位经验丰富的心理咨询师，擅长倾听并提供专业建议。请根据以下用户的问题，提供真诚、温暖且专业的回应，避免提供未经验证的信息。", # instruction
        "我最近好焦虑", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# system_prompt = You MUST copy from above!

inputs = tokenizer(
[
    system_prompt.format(
        "你是一位经验丰富的心理咨询师，擅长倾听并提供专业建议。请根据以下用户的问题，提供真诚、温暖且专业的回应，避免提供未经验证的信息。", # instruction
        "我最近一直感到非常焦虑，不知道该如何应对", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024)

# Save to q4_k_m GGUF
if False:  model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )
