#!/usr/bin/env python
# coding: utf-8

# ### Inittial the notebook

# In[ ]:


import os

# Set wandb API secret token
os.environ['WANDB_API_KEY'] = 'plz input your token'


# ### Install Dependencies 

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install unsloth\n!pip install --force-reinstall  --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git\n!pip install wandb\n')


# ### Load the model 

# In[ ]:


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


# ### Configure the fine-tuning

# In[ ]:


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


# ### Dataset Preparation

# In[ ]:


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
dataset = load_dataset("Kedreamix/psychology-10k-Deepseek-R1-zh")

# Get the train, test, and validation splits
train_dataset = dataset["train"].select(range(0, 500))  # 0-499
test_dataset = dataset["train"].select(range(500, 625))  # 500-624
val_dataset = dataset["train"].select(range(625, 750))  # 625-749

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


# ### Train the model 

# In[ ]:


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
        "learning_rate": 2e-4,
        "architecture": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "dataset": "psychology-10k-zh",
        "epochs": 10,  # Match withnum_train_epochs
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
        gradient_accumulation_steps = 8,
        # batch size =  per_device_train_batch_size x  per_device_train_batch_size
        warmup_steps = 5,
        num_train_epochs = 3, # Set this for 1 full training run.
        #max_steps = 2000,
        learning_rate = 2e-4,
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


# ### Show current memory stats

# In[ ]:


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# ### Start Training

# In[ ]:


trainer_stats = trainer.train()
trainer.save_model("outputs/final_model")

# Record the final model after training 
wandb.finish()


# ### Import Chinese for matplotlib

# In[ ]:


import matplotlib
import matplotlib.font_manager as fm
get_ipython().system('wget -O MicrosoftJhengHei.ttf https://github.com/a7532ariel/ms-web/raw/master/Microsoft-JhengHei.ttf')
get_ipython().system('wget -O ArialUnicodeMS.ttf https://github.com/texttechnologylab/DHd2019BoA/raw/master/fonts/Arial%20Unicode%20MS.TTF')

fm.fontManager.addfont('MicrosoftJhengHei.ttf')
matplotlib.rc('font', family='Microsoft Jheng Hei')

fm.fontManager.addfont('ArialUnicodeMS.ttf')
matplotlib.rc('font', family='Arial Unicode MS')


# ### Display the linear Fit Plot

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 假設你有一個數據集，x 是特徵，y 是目標變量
# 這裡生成一些示例數據
np.random.seed(42)
x = np.random.rand(100, 1) * 10  # 特徵
y = 2.5 * x + np.random.randn(100, 1) * 2  # 目標變量（帶噪聲）

# 將數據集劃分為訓練集、驗證集和測試集
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 訓練線性回歸模型
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# 在驗證集上預測
y_val_pred = linear_model.predict(x_val)

# 繪製驗證集上的線性擬合圖
plt.figure(figsize=(8, 6))
plt.scatter(x_val, y_val, color='blue', label='驗證集真實值')
plt.plot(x_val, y_val_pred, color='red', linewidth=2, label='擬合線')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('驗證集上的線性擬合圖')
plt.legend()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 假設你有一個數據集，x 是特徵，y 是目標變量
# 這裡生成一些示例數據
np.random.seed(42)
x = np.random.rand(100, 1) * 10  # 特徵
y = 2.5 * x + np.random.randn(100, 1) * 10  # 目標變量（增加噪聲）

# 將數據集劃分為訓練集、驗證集和測試集
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 訓練線性回歸模型
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# 在所有數據集上進行預測
y_train_pred = linear_model.predict(x_train)
y_val_pred = linear_model.predict(x_val)
y_test_pred = linear_model.predict(x_test)

# 創建一個更細的 x 值範圍用於繪製擬合線
x_range = np.linspace(min(x), max(x), 1000).reshape(-1, 1)

# 在這個新的 x_range 上預測
y_range_pred = linear_model.predict(x_range)

# 設置繪圖
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 橫向排列三個子圖

# 1. 繪製訓練集的線性擬合圖
axes[0].scatter(x_train, y_train, color='blue', label='訓練集真實值')
axes[0].plot(x_range, y_range_pred, color='blue', linewidth=2, label='訓練集擬合線')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('訓練集上的線性擬合圖')
axes[0].legend()

# 2. 繪製驗證集的線性擬合圖
axes[1].scatter(x_val, y_val, color='orange', label='驗證集真實值')
axes[1].plot(x_range, y_range_pred, color='orange', linewidth=2, linestyle='--', label='驗證集擬合線')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('驗證集上的線性擬合圖')
axes[1].legend()

# 3. 繪製測試集的線性擬合圖
axes[2].scatter(x_test, y_test, color='green', label='測試集真實值')
axes[2].plot(x_range, y_range_pred, color='green', linewidth=2, linestyle=':', label='測試集擬合線')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')
axes[2].set_title('測試集上的線性擬合圖')
axes[2].legend()

# 顯示圖表
plt.tight_layout()
plt.show()


# ### Inference 

# In[ ]:


# system_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    system_prompt.format(
        "你是一位经验丰富的心理咨询师，擅长倾听并提供专业建议。请根据以下用户的问题，提供真诚、温暖且专业的回应，避免提供未经验证的信息。", # instruction
        "我想离开这个悲伤的世界", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)


# ### Saving, loading finetuned models

# In[ ]:


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


# In[ ]:


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
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# ### Quantizatative and save model

# In[ ]:


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

