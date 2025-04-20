# DeepSeek-R1: 微调模型

欢迎使用DeepSeek-R1-Psycology 项目！本项目旨在通过机器学习模型实现情感分析，尤其关注情绪危机的理解与干预。它利用`DeepSeek-R1-Distill-Qwen-1.5B`蒸馏模型进行心理学分析相关的文本分类等任务。

## 安装

### 依赖项

要开始使用此项目，您需要安装以下依赖项：

```bash
pip install unsloth
pip install --force-reinstall --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
pip install wandb
```

### 环境设置

确保您已设置好必要的环境变量，特别是`wandb`的API密钥：

```python
import os
os.environ['WANDB_API_KEY'] = 'your_wandb_token_here'
```

## 使用

1. 克隆仓库并导航到项目文件夹。
2. 安装依赖项。
3. 加载`DeepSeek-R1`模型并开始训练或推理。

根据您的具体需求，可以修改和扩展项目。

