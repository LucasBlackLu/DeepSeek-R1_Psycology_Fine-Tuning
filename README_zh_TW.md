# DeepSeek-R1: 微調模型

歡迎使用DeepSeek-R1-Psycology 項目！本項目旨在透過機器學習模型實現情感分析，尤其關注情緒危機的理解與干預。它利用`DeepSeek-R1-Distill-Qwen-1.5B`蒸餾模型進行心理學分析相關的文本分類等任務。

## 安裝

### 依賴項

要開始使用此項目，您需要安裝以下依賴項：

```bash
pip install unsloth
pip install --force-reinstall --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
pip install wandb
```

### 環境設置

確保您已設置好必要的環境變數，特別是`wandb`的API金鑰：

```python
import os
os.environ['WANDB_API_KEY'] = 'your_wandb_token_here'
```

## 使用

1. 克隆倉庫並導航到項目資料夾。
2. 安裝依賴項。
3. 加載`DeepSeek-R1`模型並開始訓練或推理。

根據您的具體需求，可以修改和擴展項目。
