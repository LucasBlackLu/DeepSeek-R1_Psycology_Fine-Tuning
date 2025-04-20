# DeepSeek-R1: Fine-tuing model 

Welcome to the DeepSeek-R1-Psycology repository! This project is designed to work with AI-driven emotional analysis, particularly focusing on understanding and intervening in emotional crises through machine learning models. It utilizes the `DeepSeek-R1-Distill-Qwen-1.5B` model for various tasks, including text-based classification related to psychological analysis.

## Setup

### Requirements

To get started with this project, you need to install the following dependencies:

```bash
pip install unsloth
pip install --force-reinstall --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
pip install wandb
```

### Environment Setup

Make sure you have set up the necessary environment variables, especially for `wandb`:

```python
import os
os.environ['WANDB_API_KEY'] = 'your_wandb_token_here'
```

## Usage

1. Clone the repository and navigate to the project folder.
2. Install dependencies.
3. Load the `DeepSeek-R1` model and begin training or inference.

Feel free to modify and extend the project as needed for your specific use case.

