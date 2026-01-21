# CLI-LLM
# CLI-LLM Running Guide

## Prerequisites
- Python 3.8+ installed
- Sufficient disk space (≥20GB recommended for storing model weights and training files)
- CUDA-enabled GPU (≥16GB VRAM recommended for accelerated training)

## 1. Environment Setup
Install project dependencies with the following command:
```bash
pip install -r requirements.txt

```

> Note: If dependency conflicts occur during installation, try upgrading pip or using a virtual environment for isolation:
> ```bash
> pip install --upgrade pip
> python -m venv mfc-env
> # Activate virtual environment on Windows
> mfc-env\Scripts\activate
> # Activate virtual environment on Linux/Mac
> source mfc-env/bin/activate
> pip install -r requirements.txt
> 
> ```
> 
> 

## 2. Model Weight Download

Download the Qwen2.5-1.5B model from Hugging Face and place it in the specified directory:

1. Visit the Hugging Face model repository: [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
2. Download all complete model files (including config.json, model-00001-of-00002.safetensors, etc.)
3. Create the path `CLI-LLM/LLM/qwen_weight` in the project root directory (if it doesn't exist)
4. Copy all downloaded model files to the `qwen_weight` directory. The final directory structure should be:
```
CLI-LLM/
└── LLM/
    └── qwen_weight/
        ├── config.json
        ├── model-00001-of-00002.safetensors
        ├── model-00002-of-00002.safetensors
        ├── tokenizer_config.json
        └── tokenizer.model

```



> Optional: Auto-download using Hugging Face `transformers` (install `huggingface-hub` first):
> ```bash
> pip install huggingface-hub
> huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir MFC-LLM/LLM/qwen_weight --local-dir-use-symlinks False
> 
> ```
> 
> 

## 3. Dataset Preparation

Download the **hzmmmm/PHM2012_LLM** dataset from Hugging Face and organize it within the project:

1. Visit the dataset repository: [hzmmmm/PHM2012_LLM](https://huggingface.co/datasets/hzmmmm/PHM2012_LLM)
2. Create the directory `/home/hzm/my_github_project/CLI-LLM/DATA/CRWU`.
3. Copy all downloaded model files to the `data` directory. The final directory structure should be:
```
CLI-LLM/
└── data/
    ├── CRWU_data.hdf5
    └── CRWU_data.sqlite

```



## 4. Path Configuration Modification

Modify all path-related configurations in the project according to your actual deployment environment:

* **Model weight path:** Ensure the model loading path in the code points to `CLI-LLM/LLM/qwen_weight`
* **Dataset path:** Update the reading path for pre-training/fine-tuning datasets to point to `CLI-LLM/data` (matches Step 3)
* **Output path:** Specify the save path for model checkpoints and log files
* **Other paths:** Adjust paths for configuration files, cache files, etc. (locate via code search)

> Tip: Use global search for keywords like `path`, `dir`, or `load_from` to quickly find path configurations that need modification.


---

## 5. Run Experiment

Once the environment is configured and the paths are updated, execute the experiment script using the following command:

```bash
python run_length_experiment.py

```

> **Note:**
> * **Environment:** Ensure your virtual environment (from Step 1) is currently active before running the command.
> * **Logs:** Monitor the console for immediate output. Training logs and checkpoints will be saved to the `Output path` you configured in Step 4.
> * **GPU Usage:** If you have multiple GPUs, you may need to specify visibility (e.g., `CUDA_VISIBLE_DEVICES=0 python run_length_experiment.py`).
> 
> 
---
