"""
LoRA Fine-Tuning Script for OpenLLaMA 3B
----------------------------------------
This script fine-tunes a small LLaMA-like model using LoRA (Low-Rank Adaptation).
It is designed for an 8GB GPU (like RTX 4060) and is Windows-friendly.

Key steps:
1. Load and preprocess your dataset
2. Load a small base model and tokenizer
3. Apply LoRA for parameter-efficient fine-tuning
4. Train using Hugging Face Trainer
5. Save the LoRA adapter for later use in your RAG pipeline
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ----------------------------
# 1. Configuration
# ----------------------------
DATA_PATH = "data/processed/lora_dataset.json"        # Our LoRA dataset
BASE_MODEL = "openlm-research/open_llama_3b"          # Small model suitable for 8GB GPU
OUTPUT_DIR = "lora_model"                             # Where to save the trained adapter

# Training hyperparameters (tuned for 8GB GPU)
BATCH_SIZE = 1                    # Small batch to fit into VRAM
GRAD_ACCUM_STEPS = 8              # Accumulate gradients to simulate larger batch
EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_LENGTH = 256                  # Max token length for each example

# ----------------------------
# 2. Load the Dataset
# ----------------------------
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH)
print("✅ Dataset loaded:", dataset)

# ----------------------------
# 3. Load Tokenizer and Base Model
# ----------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA models have no pad token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map=None,                    # Automatically use GPU if available
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda")
print("✅ Model loaded on device:", model.device)

# ----------------------------
# 4. Configure LoRA
# ----------------------------
lora_config = LoraConfig(
    r=8,                                 # Low-rank dimension
    lora_alpha=32,                        # Scaling factor
    target_modules=["q_proj", "v_proj"],   # Typical for LLaMA-family models
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
#model.gradient_checkpointing_enable()
print("✅ LoRA applied to the model.")

# ----------------------------
# 5. Tokenization Function
# ----------------------------
def format_and_tokenize(example):
    """
    Converts each dataset row into an instruction-based prompt:
    Instruction + Input + Expected Response
    """
    prompt = (
        f"### Instruction:\n{example['instruction']}\n"
        f"### Input:\n{example['input']}\n"
        f"### Response:\n{example['output']}"
    )
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH)

print("Tokenizing dataset...")
tokenized_dataset = dataset["train"].map(
    format_and_tokenize,
    remove_columns=dataset["train"].column_names
)
print("✅ Tokenized dataset ready.")

# ----------------------------
# 6. Data Collator for Causal LM
# ----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ----------------------------
# 7. Define Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    save_steps=200,
    fp16=torch.cuda.is_available(),       # Mixed precision if GPU supports it
    save_total_limit=2,
    report_to="none"                      # Disable W&B or other loggers
)

# ----------------------------
# 8. Start Training
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("🚀 Starting LoRA fine-tuning...")
trainer.train()

# ----------------------------
# 9. Save the LoRA Adapter
# ----------------------------
model.save_pretrained(OUTPUT_DIR)
print(f"🎉 LoRA fine-tuning complete. Model saved at: {OUTPUT_DIR}")
