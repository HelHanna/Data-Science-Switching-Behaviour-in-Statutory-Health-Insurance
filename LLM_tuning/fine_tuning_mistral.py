from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
import torch
import os

# 1. Parameters
HF_TOKEN= "Your Huggingface Token"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = "../preprocessing/participant_prompts.jsonl"
OUTPUT_DIR = "OUTPUT_DIR"
LOG_DIR = "LOG_DIR"
MAX_LENGTH = 2048

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# 3. Quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 4. Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16
)

# 5. Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 6. Load & preprocess dataset
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ChatML-style format for Mistral-Instruct
def format_example(example):
    prompt = example["prompt"].strip()
    completion = example["completion"].strip()
    full_text = f"<s>[INST] {prompt} [/INST] {completion}</s>"
    return {"text": full_text}

train_dataset = dataset["train"].map(format_example)
eval_dataset = dataset["test"].map(format_example)

# Tokenization
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

train_dataset = train_dataset.map(tokenize, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    logging_dir=LOG_DIR,
    logging_steps=10,
    logging_first_step=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# 9. Trainer setup
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 10. Train
trainer.train()
print("Training abgeschlossen")

# 11. Save
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")

