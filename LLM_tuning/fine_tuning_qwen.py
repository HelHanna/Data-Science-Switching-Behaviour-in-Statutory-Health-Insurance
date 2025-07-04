import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

# === Parameters ===
HF_TOKEN= "Your Token"
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
DATA_PATH = "../preprocessing/participant_prompts.jsonl"
OUTPUT_DIR = "OUTPUT_DIR"
LOG_DIR = "LOG_DIR"
MAX_LENGTH = 2048

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

# === Load quantized model (QLoRA) ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
model.config.use_cache = False

# === Apply LoRA ===
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# === Load dataset ===
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# === Qwen ChatML-style prompt formatting ===
def format_example(example):
    prompt = example["prompt"].strip()
    completion = example["completion"].strip()
    full_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )
    return {"text": full_text}

train_dataset = dataset["train"].map(format_example)
eval_dataset = dataset["test"].map(format_example)

# === Tokenize ===
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

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    logging_dir=LOG_DIR,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# === Trainer ===
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# === Train ===
trainer.train()

# === Save ===
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")
