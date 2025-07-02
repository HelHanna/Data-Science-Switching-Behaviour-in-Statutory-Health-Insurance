import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

def load_and_format_openai_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = ""
            for message in obj["messages"]:
                prefix = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}[message["role"]]
                text += f"{prefix}\n{message['content'].strip()}\n"
            data.append({"text": text.strip()})
    return Dataset.from_list(data)

if __name__ == '__main__':
    model_name = "ybelkada/Mistral-7B-v0.1-bf16-sharded"
    train_dataset = load_and_format_openai_jsonl("train_openai.jsonl")
    val_dataset = load_and_format_openai_jsonl("val_openai.jsonl")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=10,
        learning_rate=2e-5,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=10,
        warmup_ratio=0.03,
        group_by_length=False,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        save_strategy="steps",
        evaluation_strategy="steps",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    trainer.train()

    #save model
    OUTPUT_DIR = "./results"
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")
