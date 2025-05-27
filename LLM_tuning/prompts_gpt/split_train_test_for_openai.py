import json
import random

with open("participant_prompts_openai_format.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

random.seed(42)
random.shuffle(lines)

split_idx = int(0.9 * len(lines))
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

with open("train_openai.jsonl", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("val_openai.jsonl", "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print(f"Train samples: {len(train_lines)}, Validation samples: {len(val_lines)}")
