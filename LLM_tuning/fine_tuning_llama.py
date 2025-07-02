from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
import torch



# set your parameters: which model to use, your access keys, directories etc.
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN"
DATA_PATH = "../preprocessing/participant_prompts.jsonl"
OUTPUT_DIR = "OUTPUT_DIR"
LOG_DIR = "LOG_DIR"
MAX_LENGTH = 2048

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
)

# use lora
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# load dataset and split into training and test set
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

def format_example(example):
    prompt = example["prompt"].strip()
    completion = example["completion"].strip()
    full_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>\n{completion}<|eot_id|>"
    return {"text": full_text}

train_dataset = dataset["train"].map(format_example)
eval_dataset = dataset["test"].map(format_example)

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

# define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=20,  # set a number of epochs
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
    load_best_model_at_end=True,  # load best model after early stopping
    metric_for_best_model="eval_loss",  # define the metric
    greater_is_better=False,
)

# this class adds test prompts between the epochs to check for the quality. 
# adjust according to your needs or delete if its not needed
class ExampleGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def on_epoch_end(self, args, state, control, **kwargs):
        test_prompts = [
            "Eine Person hat folgende Eigenschaften und hat folgende Angaben gemacht. ..."
            # Your test prompt here
        ]
        self.model.eval()
        for prompt in test_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=100)
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"\nPrompt:\n{prompt}\n\nResponse:\n{decoded}\n")


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2), ExampleGenerationCallback(tokenizer, model)],
)


# start training
trainer.train()
print("Training abgeschlossen")

# save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")
