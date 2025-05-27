from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load tokenizer (same one used for training)
tokenizer = AutoTokenizer.from_pretrained("/LLM_tuning")

# Load base model (exact model used during LoRA training)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=True  # if required
)

# Load the LoRA adapter into the base model
model = PeftModel.from_pretrained(base_model, "/LLM_tuning")

# Merge LoRA weights into the base model
merged_model = model.merge_and_unload()

# Save the full merged model
merged_model.save_pretrained("/LLM_tuning/merged_llama3")
tokenizer.save_pretrained("/LLM_tuning/merged_llama3")
