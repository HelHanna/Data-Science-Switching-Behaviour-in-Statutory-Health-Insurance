import os
import sys
import json
import re
import argparse
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

# ==================== ARGUMENTE PARSEN ====================

parser = argparse.ArgumentParser()
parser.add_argument("model_name", type=str, help="Hugging Face model name or local path")
parser.add_argument("--use_lora", action="store_true", help="Whether to apply LoRA adapter")
args = parser.parse_args()

model_name = args.model_name
enable_lora = args.use_lora

# ==================== KONFIGURATION ====================

hf_token = os.getenv("HUGGINGFACE_TOKEN", None)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

lora_path = "logs/mistral_output_5"

model_aliases = {
    "mistralai/Mistral-7B-Instruct-v0.3": "mistral_base",
    lora_path: "mistral_finetuned_5",
}

#model_alias = model_aliases.get(model_name, re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name))
base_alias = model_aliases.get(model_name, re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name))
model_alias = base_alias.replace("_base", "_finetuned_5") if enable_lora else base_alias

# ==================== MODELL LADEN ====================

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_auth_token=hf_token,
    quantization_config=bnb_config,
)

if enable_lora:
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer_path = lora_path
else:
    model = base_model
    tokenizer_path = model_name

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    trust_remote_code=True
)

# ==================== FUNKTIONEN ====================

def shap_to_text(shap_dict, top_n=10):
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    parts = []
    for feature, value in top_features:
        readable_feature = feature.replace('_', ' ')
        direction = "positive" if value > 0 else "negative"
        parts.append(f'"{readable_feature}" with {value:.2f} ({direction})')
    return "The most important features are: " + ", ".join(parts) + "."

def build_prompt(shap_text, class_label):
    system_message = (
        "You are a helpful assistant that explains how different features influence "
        "a model's prediction based on SHAP values."
    )
    user_message = (
        f"The predicted class is: {class_label}\n\n"
        f"Here are the most important features for an example:\n\n{shap_text}\n\n"
        "Please explain in simple terms how these features affect the prediction. The explanation should be short and precise."
    )
    return system_message + "\n" + user_message

def get_llm_explanation(shap_example_dict, class_label):
    shap_text = shap_to_text(shap_example_dict)
    prompt = build_prompt(shap_text, class_label)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        temperature=0.0,
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    llm_text = full_output[len(prompt):].strip()

    return shap_text, llm_text

# ==================== MAIN ====================

if __name__ == "__main__":
    with open("shap_dict_predicted.json", "r", encoding="utf-8") as file:
        shap_dict_all = json.load(file)

    results = []
    max_examples = 100
    count = 0

    for example_idx_str, shap_example_dict in shap_dict_all.items():
        if count >= max_examples:
            break

        example_idx = int(example_idx_str)
        shap_values_only = shap_example_dict["shap_values"]
        predicted_class = shap_example_dict["predicted_class"]
        predicted_class_label = shap_example_dict.get("predicted_class_label", str(predicted_class))

        shap_text, llm_text = get_llm_explanation(shap_values_only, predicted_class_label)

        print(f"\n--- Example {example_idx} ---")
        print(f"Predicted Class: {predicted_class}")
        print(f"Predicted Class Label: {predicted_class_label}")
        print("SHAP Text:")
        print(shap_text)
        print("LLM Explanation:")
        print(llm_text)
        print("-" * 60)

        results.append({
            "example_index": example_idx,
            "predicted_class": predicted_class,
            "shap_text": shap_text,
            "llm_text": llm_text,
            "shap_dict_str": str(shap_values_only),
        })
        count += 1

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"shap_llm_explanations_100_less_token_{model_alias}.csv", index=False)
