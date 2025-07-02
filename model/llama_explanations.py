import os
import re
import sys
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load command-line argument
model_name = sys.argv[1] if len(sys.argv) > 1 else "unknown_model"

# Define model aliases
model_aliases = {
    "/Data-Science-Switching-Behaviour-in-Statutory-Health-Insurance/LLM_tuning/merged_llama3": "llama_finetuned",
    "meta-llama/Llama-3.1-8B-Instruct": "llama_base",
}
model_alias = model_aliases.get(model_name, re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name))

# Get HF token (for nonlocal models)
hf_token = os.getenv("HUGGINGFACE_TOKEN", None)

# Load model: local directory or Hugging Face repo
if os.path.isdir(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16, use_auth_token=hf_token
    )

# Format SHAP values into natural language
def shap_to_text(shap_dict, top_n=10):
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    parts = []
    for feature, value in top_features:
        readable_feature = feature.replace('_', ' ')
        direction = "positive" if value > 0 else "negative"
        parts.append(f'"{readable_feature}" with {value:.2f} ({direction})')
    return "The most important features are: " + ", ".join(parts) + "."

# Build prompt as a single string
def build_prompt(shap_text):
    system_message = (
        "You are a helpful assistant that explains how different features influence "
        "a model's prediction based on SHAP values."
    )
    user_message = (
        f"Here are the most important features for an example:\n\n{shap_text}\n\n"
        "Please explain in simple terms how these features affect the prediction."
    )
    return system_message + "\n" + user_message

# Generate explanation
def get_llm_explanation(shap_example_dict):
    shap_text = shap_to_text(shap_example_dict)
    prompt = build_prompt(shap_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=600,
        do_sample=False,    # disable sampling for deterministic output
        temperature=0       # usually ignored when do_sample=False
    )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    llm_text = full_output[len(prompt):].strip()

    return shap_text, llm_text


if __name__ == "__main__":
    with open("shap_dict_all.json", "r", encoding="utf-8") as file:
        shap_dict_all = json.load(file)

    results = []
    max_prompts = 100 # Limit per class

    for class_label, examples in shap_dict_all.items():
        count = 0
        for example_idx_str, shap_example_dict in examples.items():
            if count >= max_prompts:
                break
            example_idx = int(example_idx_str)

            shap_text, llm_text = get_llm_explanation(shap_example_dict)

            print(f"\n--- Example {example_idx} (Class: {class_label}) ---")
            print("SHAP Text:")
            print(shap_text)
            print("LLM Explanation:")
            print(llm_text)
            print("-" * 60)

            results.append({
                "class_label": class_label,
                "example_index": example_idx,
                "shap_text": shap_text,
                "llm_text": llm_text,
                "shap_dict_str": str(shap_example_dict)
            })
            count += 1

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"shap_llm_explanations_{model_alias}.csv", index=False)
