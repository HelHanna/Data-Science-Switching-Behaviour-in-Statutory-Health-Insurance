import sys
import json
import openai
import pandas as pd
from openai import OpenAI

# Get model name from command-line argument
model_name = sys.argv[1] if len(sys.argv) > 1 else "unknown_model"

# create model aliases for file names
model_aliases = {
    "ft:gpt-4o-mini-2024-07-18:saarland-university-computational-linguistics:health-insurance-churn:BbncVy4E": "ft_gpt4o",
    "gpt-4o-mini-2024-07-18": "gpt4o_base",
}

model_alias = model_aliases.get(model_name, model_name)

api_key = "Your API key"
client = OpenAI(api_key=api_key)

def shap_to_text(shap_dict, top_n=10):
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    parts = []
    for feature, value in top_features:
        readable_feature = feature.replace('_', ' ')
        direction = "positive" if value > 0 else "negative"
        parts.append(f'"{readable_feature}" with {value:.2f} ({direction})')
    return "The most important features are: " + ", ".join(parts) + "."

def build_prompt(shap_text):
    system_message = (
        "You are a helpful assistant that explains how different features influence "
        "a model's prediction based on SHAP values."
    )
    user_message = (
        f"Here are the most important features for an example:\n\n{shap_text}\n\n"
        "Please explain in simple terms how these features affect the prediction."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

def get_llm_explanation(shap_example_dict):
    shap_text = shap_to_text(shap_example_dict)
    messages = build_prompt(shap_text)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    llm_text = response.choices[0].message.content.strip()
    return shap_text, llm_text


if __name__ == "__main__":
    # Load SHAP dict from JSON file 
    with open("shap_dict_all.json", "r", encoding="utf-8") as file:
        shap_dict_all = json.load(file)

    results = []
    max_prompts = 100 # define max prompts
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
            count +=1

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"shap_llm_explanations_{model_alias}.csv", index=False)

