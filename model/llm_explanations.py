import openai
import pandas as pd
import json

openai.api_key = "Your API key"

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
    response = openai.ChatCompletion.create(
        model="ft:gpt-4o-mini-2024-07-18:saarland-university-computational-linguistics:health-insurance-churn:BbncVy4E",
        messages=messages
    )
    llm_text = response['choices'][0]['message']['content']
    return shap_text, llm_text

if __name__ == "__main__":
    # Load SHAP dict from JSON file
    with open("shap_dict_all.json", "r", encoding="utf-8") as file:
        shap_dict_all = json.load(file)

    results = []
    for class_label, examples in shap_dict_all.items():
        for example_idx_str, shap_example_dict in examples.items():
            example_idx = int(example_idx_str)
            shap_text, llm_text = get_llm_explanation(shap_example_dict)
            results.append({
                "class_label": class_label,
                "example_index": example_idx,
                "shap_text": shap_text,
                "llm_text": llm_text,
                "shap_dict_str": str(shap_example_dict)
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv("shap_llm_explanations.csv", index=False)
