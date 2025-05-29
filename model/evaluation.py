import re
import sys
import ast
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Get model name from command-line argument
model_name = sys.argv[1] if len(sys.argv) > 1 else "unknown_model"

# Define aliases for long model names 
model_aliases = {
    "ft:gpt-4o-mini-2024-07-18:saarland-university-computational-linguistics:health-insurance-churn:BbncVy4E": "ft_gpt4o",
    "gpt-4o-mini-2024-07-18": "gpt4o_base",
    "/nethome/hhelbig/Neural_Networks/LLM_tuning/merged_llama3": "llama_finetuned",
    "meta-llama/Llama-3.1-8B-Instruct": "llama_base",
}

# Sanitize or shorten model name if not found in alias dict
def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

model_alias = model_aliases.get(model_name, sanitize_filename(model_name))

    
synonym_cache = {}

def get_synonyms_german(word):
    if word in synonym_cache:
        return synonym_cache[word]
    
    url = f"https://www.openthesaurus.de/synonyme/search?q={word}&format=application/json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        synonyms = set()
        for entry in data.get('synsets', []):
            for term in entry.get('terms', []):
                synonyms.add(term['term'].lower())
        result = list(synonyms) or [word.lower()]
        synonym_cache[word] = result
        return result
    except requests.RequestException as e:
        print(f"API error for '{word}': {e}")
        synonym_cache[word] = [word.lower()]
        return [word.lower()]


def get_top_n_features(shap_values_dict, n=10):
    sorted_feats = sorted(shap_values_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feats = [feat.replace('_', ' ') for feat, val in sorted_feats[:n]]
    return top_feats

def add_synonyms_to_features(features):
    features_with_syns = {}
    for feat in features:
        syns = get_synonyms_german(feat)
        features_with_syns[feat] = syns
    return features_with_syns

def count_top_keywords_in_text(text, features_with_syns):
    text = text.lower()
    count = 0
    for feat, syns in features_with_syns.items():
        if any(syn in text for syn in syns):
            count += 1
    return count

def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    sim = cosine_similarity(vectors[0], vectors[1])[0][0]
    return sim

def process_row(row, top_n=10):
    shap_dict = ast.literal_eval(row['shap_dict_str'])
    top_feats = get_top_n_features(shap_dict, n=top_n)
    features_with_syns = add_synonyms_to_features(top_feats)
    matched_count = count_top_keywords_in_text(row['llm_text'], features_with_syns)
    accuracy = matched_count / len(top_feats) if top_feats else 0
    features_syns_str = str(features_with_syns)
    return pd.Series([matched_count, accuracy, features_syns_str])

if __name__ == "__main__":
    df = pd.read_csv(f"shap_llm_explanations_{model_alias}.csv")

    df = df.groupby("class_label").head(100).reset_index(drop=True)

    df[['matched_keyword_count', 'keyword_accuracy', 'features_with_synonyms']] = df.apply(process_row, axis=1)

    df['cosine_similarity'] = df.apply(lambda row: compute_cosine_similarity(row['shap_text'], row['llm_text']), axis=1)

    df.to_csv(f"shap_llm_explanations_enhanced_{model_alias}.csv", index=False)


