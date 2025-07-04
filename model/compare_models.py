import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
from itertools import combinations
from scipy.stats import ttest_ind, mannwhitneyu

#List of (filename, model_name) tuples

model_files = [
    ("shap_llm_explanations_enhanced_llama_base.csv", "llama_base"),
    ("shap_llm_explanations_enhanced_llama_finetuned_20.csv", "llama_ft_20"),
    ("shap_llm_explanations_enhanced_llama_finetuned_5.csv", "llama_ft_5"),
    ("shap_llm_explanations_enhanced_qwen_base.csv", "qwen_base"),
    ("shap_llm_explanations_enhanced_qwen_finetuned_20.csv", "qwen_ft_8"),
    ("shap_llm_explanations_enhanced_qwen_finetuned_5.csv", "qwen_ft_5"),
    ("shap_llm_explanations_enhanced_mistral_base.csv", "mistral_base"),
    ("shap_llm_explanations_enhanced_mistral_finetuned_20.csv", "mistral_ft_6"),
    ("shap_llm_explanations_enhanced_mistral_finetuned_5.csv", "mistral_ft_5"),
]


# Load and label all dataframes
df_list = []
for file, model in model_files:
    df = pd.read_csv(file)
    df["model"] = model
    df_list.append(df)

# Combine all into a single DataFrame
df_all = pd.concat(df_list, ignore_index=True)

# Summary statistics
print("\nAverage Cosine Similarity:")
print(df_all.groupby("model")["cosine_similarity"].mean())

print("\nAverage Keyword Accuracy:")
print(df_all.groupby("model")["keyword_accuracy"].mean())

df_all["token_size"] = df_all["model"].apply(lambda x: int(x.split("_")[-1]))
df_all["model_family"] = df_all["model"].apply(lambda x: x.split("_")[0])


results = []

for model in df_all["model_family"].unique():
    df_300 = df_all[(df_all["model_family"] == model) & (df_all["token_size"] == 300)]
    df_600 = df_all[(df_all["model_family"] == model) & (df_all["token_size"] == 600)]

    # Cosine similarity
    t_cos = ttest_ind(df_300["cosine_similarity"], df_600["cosine_similarity"], equal_var=False)
    # Accuracy
    t_acc = ttest_ind(df_300["keyword_accuracy"], df_600["keyword_accuracy"], equal_var=False)

    results.append({
        "model_family": model,
        "cosine_p": t_cos.pvalue,
        "accuracy_p": t_acc.pvalue,
        "cosine_mean_300": df_300["cosine_similarity"].mean(),
        "cosine_mean_600": df_600["cosine_similarity"].mean(),
        "acc_mean_300": df_300["keyword_accuracy"].mean(),
        "acc_mean_600": df_600["keyword_accuracy"].mean()
    })

# Display results
results_df = pd.DataFrame(results)
print(results_df.round(4))


# Violin plots
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_all, x="model", y="cosine_similarity", inner="box")
plt.title("Cosine Similarity Distribution by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df_all, x="model", y="keyword_accuracy", inner="box")
plt.title("Keyword Accuracy Distribution by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Swarm plot
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df_all, x="model", y="cosine_similarity", size=3)
plt.title("Cosine Similarity by Model (Swarm View)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summary for error bar plot
summary = df_all.groupby("model").agg({
    "cosine_similarity": ["mean", "sem"],
    "keyword_accuracy": ["mean", "sem"]
}).reset_index()
summary.columns = ["model", "cosine_mean", "cosine_sem", "acc_mean", "acc_sem"]

# Bar plot with error bars using matplotlib
x = np.arange(len(summary["model"]))

# Cosine Similarity Bar Plot
plt.figure(figsize=(10, 5))
plt.bar(x, summary["cosine_mean"], yerr=summary["cosine_sem"], capsize=5)
plt.xticks(x, summary["model"], rotation=45)
plt.ylabel("Cosine Similarity")
plt.title("Average Cosine Similarity (with SEM)")
plt.tight_layout()
plt.show()

colors = plt.cm.tab10(np.linspace(0, 1, len(x)))  # oder 'tab20', 'Set3', etc.

plt.figure(figsize=(10, 5))
plt.bar(x, summary["cosine_mean"], yerr=summary["cosine_sem"], capsize=5, color=colors)
plt.xticks(x, summary["model"], rotation=45)
plt.ylabel("Cosine Similarity")
plt.title("Average Cosine Similarity (with SEM)")
plt.tight_layout()
plt.show()

# Keyword Accuracy Bar Plot
plt.figure(figsize=(10, 5))
plt.bar(x, summary["acc_mean"], yerr=summary["acc_sem"], capsize=5, color=colors)
plt.xticks(x, summary["model"], rotation=45)
plt.ylabel("Keyword Accuracy")
plt.title("Average Keyword Accuracy (with SEM)")
plt.tight_layout()
plt.show()

# Optional: Pairwise p-value matrices (heatmap)
models = df_all["model"].unique()
cosine_p = pd.DataFrame(index=models, columns=models, dtype=float)
keyword_p = pd.DataFrame(index=models, columns=models, dtype=float)

for m1, m2 in combinations(models, 2):
    cs1 = df_all[df_all["model"] == m1]["cosine_similarity"]
    cs2 = df_all[df_all["model"] == m2]["cosine_similarity"]
    ka1 = df_all[df_all["model"] == m1]["keyword_accuracy"]
    ka2 = df_all[df_all["model"] == m2]["keyword_accuracy"]
    
    cosine_p.loc[m1, m2] = cosine_p.loc[m2, m1] = ttest_ind(cs1, cs2, equal_var=False).pvalue
    keyword_p.loc[m1, m2] = keyword_p.loc[m2, m1] = ttest_ind(ka1, ka2, equal_var=False).pvalue

# Fill diagonal with NaNs
np.fill_diagonal(cosine_p.values, np.nan)
np.fill_diagonal(keyword_p.values, np.nan)

def mask_non_significant(p_matrix, alpha=0.05):
    return p_matrix >= alpha

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_p, annot=True, cmap="coolwarm_r", fmt=".3f", 
            mask=mask_non_significant(cosine_p), cbar_kws={'label': 'p-value'})
plt.title("Significant Pairwise t-test p-values (Cosine Similarity)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(keyword_p, annot=True, cmap="coolwarm_r", fmt=".3f", 
            mask=mask_non_significant(keyword_p), cbar_kws={'label': 'p-value'})
plt.title("Significant Pairwise t-test p-values (Keyword Accuracy)")
plt.tight_layout()
plt.show()



