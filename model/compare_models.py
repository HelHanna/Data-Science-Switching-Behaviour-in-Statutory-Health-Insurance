import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Load both enhanced CSVs
df_base = pd.read_csv("shap_llm_explanations_enhanced_gpt4o_base.csv")
df_ft = pd.read_csv("shap_llm_explanations_enhanced_ft_gpt4o.csv")

# Label the source/model
df_base["model"] = "gpt4o_base"
df_ft["model"] = "ft_gpt4o"

# Combine into one DataFrame
df_all = pd.concat([df_base, df_ft], ignore_index=True)

# Compare cosine similarity
cosine_ttest = ttest_ind(
    df_base["cosine_similarity"],
    df_ft["cosine_similarity"],
    equal_var=False
)

# Compare keyword accuracy
accuracy_ttest = ttest_ind(
    df_base["keyword_accuracy"],
    df_ft["keyword_accuracy"],
    equal_var=False
)

print("Cosine Similarity t-test:", cosine_ttest)
print("Keyword Accuracy t-test:", accuracy_ttest)
print("\nAverage Cosine Similarity:")
print(df_all.groupby("model")["cosine_similarity"].mean())
print("\nAverage Keyword Accuracy:")
print(df_all.groupby("model")["keyword_accuracy"].mean())

# Optional: Visual comparison
sns.boxplot(data=df_all, x="model", y="cosine_similarity")
plt.title("Cosine Similarity by Model")
plt.show()

sns.boxplot(data=df_all, x="model", y="keyword_accuracy")
plt.title("Keyword Accuracy by Model")
plt.show()
