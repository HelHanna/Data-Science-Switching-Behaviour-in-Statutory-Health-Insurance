import json
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from preprocessing import Preprocessing
import shap


def train_lightgbm(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    model = lgb.LGBMClassifier(objective='multiclass', num_class=len(set(y)), random_state=random_state)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
    )

    return model, X_val, y_val

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted')

def create_shap_dict(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_dict_all = {}

    num_classes = shap_values.shape[2]
    num_samples = shap_values.shape[0]
    feature_names = X.columns.tolist()

    for k in range(num_classes):
        shap_dict_all[k] = {}
        shap_df_k = pd.DataFrame(shap_values.values[:, :, k], columns=feature_names)

        for i in range(num_samples):
            shap_dict_all[k][i] = shap_df_k.iloc[i].to_dict()

    return shap_dict_all

if __name__ == "__main__":

    pp = Preprocessing("230807_Survey.xlsx", "Q18", "Result")
    pp.drop_columns_with_nan(50)

    nan_replacements = {
        "Q7.1.2": "keine angabe",
        "Q7.1.3": "keine angabe",
        "Q13": 99,
        "Q55.12": 99,
        "Q59": 99,
        "Q75": 99,
        "Q99": 99,
        "Q103": 99,
        "M1": 99
    }
    for col, val in nan_replacements.items():
        pp.replace_nan(col, val)

    for col in ["Q7.1.2", "Q7.1.3"]:
        pp.lowercase_strip(col)

    for col in ["Q7.1.1", "Q7.1.2", "Q7.1.3"]:
        pp.standardize_categories(col,70)

    pp.sentiment_analysis("Q44")

    irrelevant_cols = ['Q25', 'Q27', 'Q12', 'Q13', 'Participant', 'Weight', 'State',
                       'Begin', 'End', 'Duration', 'User Agent', 'pid', 'Source',
                       'Locale', 'Project ID', 'Phase']
    pp.drop_irrelevant_cols(irrelevant_cols)
    pp.set_category('Q7.1.1')
    pp.set_category('Q7.1.2')
    pp.set_category('Q7.1.3')
    pp.set_category('Q44')

    pp.rename_features()

    X, y = pp.get_features_and_target()
    y = y.astype(int)

    model, X_val, y_val = train_lightgbm(X, y)
    acc, f1 = evaluate_model(model, X_val, y_val)
    shap_dict_all = create_shap_dict(model, X_val)
    
    with open("shap_dict_all.json", "w", encoding="utf-8") as f:
        json.dump(shap_dict_all, f, ensure_ascii=False, indent=2)
