import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import process
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Preprocessing:
    model_name = "oliverguhr/german-sentiment-bert"
    with open("known_kk.json", encoding="utf-8") as f:
        known_kassen = json.load(f)

    with open("feature_dict.json", encoding="utf-8") as f:
        feature_dict = json.load(f)

    def __init__(self, file, target):
        df = pd.read_excel(file, sheet_name="Result")  
        print("Loaded columns:", df.columns.tolist())
        self.train = df.drop([target], axis=1)  

        self.target = df[target]

    def drop_columns_with_nan(self, threshold):
        dropped_cols = []
        for col in self.train.columns:
            nans = self.train[col].isnull().sum()
            length_of_col = len(self.train[col])
            if 100*(nans/length_of_col) > threshold:
                dropped_cols.append(col)
        self.train=self.train.drop(columns=dropped_cols, axis=1)
        return self.train

    def lowercase_strip(self, col):
        self.train[col]=self.train[col].str.lower().str.strip()
        return self.train
    def replace_nan(self, col, replacement):
        if col not in self.train.columns:
            raise KeyError(f"Column '{col}' not found in training data")
        self.train[col] = self.train[col].fillna(replacement)


    def standardize_categories(self, col, score_cutoff):
        def match_kasse(text):
            if pd.isnull(text):
                return "keine angabe"
            text = text.lower()
            if "techniker" in text:
                return "tk"
            if "dak gesundheit" in text:
                return "dak"
            if "a0k" in text:
                return "aok"
            result = process.extractOne(text, Preprocessing.known_kassen, score_cutoff=score_cutoff)
            return result[0] if result else text

        self.train[col] = self.train[col].apply(match_kasse)
        return self.train[col]

    def drop_irrelevant_cols(self, cols_to_drop):
        self.train = self.train.drop(columns=cols_to_drop,axis=1,errors="ignore")
        return self.train

    def set_category(self, col):
        if self.train[col].dtype == 'object':
            self.train = self.train[col].astype('category')
            return self.train
        if self.train[col].dtype == 'int64':
            self.train = self.train[col].astype('int')
            return self.train

    def sentiment_analysis(self, col):
        tokenizer = AutoTokenizer.from_pretrained(Preprocessing.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(Preprocessing.model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        texts = self.train[col].fillna('').astype(str).tolist()

        results = sentiment_pipeline(texts, truncation=True)

        self.train[col] = [r['label'].lower() for r in results]

        return self.train[col]

    def rename_features(self):
        self.train=self.train.rename(columns=Preprocessing.feature_dict)
        return self.train

    def fill_multiple_columns(self, col_value_dict):
        for col, val in col_value_dict.items():
            self.replace_nan(col, val)
    def get_features_and_target(self):
        X = self.train
        y = self.target
        return X, y













