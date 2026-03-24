import os
import re
import json
import numpy as np
import pandas as pd

from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier

import joblib


# ============================================================
# 1. CONFIG
# ============================================================

TRAIN_CSV = "ml_challenge_dataset.csv"
TARGET_COL = "Painting"
ID_COL = "unique_id"

RANDOM_STATE = 42
N_SPLITS = 5
OUT_DIR = "artifacts"


# ============================================================
# 2. ACTUAL DATASET COLUMNS
# ============================================================

NUMERIC_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
]

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

CATEGORICAL_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]

TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "How much (in Canadian dollars) would you be willing to pay for this painting?",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]

PRICE_TEXT_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"


# ============================================================
# 3. HELPERS
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x


def parse_likert_to_number(x):
    """
    Example inputs:
      '1 - Strongly disagree' -> 1
      '3 - Neutral/Unsure' -> 3
      '5 - Strongly agree' -> 5
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.match(r"^\s*([1-5])", s)
    if m:
        return float(m.group(1))
    return np.nan


def parse_price_to_float(x):
    """
    Extract a numeric dollar amount from free-text.
    Examples:
      '$50' -> 50
      '100 dollars' -> 100
      'around 75.5' -> 75.5
      'free' -> nan
    """
    if pd.isna(x):
        return np.nan

    s = str(x).lower().replace(",", "").strip()

    # common non-numeric answers
    if s in {"", "free", "n/a", "na", "none", "nothing", "idk", "don't know", "dont know"}:
        return np.nan

    matches = re.findall(r"\d+\.?\d*", s)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return np.nan

    return np.nan


# ============================================================
# 4. CUSTOM TRANSFORMERS
# ============================================================

class LikertTransformer(BaseEstimator, TransformerMixin):
    """
    Convert Likert string columns to numeric columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col in df.columns:
            df[col] = df[col].apply(parse_likert_to_number)
        return df.values.astype(float)


class ColumnTextCombiner(BaseEstimator, TransformerMixin):
    """
    Combine multiple text columns into a single string per row.
    Adds column names as light prefixes.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.columns)
        combined = []

        for _, row in df.iterrows():
            parts = []
            for col in self.columns:
                val = clean_text(row[col])
                if val:
                    parts.append(f"{col} {val}")
            combined.append(" ".join(parts))

        return np.array(combined)


class PriceFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Convert the price text field into:
      [parsed_price, missing_flag]
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = pd.Series(X.squeeze())
        vals = s.apply(parse_price_to_float)

        if vals.notna().any():
            fill_value = vals.median()
        else:
            fill_value = 0.0

        parsed = vals.fillna(fill_value).astype(float)
        missing = vals.isna().astype(float)

        out = np.column_stack([parsed.values, missing.values])
        return out


class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Convert sparse matrix to dense when needed.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            return X.toarray()
        return X


# ============================================================
# 5. PREPROCESSOR
# ============================================================

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    likert_pipeline = Pipeline([
        ("likert_to_num", LikertTransformer()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    text_pipeline = Pipeline([
        ("combine_text", ColumnTextCombiner(TEXT_COLS)),
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            sublinear_tf=True,
        )),
    ])

    price_numeric_pipeline = Pipeline([
        ("price_extract", PriceFeatureExtractor()),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("likert", likert_pipeline, LIKERT_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ("text", text_pipeline, TEXT_COLS),
            ("price_num", price_numeric_pipeline, [PRICE_TEXT_COL]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessor


# ============================================================
# 6. MODEL ZOO
# ============================================================

def get_models():
    models = {
        "logreg_multinomial": LogisticRegression(
            max_iter=4000,
            solver="lbfgs",
            C=1.0,
            random_state=RANDOM_STATE,
        ),

        "linear_svc": LinearSVC(
            C=1.0,
            random_state=RANDOM_STATE,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),

        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=80,
            random_state=RANDOM_STATE,
        ),
    }
    return models


# ============================================================
# 7. PIPELINE BUILDER
# ============================================================

def build_pipeline_for_model(model_name, model):
    preprocessor = build_preprocessor()

    dense_required = {
        "gradient_boosting",
        "mlp",
    }

    if model_name in dense_required:
        return Pipeline([
            ("preprocess", preprocessor),
            ("to_dense", DenseTransformer()),
            ("model", model),
        ])

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


# ============================================================
# 8. TEXT-ONLY BASELINE
# ============================================================

def evaluate_text_nb_baseline(X, y):
    combined_text = []
    text_df = X[TEXT_COLS].copy()

    for _, row in text_df.iterrows():
        parts = []
        for col in TEXT_COLS:
            val = clean_text(row[col])
            if val:
                parts.append(f"{col} {val}")
        combined_text.append(" ".join(parts))

    pipe = make_pipeline(
        TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            sublinear_tf=True,
        ),
        MultinomialNB(alpha=1.0),
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipe,
        combined_text,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(f"[text_nb_baseline] accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")


# ============================================================
# 9. EVALUATION
# ============================================================

def evaluate_models(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print("Dataset shape:", df.shape)
    print("\nClass counts:")
    print(y.value_counts())
    print("\nNumber of classes:", y.nunique())

    print("\n--- Text-only Naive Bayes baseline ---")
    evaluate_text_nb_baseline(X, y)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    models = get_models()

    print("\n--- Full-model comparison ---")
    for name, model in models.items():
        print(f"\nEvaluating {name} ...")
        pipe = build_pipeline_for_model(name, model)

        scores = cross_val_score(
            pipe,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{name}: {mean_score:.4f} +/- {std_score:.4f}")
        results.append((name, mean_score, std_score))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Summary ===")
    for name, mean_score, std_score in results:
        print(f"{name:22s} {mean_score:.4f} +/- {std_score:.4f}")

    return results


# ============================================================
# 10. TRAIN FINAL MODEL + SAVE
# ============================================================

def train_and_save_best(df, best_model_name, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = get_models()[best_model_name]
    pipe = build_pipeline_for_model(best_model_name, model)

    pipe.fit(X, y)

    model_path = os.path.join(out_dir, f"{best_model_name}_pipeline.joblib")
    joblib.dump(pipe, model_path)

    metadata = {
        "train_csv": TRAIN_CSV,
        "target_col": TARGET_COL,
        "id_col": ID_COL,
        "numeric_cols": NUMERIC_COLS,
        "likert_cols": LIKERT_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "text_cols": TEXT_COLS,
        "price_text_col": PRICE_TEXT_COL,
        "best_model_name": best_model_name,
    }

    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved fitted pipeline to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")


# ============================================================
# 11. MAIN
# ============================================================

def main():
    df = pd.read_csv(TRAIN_CSV)

    # Drop ID column from features if present
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Sanity check required columns
    expected_cols = set(
        [TARGET_COL]
        + NUMERIC_COLS
        + LIKERT_COLS
        + CATEGORICAL_COLS
        + TEXT_COLS
    )

    missing = expected_cols - set(df.columns)
    if missing:
        print("Missing columns detected:")
        for col in sorted(missing):
            print(" -", col)
        return

    results = evaluate_models(df)

    best_model_name = results[0][0]
    print(f"\nBest model based on CV: {best_model_name}")

    train_and_save_best(df, best_model_name)


if __name__ == "__main__":
    main()
