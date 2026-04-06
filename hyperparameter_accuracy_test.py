# Best alpha: 0.215, binarize: 0.0, accuracy: 0.8546924657173459, std dev: 0.01540543724161127
import pandas as pd
import re
import numpy as np
import string
import os
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def parse_likert(x):
    if pd.isna(x):
        return np.nan
    match = re.match(r"^\s*([1-5])", str(x))
    return int(match.group(1)) if match else np.nan

def contains_word(text, word):
    if pd.isna(text):
        return 0
    words = str(text).lower().translate(str.maketrans("", "", string.punctuation)).split()
    return 1 if word in words else 0

# Load data
df = pd.read_csv("ml_challenge_dataset.csv")

cols = df.columns
emotion = cols[2]
feel_text_col = cols[3]
raw_sombre = cols[4]
raw_content = cols[5]
raw_calm = cols[6]
raw_uneasy = cols[7]
raw_prominent_colors = cols[8]
raw_objects_caught_eye = cols[9]
raw_room_to_put_painting = cols[11]
raw_person_to_view_with = cols[12]
raw_season = cols[13]
food_text_col = cols[14]
sountrack_text_col = cols[15]

df["sombre"] = df[raw_sombre].apply(parse_likert)
df["content"] = df[raw_content].apply(parse_likert)
df["calm"] = df[raw_calm].apply(parse_likert)
df["uneasy"] = df[raw_uneasy].apply(parse_likert)


FEEL_KEYWORDS = ["time", "sad", "clocks", "melting",
            "passing", "sky", "night", "quiet", "wonder",
            "happy", "warm", "nature", "joyful", "bright",
            "relaxed", "awe", "content"]
feel_keyword_cols = []
for kw in FEEL_KEYWORDS:
    feel_keyword_cols.append(f"feel_contains_{kw}")
    df[f"feel_contains_{kw}"] = df[feel_text_col].apply(
        lambda x, k=kw: contains_word(x, k)
    )

sountrack_keyword_cols = []

SOUNDTRACK_KEYWORDS = ["sad", "time", "quiet", "low", "violin",
                       "wind", "eerie", "sombre", "ticking", "classical",
                       "calming", "peaceful", "night", "emotional", "jazz",
                       "strings", "happy", "upbeat", "light", "birds", "gentle",
                       "chirping", "nature", "flute", "bright"]
for kw in SOUNDTRACK_KEYWORDS:
    sountrack_keyword_cols.append(f"sountrack_contains_{kw}")
    df[f"sountrack_contains_{kw}"] = df[sountrack_text_col].apply(
        lambda x, k=kw: contains_word(x, k)
    )


food_keyword_cols = []

FOOD_KEYWORDS = ["bread", "cheese", "pizza", "cake", "soup",
                 "blueberry", "salad", "fresh", "green", "matcha",
                 "chocolate", "fruit", "tea", "strawberry"]
for kw in FOOD_KEYWORDS:
    food_keyword_cols.append(f"food_contains_{kw}")
    df[f"food_contains_{kw}"] = df[food_text_col].apply(
        lambda x, k=kw: contains_word(x, k)
    )


ROOM = ["Bathroom", "Dining room", "Office", "Bedroom", "Living room"]
for room in ROOM:
    df[room] = df[raw_room_to_put_painting].apply(lambda x, c=room: 1 if pd.notna(x) and c in str(x) else 0) # creating binary indicator for each room

PERSON_TO_VIEW_WITH = ["By yourself", "Coworkers/Classmates", "Friends", "Family members", "Strangers"]
for person in PERSON_TO_VIEW_WITH:
    df[person] = df[raw_person_to_view_with].apply(lambda x, c=person: 1 if pd.notna(x) and c in str(x) else 0)

SEASONS = ["Spring", "Summer", "Fall", "Winter"]
for season in SEASONS:
    df[season] = df[raw_season].apply(lambda x, c=season: 1 if pd.notna(x) and c in str(x) else 0)


numeric_cols = [emotion, raw_prominent_colors, raw_objects_caught_eye, "sombre", "content", "calm", "uneasy"]

binary_cols = ROOM + PERSON_TO_VIEW_WITH + SEASONS + feel_keyword_cols + sountrack_keyword_cols + food_keyword_cols

feature_cols = numeric_cols + binary_cols

# use the median of each column to fill in NaN entries
# median is more stable than the mean, particularly when there are outliers
# median is in the domain of likert score or binary score while mean is most probably not (e.g: 3 vs 2.7)

medians = df[numeric_cols].median() # returns a series of medians across all columns of numeric_cols
df[numeric_cols] = df[numeric_cols].fillna(medians) # for each column, pandas fills the NaN vals of that column with the median of the column
# no need to do the same for binary_cols because if an entry is NaN for a binary column, the value stored is 0

# Normalize
X = df[feature_cols].values # data matrix
mean = X.mean(axis=0) # axis=0 => row, axis=1 => column
std = X.std(axis=0)
X_scaled = (X - mean) / std # abuse of syntax notation. pandas makes the syntax look too simple for what the instruction actually does

y = df["Painting"].values # store the vector of targets

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.model_selection import cross_val_score

gb_scores = cross_val_score(
    GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.15, random_state=42),
    X_scaled, y, cv=cv, scoring='accuracy'
)
lr_scores = cross_val_score(
    LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    X_scaled, y, cv=cv, scoring='accuracy'
)
gnb_scores = cross_val_score(
    GaussianNB(var_smoothing=0.325),
    X_scaled, y, cv=cv, scoring='accuracy'
)
bnb_scores = cross_val_score(
    BernoulliNB(alpha=0.215, binarize=0.0),
    X, y, cv=cv, scoring='accuracy'
)

print(f"GBM alone:          {gb_scores.mean():.4f} +/- {gb_scores.std():.4f}")
print(f"LR alone:           {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")
print(f"Gaussian NB alone:  {gnb_scores.mean():.4f} +/- {gnb_scores.std():.4f}")
print(f"Bernoulli NB alone: {bnb_scores.mean():.4f} +/- {bnb_scores.std():.4f}")

from sklearn.model_selection import StratifiedKFold

fold_accuracies_gnb = []
fold_accuracies_bnb = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X_scaled, y):
    # Get test predictions for each model on this fold
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.15, random_state=42)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    gnb = GaussianNB(var_smoothing=0.325)
    bnb = BernoulliNB(alpha=0.215, binarize=0.0)

    # Fit on train fold
    gb.fit(X_scaled[train_idx], y[train_idx])
    lr.fit(X_scaled[train_idx], y[train_idx])
    gnb.fit(X_scaled[train_idx], y[train_idx])
    bnb.fit(X[train_idx], y[train_idx])

    # Predict on test fold
    gb_p = gb.predict(X_scaled[test_idx])
    lr_p = lr.predict(X_scaled[test_idx])
    gnb_p = gnb.predict(X_scaled[test_idx])
    bnb_p = bnb.predict(X[test_idx])

    # Majority vote for GNB ensemble
    fold_preds_gnb = []
    for g, l, n in zip(gb_p, lr_p, gnb_p):
        votes = [g, l, n]
        unique, counts = np.unique(votes, return_counts=True)
        fold_preds_gnb.append(unique[np.argmax(counts)])
    fold_accuracies_gnb.append((np.array(fold_preds_gnb) == y[test_idx]).mean())

    # Majority vote for BNB ensemble
    fold_preds_bnb = []
    for g, l, n in zip(gb_p, lr_p, bnb_p):
        votes = [g, l, n]
        unique, counts = np.unique(votes, return_counts=True)
        fold_preds_bnb.append(unique[np.argmax(counts)])
    fold_accuracies_bnb.append((np.array(fold_preds_bnb) == y[test_idx]).mean())

print(f"Ensemble with Gaussian NB: {np.mean(fold_accuracies_gnb):.4f} +/- {np.std(fold_accuracies_gnb):.4f}")
print(f"Ensemble with Bernoulli NB: {np.mean(fold_accuracies_bnb):.4f} +/- {np.std(fold_accuracies_bnb):.4f}")
