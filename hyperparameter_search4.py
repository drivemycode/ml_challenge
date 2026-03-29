from joblib import Parallel, delayed
import pandas as pd
import re
import numpy as np
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

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

"""Preprocessing Phase"""
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

# FEEL_KEYWORDS = ["time", "sad", "clocks", "melting", 
#             "passing", "sky", "night", "quiet", "wonder",
#             "happy", "warm", "nature", "joyful", "bright"]

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
# SOUNDTRACK_KEYWORDS = ["sad", "time", "quiet", "low", "violin", 
#                        "wind", "eerie", "sombre", "ticking", "classical",
#                        "calming", "peaceful", "night", "emotional", "jazz",
#                        "strings", "happy", "upbeat", "light", "birds", "gentle",
#                        "chirping", "nature", "flute", "bright"]

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
# FOOD_KEYWORDS = ["bread", "cheese", "pizza", "cake", "soup", 
#                  "blueberry", "salad", "fresh", "green", "matcha"]

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

# df["dark_mood"] = df["sombre"] + df["uneasy"] - df["calm"] - df["content"]
# df["season_certainty"] = df["Spring"] + df["Summer"] + df["Fall"] + df["Winter"]

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

def evaluate_config(num, depth, rate):
    gb = GradientBoostingClassifier(
        n_estimators=num,
        max_depth=depth,
        learning_rate=rate, 
        random_state=42
    )
    scores = cross_val_score(gb, X_scaled, y, cv=cv, scoring='accuracy')
    return scores.mean(), num, depth, rate



n_estimators_options = [550, 600, 650]
max_depth_options = [2, 3, 4, 5]
learning_rate_options = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2]

configs = [(num, depth, rate) for num in n_estimators_options for depth in max_depth_options for rate in learning_rate_options]
results = Parallel(n_jobs=-1)(
    delayed(evaluate_config)(num, depth, rate)
    for num, depth, rate in configs
)

results.sort(key=lambda x: x[0], reverse=True)
best = results[0]
curr_best = 0.8956156831071234
print(f"Best accuracy: {best[0]}")
print(f"Best config: n_estimators={best[1]}, max_depth={best[2]}, learning_rate={best[3]}")
if best[0] > curr_best:
    print("Found a better config!")
else:
    print("Failed to find a better config :(")
