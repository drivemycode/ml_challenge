import pandas as pd
import numpy as np
import os
import re
import string


# Load parameters
params = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gb_model_params.npz"), allow_pickle=True)
features_arr = params["features"]
thresholds_arr = params["thresholds"]
left_arr = params["left"]
right_arr = params["right"]
values_arr = params["values"]
init_scores = params["init_scores"]
learning_rate = float(params["learning_rate"][0])
classes = params["classes"]
scaler_mean = params["scaler_mean"]
scaler_std = params["scaler_std"]
medians = params["medians"]

# Preprocess functions
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


def traverse_tree_vectorized(X, feature, threshold, left, right, value):
    nodes = np.zeros(X.shape[0], dtype=int) # array tracking which node each sample is currently at. All start at node 0 (root)
    is_leaf = (feature[nodes] == -2)
    while not is_leaf.all(): # only keep looping when there exists a sample that has not reached a leaf node
        feature_vals = X[np.arange(X.shape[0]), feature[np.clip(nodes, 0, len(feature) - 1)]] # For each sample, look up the value of the feature that the current node splits on
        go_left = feature_vals <= threshold[nodes]
        # Update: go left or right (only for non-leaf nodes)
        new_nodes = np.where(go_left, left[nodes], right[nodes])
        nodes = np.where(is_leaf, nodes, new_nodes) # only update samples that haven't reached a leaf yet
        is_leaf = (feature[nodes] == -2)

    return value[nodes]
        

def predict_all(filename):
    df = pd.read_csv(filename)
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

    df[numeric_cols] = df[numeric_cols].fillna(pd.Series(medians, index=numeric_cols)) # for each column, pandas fills the NaN vals of that column with the median of the column from training phase

    # Normalize 
    X = df[feature_cols].values # data matrix
    
    X_scaled = (X - scaler_mean) / scaler_std

    n_estimators = 200
    n_classes = 3
    scores = np.tile(init_scores, (X_scaled.shape[0], 1))

    for i in range(n_estimators):
        for j in range(n_classes):
            tree_idx = i * n_classes + j
            leaf_vals = traverse_tree_vectorized(
                X_scaled,
                features_arr[tree_idx],
                thresholds_arr[tree_idx],
                left_arr[tree_idx],
                right_arr[tree_idx],
                values_arr[tree_idx]
            )
            scores[:, j] += learning_rate * leaf_vals
    predictions = classes[np.argmax(scores, axis=1)]
    return predictions.tolist()

