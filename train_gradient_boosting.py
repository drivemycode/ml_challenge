import pandas as pd
import re
import numpy as np
import string
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

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


"""Training with Gradient Boosting"""
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.15,
    random_state=42
    )
# gb_model.fit(X_scaled, y)
# Extracting the structure of 200 tree estimators
# all_features = []
# all_thresholds = []
# all_left = []
# all_right = []
# all_values = []

# for i in range(gb_model.n_estimators):
#     for j in range(gb_model.n_classes_):
#         tree = gb_model.estimators_[i][j].tree_
#         all_features.append(tree.feature)
#         all_thresholds.append(tree.threshold)
#         all_left.append(tree.children_left)
#         all_right.append(tree.children_right)
#         all_values.append(tree.value.flatten())


# max_nodes = 7 # every tree either has 5 or 7 nodes
# 600 because for each round, we create 3 decision trees one for each painting
# features_arr = np.zeros((600, max_nodes), dtype=int)
# thresholds_arr = np.zeros((600, max_nodes))
# left_arr = np.zeros((600, max_nodes), dtype=int)
# right_arr = np.zeros((600, max_nodes), dtype=int)
# values_arr = np.zeros((600, max_nodes))

# for i in range(600):
#     n = len(all_features[i])
#     features_arr[i, :n] = all_features[i]
#     thresholds_arr[i, :n] = all_thresholds[i]
#     left_arr[i, :n] = all_left[i]
#     right_arr[i, :n] = all_right[i]
#     values_arr[i, :n] = all_values[i]

# get the initial predictions 
# init_scores = gb_model.init_.class_prior_ # initial guess by the model
# np.savez("gb_model_params.npz",
#          features=features_arr,
#          thresholds=thresholds_arr,
#          left=left_arr,
#          right=right_arr,
#          values=values_arr,
#          init_scores=init_scores,
#          learning_rate=np.array([gb_model.learning_rate]),
#          classes=gb_model.classes_,
#          scaler_mean=mean,
#          scaler_std=std,
#          medians=medians.values
#         )

# Best HP: n_estimators=200, max_depth=2, learning_rate=0.15, random_state=42
# Best accuracy: 0.8973890752023598 +/- 0.0055242001390046125
# scores = cross_val_score(gb_model, X_scaled, y, cv=cv, scoring='accuracy')
# print(f"CV Accuracy: {scores.mean()} +/- {scores.std()}")

# Uncomment the section below to show ROC curve
# ================================================
cv_probs = cross_val_predict(gb_model, X_scaled, y, cv=cv, method='predict_proba')

classes = gb_model.fit(X_scaled, y).classes_
y_bin = label_binarize(y, classes=classes)

for i, label in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], cv_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-Rest)')
plt.legend(loc='lower right')
plt.show()
# ================================================

# Uncomment the section below to see the confusion matrix
# ================================================
# cv_preds = cross_val_predict(gb_model, X_scaled, y, cv=cv)
# labels = ["The Persistence of Memory", "The Starry Night", "The Water Lily Pond"]
# short_labels = ["Persistence\nof Memory", "Starry\nNight", "Water Lily\nPond"]
# cm = confusion_matrix(y, cv_preds, labels=labels)
# fig, ax = plt.subplots(figsize=(7, 6))
# im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
# ax.figure.colorbar(im, ax=ax)
# ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
#        xticklabels=short_labels, yticklabels=short_labels,
#        ylabel='Actual', xlabel='Predicted',
#        title='Confusion Matrix (5-Fold Cross-Validation)')
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], 'd'),
#                 ha='center', va='center',
#                 color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
# plt.tight_layout()
# plt.savefig('confusion_matrix.png', dpi=150)
# plt.show()
# ================================================










