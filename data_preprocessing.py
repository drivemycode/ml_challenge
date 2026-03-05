import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

"""
Formatting
"""
df = pd.read_csv('ml_challenge_dataset.csv')
df = df.dropna()
# df = df.drop(columns=["unique_id"])

painting_dict = {
    "The Persistence of Memory": 1,
    "The Starry Night": 2,
    "The Water Lily Pond": 3
}

# reorder Painting column to first column
col_painting_data = df.pop("Painting").map(painting_dict)
df.insert(0, "Painting", col_painting_data)

# convert 1-5 score cols to be purely numeric
cols_int_from_str = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy."
]
for col in cols_int_from_str:
    df[col] = df[col].str.extract(r'(\d+)')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# convert multi-valued cols to indicator cols
# one field (A,B,C) -> three fields A, B, C
cols_multi_to_indicators = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?"
]
for col in cols_multi_to_indicators:
    dummies = df[col].str.get_dummies(sep=',')
    df = pd.concat([df, dummies], axis=1)
df.drop(columns=cols_multi_to_indicators)

# extract numeric money from 'how much would you pay' column
col_money_name = "How much (in Canadian dollars) would you be willing to pay for this painting?"
df[col_money_name] = df[col_money_name].str.extract(r'([\d,.]+)')[0].str.replace(',', '')

# drop any rows with invalid dollars
df = df.dropna()

# these ids require manual interpretation ("all my life savings", "i wouldn't buy this" etc.) 
# or are too large
# for now, remove the rows associated with them.
s = pd.to_numeric(df[col_money_name], errors="coerce")
vague_mask = s.isna() & df[col_money_name].notna()
i32 = np.iinfo(np.int32)
int_max_range_mask = s.notna() & ((s < i32.min) | (s > i32.max))

bad_ids = df.loc[vague_mask | int_max_range_mask, ["Painting", "unique_id"]].apply(tuple, axis=1).tolist()
df = df.set_index(["Painting", "unique_id"]).drop(bad_ids).reset_index()

df[col_money_name] = pd.to_numeric(df[col_money_name])

"""
Normalisation
"""
def z_normalize(col: str):
    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

# normalise formatted "how much would you spend column"
z_normalize(col_money_name)

# drop the raw column
df.drop(columns=col_money_name)

"""
Sentence embedding
"""
def clean_text(s) -> str:
    if pd.isna(s):
        return ""
    return " ".join(str(s).split())

model = SentenceTransformer("all-MiniLM-L6-v2")
cols_descriptive = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting."
]

embeddings = []
for col in cols_descriptive:
    texts = df[col].map(clean_text).tolist()
    E = model.encode(texts, 
                     batch_size=64, 
                     convert_to_numpy=True, 
                     show_progress_bar=True, 
                     normalize_embeddings=True)
    embeddings.append(E)
df.drop(columns=cols_descriptive)

print(embeddings)
print(df)