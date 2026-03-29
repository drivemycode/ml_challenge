from collections import Counter
import pandas as pd
import string


"""
Stop words for 'Describe this painting makes you feel'
stop = {"the", "and", "me", "feel", "makes", "it", 
        "of", "a", "is", "in", "i", "this", "to", 
        "that", "how", "like", "as", "but", "to", 
        "at", "with", "about", "feels", "sense", 
        "also", "my", "are", "also", "very", "bit", 
        "painting", "on", "not", "or", "make", "if", 
        "for", "has", "feeling", "everything", "an", 
        "which", "so", "gives", "be", "what", "think", 
        "out", "by", "am", "there", "can", "really", 
        "its", "away"}

Stop words for 'Describe soundtrack'
stop = {"the", "a", "and", "would", "with", "of", "be", "soundtrack", 
        "it", "in", "music", "that", "to", "is", "like", "sound", "but", 
        "very", "sounds", "or", "song", "an", "piano", "melody", "something", 
        "by", "i", "calm", "feel", "background", "notes", "piece", "this",
        "have", "at", "long", "as", "some", "slow", "soft", "maybe", "you", "for",
        "not", "slightly", "for", "painting", "instruments"}

"""

df = pd.read_csv("ml_challenge_dataset.csv")
cols = df.columns
text_col = cols[14]

for painting in df["Painting"].unique():
    subset = df[df["Painting"] == painting][text_col].dropna()
    text = " ".join(subset).lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    words = text.split()
    stop = {"the", "a", "and", "it", "of", "that", "would", "like", 
            "with", "be", "is", "you", "ice", "cream", "sandwich", "this",
            "something", "in"}
    words = [w for w in words if w not in stop]
    top = Counter(words).most_common(10)
    print(f"{painting}:")
    for word, count in top:
        print(f"    {word}: {count}")