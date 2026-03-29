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

Stop words for 'Describe a food'
stop = {"the", "a", "and", "it", "of", "that", "would", "like", 
            "with", "be", "is", "you", "ice", "cream", "sandwich", "this",
            "something", "in"}

"""

df = pd.read_csv("ml_challenge_dataset.csv")
cols = df.columns
stop1 = {"the", "and", "me", "feel", "makes", "it", 
        "of", "a", "is", "in", "i", "this", "to", 
        "that", "how", "like", "as", "but", "to", 
        "at", "with", "about", "feels", "sense", 
        "also", "my", "are", "also", "very", "bit", 
        "painting", "on", "not", "or", "make", "if", 
        "for", "has", "feeling", "everything", "an", 
        "which", "so", "gives", "be", "what", "think", 
        "out", "by", "am", "there", "can", "really", 
        "its", "away"}

stop2 = {"the", "a", "and", "would", "with", "of", "be", "soundtrack", 
        "it", "in", "music", "that", "to", "is", "like", "sound", "but", 
        "very", "sounds", "or", "song", "an", "piano", "melody", "something", 
        "by", "i", "calm", "feel", "background", "notes", "piece", "this",
        "have", "at", "long", "as", "some", "slow", "soft", "maybe", "you", "for",
        "not", "slightly", "for", "painting", "instruments"}

stop3 = {"the", "a", "and", "it", "of", "that", "would", "like", 
            "with", "be", "is", "you", "ice", "cream", "sandwich", "this",
            "something", "in"}

stop = stop1 | stop2 | stop3

for col_name, col in [("feelings", cols[3]), ("food", cols[14]), ("soundtrack", cols[15])]:
    starry = df[df["Painting"] == "The Starry Night"][col].dropna()
    water = df[df["Painting"] == "The Water Lily Pond"][col].dropna()

    starry_words = Counter(" ".join(starry).lower().translate(str.maketrans("","", string.punctuation)).split())
    water_words = Counter(" ".join(water).lower().translate(str.maketrans("", "", string.punctuation)).split())

    all_words = set(starry_words.keys()) | set(water_words.keys())

    diffs = []
    for w in all_words:
        if w in stop:
            continue
        s = starry_words.get(w, 0)
        wl = water_words.get(w, 0)
        diffs.append((w, s, wl, abs(s - wl)))
    diffs.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{col_name} - top discriminative words:")
    for word, s, wl, d in diffs[:10]:
        print(f"  {word:15s}  Starry: {s:4d}  Water Lily: {wl:4d}  diff: {d}")

