import pandas as pd
total = pd.Series(dtype="int64")
try:
    for ch in pd.read_csv("song_lyrics.csv", usecols=["tag", "language"], chunksize=500_000):
        total = total.add(ch[ch["language"] == "en"]["tag"].value_counts(), fill_value=0)
except pd.errors.ParserError as e:
    print("Berhenti di tengah file karena baris rusak, pakai data yang sudah terbaca saja.")
print(total.sort_values(ascending=False).astype(int))