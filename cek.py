import pandas as pd
df = pd.read_csv("song_lyrics.csv", nrows=20000)
print(df.columns.tolist())
print(df["tag"].value_counts())

