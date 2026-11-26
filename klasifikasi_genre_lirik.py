import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CSV_PATH = "song_lyrics.csv"
GENRES = ["pop", "rap", "rock", "country"]   
PER_GENRE = 3000                             
SEED = 42

# 1. Muat data per potongan (filenya besar), saring bahasa Inggris + genre terpilih
def sample_per_genre(data, n):
    parts = []
    for g in GENRES:
        sub = data[data["tag"] == g]
        parts.append(sub.sample(min(len(sub), n), random_state=SEED))
    return pd.concat(parts)

frames = []
try:
    for chunk in pd.read_csv(CSV_PATH, usecols=["tag", "lyrics", "language"], chunksize=200_000):
        chunk = chunk[(chunk["language"] == "en") & (chunk["tag"].isin(GENRES))]
        frames.append(sample_per_genre(chunk, 400))   
except pd.errors.ParserError:
    print("Berhenti di baris rusak, pakai data yang sudah terbaca.")

df = pd.concat(frames, ignore_index=True)
print("Distribusi awal:\n", df["tag"].value_counts())

# 2. Seimbangkan: ambil sampel acak yang sama banyak per genre
df = sample_per_genre(df, PER_GENRE)
print("Setelah dibuat seimbang:\n", df["tag"].value_counts())

# 3. Bersihkan teks
def clean(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", " ", text)       
    text = re.sub(r"[^a-z\s']", " ", text)     
    return re.sub(r"\s+", " ", text).strip()

df["clean"] = df["lyrics"].apply(clean)

# 4. Bagi data latih/uji (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["tag"], test_size=0.2, stratify=df["tag"], random_state=SEED)

# 5. TF-IDF (fit hanya di data latih)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
STOP = list(ENGLISH_STOP_WORDS | {"chorus", "ve", "ll", "re", "don", "didn", "doesn", "isn", "wasn", "won", "ain", "couldn", "wouldn", "shouldn"})
tfidf = TfidfVectorizer(stop_words=STOP, max_features=20000, ngram_range=(1, 2), min_df=5)
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# 6. Bandingkan beberapa model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
}
for name, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=GENRES),
                           display_labels=GENRES).plot(cmap="Blues")
    plt.title(name)
    plt.savefig(f"cm_{name.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
    plt.close()

# 7. Kata paling membedakan tiap genre (dari Logistic Regression) 
lr = models["Logistic Regression"]
vocab = tfidf.get_feature_names_out()
for i, genre in enumerate(lr.classes_):
    top = lr.coef_[i].argsort()[-10:][::-1]
    print(f"{genre}: {', '.join(vocab[top])}")