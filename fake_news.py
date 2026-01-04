import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ---------------------------
# Load Dataset
# ---------------------------
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------
# Clean Text
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


data["text"] = data["text"].apply(clean_text)


# ---------------------------
# Train-Test Split
# ---------------------------
X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ---------------------------
# Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=5,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ---------------------------
# Model Training
# ---------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
model.fit(X_train_vec, y_train)


# ---------------------------
# Accuracy
# ---------------------------
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# ---------------------------
# Prediction Function
# ---------------------------
def predict_news(news):
    if len(news.split()) < 8:
        return "⚠️ Too short to classify reliably"

    news_clean = clean_text(news)
    vector = vectorizer.transform([news_clean])

    prob = model.predict_proba(vector)[0]
    confidence = max(prob)

    if confidence < 0.60:
        return "⚠️ Uncertain — needs human verification"

    prediction = model.predict(vector)[0]
    return "REAL NEWS 🟢" if prediction == 1 else "FAKE NEWS 🔴"


# ---------------------------
# USER INPUT LOOP
# ---------------------------
print("\n📰 Fake News Detection System (Classical ML)")
print("✔ Best for long political & economic news")
print("✔ Uses TF-IDF + Logistic Regression")
print("Type 'exit' to quit\n")

while True:
    user_input = input("Paste news snippet here:\n")

    if user_input.lower() == "exit":
        print("Exiting program...")
        break

    result = predict_news(user_input)
    print("\nPrediction:", result)
    print("-" * 50)
