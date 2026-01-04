# 📰 Fake News Detection System

A Machine Learning–based Fake News Detection system that classifies news articles as **Real** or **Fake** using **Natural Language Processing (NLP)** techniques.

This project uses **TF-IDF Vectorization** and **Logistic Regression**, making it simple, interpretable, and suitable as a baseline ML solution for academic purposes.

---

## 📌 Features

- Accepts user-entered news text from the terminal
- Cleans and preprocesses text automatically
- Classifies news as:
  - **REAL NEWS 🟢**
  - **FAKE NEWS 🔴**
  - **⚠️ Too short to classify reliably**
  - **type exit to quit**
- Displays model accuracy
- Interactive command-line interface

---

## 🧠 How It Works

1. **Dataset**
   - `Fake.csv` → Fake news articles
   - `True.csv` → Real news articles

2. **Preprocessing**
   - Converts text to lowercase
   - Removes punctuation and numbers
   - Removes extra whitespace

3. **Vectorization**
   - TF-IDF (Term Frequency–Inverse Document Frequency)
   - Uses unigrams and bigrams

4. **Model**
   - Logistic Regression
   - Trained on labeled data

5. **Prediction**
   - Uses learned linguistic patterns
   - Rejects very short inputs for reliability

---
## 🚀 Getting Started
1. **The European Union announced new economic sanctions on Russia following diplomatic discussions among member states.**
    - REAL NEWS 🟢

2. **Shocking secret revealed as world leaders panic over hidden global collapse plan exposed online.**
    - FAKE NEWS 🔴

3. NASA confirms water on Mars
    - ⚠️ Too short to classify reliably
