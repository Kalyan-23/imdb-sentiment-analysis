<img width="1905" height="923" alt="Screenshot 2026-04-16 220438" src="https://github.com/user-attachments/assets/7a265d24-24bc-4290-b1c0-4ed572d4f94e" /># 🎬 Movie Review Sentiment Analyzer

A machine learning-based web application that analyzes movie reviews and predicts whether the sentiment is **positive** or **negative** using Natural Language Processing (NLP) techniques.

---

## 🚀 Features

* 🔍 Sentiment classification (Positive / Negative)
* 🧠 Uses multiple ML models:

  * Logistic Regression
  * Naive Bayes
  * Support Vector Machine (SVM)
* 📊 Model performance comparison (Accuracy, F1 Score)
* 📈 Visualizations (confusion matrix, graphs)
* 🧹 Text preprocessing pipeline
* ⚡ Interactive UI built with Streamlit

---

## 🧠 NLP Techniques Used

* Text Cleaning (removing HTML, special characters)
* Lowercasing
* Tokenization (basic)
* Stopword removal
* **TF-IDF Vectorization**

---

## 🗂️ Project Structure

```
project/
│
├── app.py
├── IMDB Dataset.csv
├── requirements.txt
└── README.md
```

---
## 📊 Dataset

This project uses the IMDB movie reviews dataset containing **50,000 reviews** labeled as positive or negative.

Dataset source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

* Kaggle
* Dataset: IMDB Dataset of 50K Movie Reviews

---

## ⚙️ Technologies Used

* Python 🐍
* Streamlit ⚡
* Scikit-learn 🤖
* Pandas & NumPy 📊
* Matplotlib & Seaborn 📈

---

## 📌 How It Works

1. User inputs a movie review
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical vectors
4. ML models predict sentiment
5. Results are displayed with confidence score

---

## ⭐ Future Improvements

* Add deep learning models (LSTM, BERT)
* Deploy on cloud (AWS / Render)
* Add multi-language support
* Improve UI animations

---

Screenshot
<img width="1905" height="923" alt="Screenshot 2026-04-16 220438" src="https://github.com/user-attachments/assets/91c7680d-2a7f-49eb-80c1-c5b344a88f7d" />
![Uploading Screenshot 2026-04-16 220717.png…]()
<img width="1809" height="868" alt="Screenshot 2026-04-16 220745" src="https://github.com/user-attachments/assets/b5e117de-5877-4dd0-bf10-f50f3ce334c4" />






