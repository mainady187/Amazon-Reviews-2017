# Amazon Product Reviews Sentiment Analysis 2017 

A professional, end-to-end Text Classification project designed to analyze customer sentiments for Amazon products (specifically Echo Dot 2). This project implements a modular, production-ready Machine Learning pipeline to predict whether a review is **Positive** or **Negative** based on textual data.

---

## Project Goals & Requirements
* **Data Handling:** Clean missing text reviews and drop irrelevant metadata.
* **Supervised Learning:** Build a binary classification model (1 for Positive, 0 for Negative).
* **Feature Engineering:** Convert raw text into numerical features using advanced text vectorization.
* **Model Evaluation:** Assess performance using robust metrics like Accuracy, Precision, Recall, and Confusion Matrix.

---

## Tech Stack & Skills
* **Language:** Python
* **Data Processing:** Pandas, NumPy, SciPy
* **Machine Learning:** Scikit-Learn (`MultinomialNB`, `CountVectorizer`, `train_test_split`)
* **Data Visualization:** Matplotlib, Seaborn

---

## ⚙️ Data Handling & Pipeline Architecture
The project follows a clean, modular architecture based on software engineering best practices:

1. **Data Cleaning (`load_and_clean_data`):** Removes unneeded columns (`Pageurl`, `Declaration Text`, etc.), filters out rows with empty reviews, and maps ratings (>3 Stars) to binary target labels.
2. **Feature Preparation (`prepare_features`):** Vectorizes text utilizing `CountVectorizer` with English stop-words filtering and a maximum feature limit of 2,500.
3. **Model Training & Testing (`train_and_evaluate_model`):** Splits data using a stratified 80/20 ratio and trains a **Multinomial Naive Bayes** classifier.
4. **Visualizations (`plot_project_visualizations`):** Generates analytical plots detailing class distribution and model error margins.

---

## Model Performance & Results

The model achieved an outstanding overall accuracy of **88.32%** on the unseen testing dataset.

### Classification Report:
```text

              precision    recall  f1-score   support

           0       0.77      0.67      0.72       300
           1       0.91      0.94      0.93      1070

    accuracy                           0.88      1370
   macro avg       0.84      0.81      0.82      1370
weighted avg       0.88      0.88      0.88      1370
