# Live Amazon Product Sentiment Agent (MLOps Edition)

An end-to-end, production-ready **NLP & MLOps pipeline** designed to extract and analyze customer sentiments for Amazon products in real-time. Upgraded from a static development environment into a dynamic interactive dashboard, this system leverages a **Hybrid Inference Engine** to balance data access speed with real-time web extraction.

---

## Project Architecture

The system transitions from an experimental Research & Development phase into a robust deployment pipeline:

```text
[User Insert Link] 
       │
       ▼
┌──────────────────────────────┐
│   Hybrid Query Router        │
└──────────────┬───────────────┘
               ├─────────────────────────┐ (If Matched)
               │ (If Missing)            ▼
               ▼                 ┌───────────────┐
┌──────────────────────────────┐ │ Local Secure  │
│  Anti-Fingerprint Scraper    │ │ Data Warehouse│
│   (BS4 + Mimic Headers)      │ └───────┬───────┘
└──────────────┬───────────────┘         │
               │                         │ (Hydrate Reviews)
               ▼                         ▼
        ┌──────────────────────────────────┐
        │  Serialized Asset Layer (joblib)  │
        │  - TfidfVectorizer (3,000 Features)│
        │  - Linear SVM Classifier         │
        └────────────────┬─────────────────┘
                         │
                         ▼
        ┌──────────────────────────────────┐
        │ Streamlit Web Dashboard Metrics  │
        └──────────────────────────────────┘
 Core Engineering Features:
Hybrid Query Router: When a user enters an Amazon URL, the system immediately checks the local pre-processed data warehouse (cleaned_amazon_reviews.csv). If matched, verified reviews are loaded instantly, bypassing network latency.

Defensive Web Scraper: If the product is missing from the database, an automated scraper using BeautifulSoup4 triggers, utilizing advanced HTTP header mimicry to prevent automated agent blocks.

Optimized Feature Layer: Re-engineered text vectorization using TfidfVectorizer capped at 3,000 max features with ngram_range=(1, 2) to capture contextual phrase pairs (e.g., "not good").

 Model Benchmarking & Performance
During the development phase, a benchmarking pipeline was built to evaluate three algorithms (Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Machine). The Linear SVM delivered the most robust boundary separation and was selected for production inference.

 Production Classifier Performance (Linear SVM)
The model achieves an outstanding overall accuracy of 88.32% on unseen testing data:

========================================
Overall Accuracy: 89.42%

Detailed Classification Report:
              precision    recall  f1-score   support

           0       0.80      0.69      0.74       300
           1       0.92      0.95      0.93      1070

    accuracy                           0.89      1370
   macro avg       0.86      0.82      0.84      1370
weighted avg       0.89      0.89      0.89      1370

========================================

Positive Class F1-Score: 0.93 (Highly reliable for customer satisfaction tracking).

 Project Structure & Clean Code Compliance
The repository adheres to modular software design and strict separation of concerns:

review.ipynb: Research, data profiling, model benchmarking, and asset serialization.

app.py: Production-grade deployment code implemented with strict Python type hints and functional isolation.

cleaned_amazon_reviews.csv: Local data warehouse utilized for instant query identification.

tfidf_vectorizer.pkl & sentiment_model.pkl: Serialized frozen inference assets managed via joblib.

```
## App Engine & Production Functions (`app.py`)

The production application is fully modularized into isolated components, adhering to the **Single Responsibility Principle (SRP)** and optimized for high-performance deployment:

### 1. Asset Isolation & Memory Optimization
* **`load_assets()`**: Implements Streamlit's `@st.cache_resource` memory caching. This ensures the 3,000-feature `TfidfVectorizer` and the trained `Linear SVM` model are loaded into memory **only once** upon server startup, preventing redundant disk reads and minimizing latency during concurrent user sessions.

### 2. Intelligent Data Warehouse Hydration
* **`load_and_prepare_database()`**: Dynamically inspects incoming file schemas. It contains an automated fallback mechanism to detect variations in source columns (`pageurl`, `product link`, `link`, `url`) and enforces strict text normalization (`strip().lower()`) across the database fields to guarantee stable lookups.

### 3. Defensive Bypassing Web Scraper
* **`scrape_live_reviews()`**: A robust web scraping engine engineered with:
  * **HTTP Header Mimicry:** Injects dynamic user-agents and connection handshakes to mirror real Google Chrome browser signatures, mitigating automated script blocks.
  * **Fallback Selectors:** Uses multi-layered CSS selector routing (`review-body` spans falling back to `.review-text-content` blocks) to adapt to unexpected shifts in Amazon's UI architecture.
  * **Rate Limiting Protection:** Enforces programmatic cooldowns (`time.sleep`) to simulate human pacing and prevent IP throttling.

### 4. Interactive Telemetry & Plotly Graphics
* **`display_dashboard_metrics()`**: Isolates operational Key Performance Indicators (KPIs), rendering high-visibility Streamlit metrics detailing review volume and sentiment percentages.
* **`plot_visualizations()`**: Migrated entirely to **Plotly Express** and **Plotly Graph Objects** to unlock dynamic, interactive web graphics:
  * **Deterministic Distribution:** Implements explicit categorical categorical reindexing (`.reindex([0, 1], fill_value=0)`) on data counts. This guarantees that both sentiment categories are represented in the correct sequence, completely eliminating label misalignment bugs even when a class has zero counts.
  * **Cyberpunk Aesthetic Telemetry:** Forces hardware-accelerated rendering mapping precise high-contrast neon hex values (`#00f2ff` for Positive, `#ff00ff` for Negative) embedded directly over custom dark-mode paper backgrounds (`#0F172A`).
  * **Advanced Callouts:** Utilizes dynamic text templates (`textposition='outside'`) to display exact frequency telemetry floating above the visual bars alongside optimized donut charts featuring hybrid ratio readouts.
