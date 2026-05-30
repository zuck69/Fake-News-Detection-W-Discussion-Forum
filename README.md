# Fake News Detection Platform & Integrated Discussion Forum

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/backend-Flask%20%2F%20Django-green.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

An advanced, full-stack web application built to combat digital misinformation through a dual-layered approach: **automated Machine Learning classification** and **crowdsourced community verification**. 

By pairing predictive analytics with an active discussion ecosystem, this platform bridges the gap between algorithmic detection and human context.

---

## 🚀 Key Features & Architecture

### 🔍 Automated Fake News Detection
* **Multi-Classifier Ensemble:** Developed training pipelines utilizing **Logistic Regression**, **Random Forest**, and **Gradient Boosting** architectures to evaluate and cross-reference text credibility.
* **Natural Language Processing:** Implemented a **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** to extract key linguistic features and semantic weights from raw input text.
* **On-Demand Predictions:** Provides an interactive interface where users can submit custom articles, paragraphs, or links for real-time veracity scoring.

### 💬 Interactive Discussion Forum
* **Crowdsourced Fact-Checking:** A dedicated community hub (`simple_forum.py`) allowing users to post trending news, debate findings, and collaboratively debunk viral hoaxes.
* **User Engagement Ecosystem:** Full support for thread creation, commenting, and user-driven discussions to leverage collective intelligence against misinformation.
* **Database & Migrations:** Powered by a robust relational schema with version-controlled database migrations (`migrations/`) ensuring seamless state management.

  ---

## 📂 Dataset Overview & Ingestion

This platform's predictive engine was built and evaluated using the **WELFake Dataset**, a premier, large-scale benchmarks corpus designed specifically for robust fake news classification.

* **Dataset Source:** [Kaggle - Fake News Classification (WELFake)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
* **Total Records:** 72,134 serialized news articles
  * **Real News Samples:** 35,028 entries
  * **Fake News Samples:** 37,106 entries
* **Target Vector / Classification Labels:**
  * `0` = **Fake News**
  * `1` = **Real News**

### Multi-Corpus Generalization
To prevent algorithmic overfitting and ensure the multi-classifier ensemble could identify structural patterns across different types of journalism, the underlying dataset synthesizes data across four major public corpora:
1. **Reuters** (Traditional news structures)
2. **Kaggle Fake News Corpus**
3. **McIntire Dataset**
4. **BuzzFeed Political** (Highly dynamic clickbait/viral news structures)

### Local Configuration
To run training sequences locally, download the dataset from the Kaggle source above and place the uncompressed `.csv` file into your local directory structure as follows:

```text
machine_learning/WELFake_Dataset.csv
```

---

## 🛠️ Installation & Environment Setup

Ensure your local virtual environment (`.venv`) is active before running the commands below.

### 1. Install Dependencies
Run the appropriate command to install all required libraries and packages listed in `requirements.txt`:

* **For Windows:**
  ```bash
  pip install -r requirements.txt

* **For Mac & Linux:**
  ```bash
  pip3 install -r requirements.txt

## 🛠️ 2. Machine Learning Model Setup

> ⚠️ **Note on Repository Assets:** Pre-trained `.pkl` model files and heavy `.csv` training datasets are excluded from this repository to optimize version control and maintain a lightweight codebase.

To initialize the detection engine locally:
1. Place your training dataset into the `machine learning/` directory.
2. Execute your local training script to generate the serialized machine learning objects (`.pkl` files).
3. Connect the generated components to the backend web application engine by linking the path in **`app/main/fnd.py` near line 68**.

---

## 📈 Engineering Insights & Dataset Constraints

> ### 🧠 Technical Note for Reviewers & Employers
> The models were trained on a historical corpus (e.g., `WELFake_Dataset`). Because digital journalism formats, clickbait structures, and misinformation syntax have evolved drastically over recent years, **the model may exhibit lower accuracy when predicting real-time, modern news articles.**
> 
> To adapt this application for modern production environments, the ingestion pipeline is designed to be plug-and-play: **developers can easily drop a modern dataset into the training pipeline to update the underlying vectorizer and weights without rewriting the core backend application logic.**

---

## 🔐 Testing & Mock Credentials

To review the administrative features, user roles, and protected forum operations without creating a new account, utilize the following pre-configured database seeds:

* **Administrator Username:** `admin`  
* **Default Password (All Pre-seeded Users):** `user`

---

## 👨‍💻 Core Technologies Used
* **Languages:** Python
* **ML/NLP Stack:** Scikit-Learn, Pandas, NumPy, Joblib
* **Web Architecture:** Relational Database (`app.db`), SQL-Alchemy/Migrations, Jinja Templates

## 🏗️ Architectural Evolution & API Integration

> ### 🛠️ Technical Insights for System Reviewers
> To fully appreciate the layout of the current codebase, it is helpful to understand the platform's architectural transitions and multi-database strategy.

### 🔌 API Integration Capability
* **Seamless Third-Party Extension:** The platform provides a production-ready API endpoint allowing developers to securely integrate our Fake News Detection models into external web, mobile, or desktop applications.
* **NoSQL Security & Identity Management:** To maintain ultra-low read latency and handle high-throughput authentication, active API keys and client access metadata are decoupled from the main database and stored in a **NoSQL architecture (configured in `main/routes.py` around line 212)**.

### 🔄 Legacy Core & De-scoped Features (Project X)
* **LLM Chatbot Deprecation:** The initial design phase of this application included a proprietary module codenamed **Project X**—an integrated Large Language Model (LLM) chatbot. This module was intentionally de-scoped and removed prior to the public GitHub deployment to isolate core codebase stability.
* **Regional Monetization Engine:** While hints of the enterprise subscription module remain in the codebase (`main/payment.py`), the billing system intentionally bypasses traditional Western payment gateways like Stripe. Instead, it utilizes a custom-engineered integration for **Khalti**, a prominent regional digital wallet and payment gateway system.
* **Hybrid Database Paradigm:** The platform leverages a hybrid storage model to match data shapes to performance needs:
  * **Relational Database (`app.db`):** Manages standard application states, user entities, migrations, and forum text.
  * **NoSQL Storage Engine:** Handles enterprise user subscription cycles, token usage trackers, and rate-limiting rules **(mapped inside `main/routes.py` near line 16)** to optimize memory performance during heavy request volumes.  
