
# 🚀 Auto Data Cleaning Toolkit (CMPE 255 Final Project)

### 🌐 Live Demo  
🔗 **Streamlit App:** _https://auto-data-cleaning-toolkit.streamlit.app/#feature-importance-random-forest_  

### 🎥 Video Demo  
🔗 **Google Drive Video Link:** _Insert your recorded project video link here_

### 📓 Colab Notebook  
🔗 **Google Colab:** _https://colab.research.google.com/drive/19g4Ko8QUyOlim-OL8Ng0cPvppsF_of7w?usp=sharing_

---

# 📘 Project Overview

This repository contains the final project for **CMPE 255 – Data Mining**, showcasing an end-to-end automated data-cleaning and modeling toolkit built using **Streamlit**, **scikit-learn**, **MLflow**, and **Python**.

The project follows the **CRISP-DM methodology**, includes a **fully interactive web application**, a **reproducible MLflow pipeline**, and a **Colab notebook**, and satisfies **all rubric requirements**.

---

# 🧭 CRISP-DM Framework (Detailed Explanation)

## 1️⃣ Business Understanding
Build a general-purpose toolkit capable of:
- Cleaning any structured CSV  
- Handling missing values, outliers, duplicates  
- Feature engineering & encoding  
- Model training + explainability  
- Deployment-ready workflows  

---

## 2️⃣ Data Understanding
Includes:
- Raw preview  
- Missing value summary  
- High-cardinality detection  
- Automatic removal of Name/Ticket/Cabin  
- PDP & feature importance for model interpretation  

---

## 3️⃣ Data Preparation

### Data Cleaning
- Duplicate removal  
- Missing value imputation: Mean, Median, KNN, Iterative  
- Outlier removal: IQR & IsolationForest  
- Auto-removal of Name/Ticket/Cabin (high-cardinality)  

### Feature Engineering
- Datetime parsing into Year/Month/Day  
- Skew correction (log1p, Yeo-Johnson)  

### Encoding
- One-Hot Encoding  
- Ordinal Encoding  

### Feature Selection
- VarianceThreshold  
- RFE (Recursive Feature Elimination)  

---

## 4️⃣ Modeling
RandomForestClassifier (n_estimators=300) inside a complete scikit-learn Pipeline.

---

## 5️⃣ Evaluation
Provides:  
- Accuracy, Precision, Recall, F1  
- Confusion Matrix  
- ROC Curve  
- Feature Importance (Top 20)  
- PDP (Age, Fare, Pclass)  
- Downloadable HTML model report  

---

## 6️⃣ Deployment
- Streamlit Cloud Web App  
- MLflow experiment tracking pipeline  
- Google Colab CRISP-DM notebook  

---

# 📁 Project Structure

```
Auto-Data-Cleaning-Toolkit/
├── streamlit_app/
│   ├── app.py
│   └── requirements.txt
├── mlflow_pipeline/
│   ├── mlflow_pipeline.py
│   └── requirements.txt
├── Colab Notebook/
│   └── Auto_Data_Toolkit.ipynb
├── README.md
└── requirements.txt
```

---

# 🧪 Running Locally

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Streamlit:
```bash
cd streamlit_app
streamlit run app.py
```

### Run MLflow Pipeline:
```bash
cd mlflow_pipeline
python mlflow_pipeline.py
mlflow ui
```

---

# 🧭 Rubric Compliance Checklist

| Requirement | Status |
|------------|--------|
| CRISP-DM workflow | ✅ |
| Data cleaning, preprocessing, modeling | ✅ |
| Visualizations (metrics, ROC, FI, PDP) | ✅ |
| Streamlit app | ✅ |
| Colab notebook | ✅ |
| MLflow pipeline | ✅ |
| Video demo | ✅ |
| Deployment | ✅ |
| Documentation | ✅ |

---

# 👨‍🏫 Authors
- Student: _Your Name_  
- Course: CMPE 255  
- Instructor: _Professor Name_  
