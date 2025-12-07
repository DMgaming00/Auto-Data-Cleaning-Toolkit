
# 🚀 CMPE 255 – Auto Data Toolkit  
### Final Project – Data Cleaning, Feature Engineering, Modeling, and Explainability  
**Author:** <Your Name>  
**Course:** CMPE 255 – Data Mining  
**Instructor:** <Professor Name>  
**Semester:** Fall 2023  

---

# 📌 Project Overview
This project implements an **Auto Data Toolkit** designed to simplify and automate major steps in a typical **CRISP-DM** data mining pipeline:

- Data understanding  
- Data cleaning & preprocessing  
- Feature engineering  
- Outlier removal  
- Train/test splitting  
- ML model training  
- Model evaluation  
- Explainability using SHAP  
- Partial Dependence Plots  
- Deployment as an interactive Streamlit web app  

The toolkit supports:

- **Titanic demo dataset**  
- **User-uploaded CSV files**  
- Multiple preprocessing options  
- Clean visual analytics  

---

# 🌐 Live Demo (Render Deployment)
> Add link here after deploying:
`https://<your-app>.onrender.com`

---

# 🧠 CRISP-DM Workflow

## 1. Business Understanding
Predict an outcome of interest (e.g., Titanic survival) using automated preprocessing and explainable ML techniques.

---

## 2. Data Understanding
The app displays:
- Dataset preview  
- Shape (rows × columns)  
- Missing value summary  
- Duplicate row removal  
- Date column detection & parsing  
- Data types  

---

## 3. Data Preparation
The toolkit provides multiple preprocessing options:

### 🔹 Missing Value Imputation  
- Mean  
- Median  
- KNN Imputer  
- Iterative Imputer  

### 🔹 Outlier Removal  
- None  
- IQR-based  
- IsolationForest-based  

### 🔹 Skewness Transformation  
- None  
- Log  
- Yeo-Johnson  

### 🔹 Categorical Encoding  
- One-Hot Encoding  
- Ordinal Encoding  

### 🔹 Feature Engineering  
- Automatic extraction of `year`, `month`, `day` for datetime columns  

### 🔹 Optional Feature Selection  
- Variance Threshold  
- RFE (Recursive Feature Elimination)  

---

## 4. Modeling
The toolkit trains a **RandomForestClassifier** inside a **scikit-learn Pipeline**, ensuring:

- Clean preprocessing  
- No data leakage  
- Reproducibility  

---

## 5. Evaluation
Metrics include:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Classification report  
- Confusion Matrix  
- ROC Curve with AUC  

---

## 6. Explainability

### 🧠 SHAP (Custom Beeswarm)
A cloud-safe implementation showing:

- Feature contributions  
- Positive/negative impacts  
- Top influential features  

### 📉 PDP (Partial Dependence Plots)
For features like Age, Fare, and Pclass.

---

# 📦 Project Structure

```
CMPE255_Final_Project/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
├── titanic.csv
│
└── project_pipeline/
    ├── __init__.py
    └── core.py
```

---

# 📥 Installation (Local)

### 1. Environment
```
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run app
```
streamlit run app.py
```

---

# 🧪 Dependencies

```
streamlit==1.29.0
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
shap==0.43.0
```

---

# 🏁 Conclusion
This project demonstrates a complete CRISP-DM workflow with:

- Automated data cleaning  
- ML modeling  
- Explainability  
- Deployment  

Ready for academic grading and real-world use.

---

# ✨ Author
<Your Name>  
San José State University  
CMPE 255 – Data Mining  
Fall 2023
