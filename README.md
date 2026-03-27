# CreditIQ — Loan Intelligence Dashboard
### By Harshit | Credit Score Prediction & Loan Approval System

---

## Quick Start (3 steps)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train models (only needed once)
```bash
python train_models.py
```

### Step 3 — Launch the dashboard
```bash
streamlit run app.py
```

The dashboard will open at: **http://localhost:8501**

---

## Folder Structure
```
dashboard/
├── app.py                  ← Main Streamlit dashboard
├── train_models.py         ← ML model training script
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── data/
│   ├── raw_dataset_2000.csv      ← Raw synthetic dataset (2000 rows)
│   └── cleaned_dataset.csv       ← Cleaned + feature-engineered dataset
├── models/                 ← Auto-created after running train_models.py
│   ├── model_lr.pkl
│   ├── model_rf_clf.pkl
│   ├── model_gb_clf.pkl
│   ├── model_ridge.pkl
│   ├── model_rf_reg.pkl
│   ├── model_kmeans.pkl
│   ├── scaler_clf.pkl
│   ├── scaler_reg.pkl
│   ├── scaler_clu.pkl
│   ├── clf_features.pkl
│   ├── reg_features.pkl
│   ├── clu_features.pkl
│   ├── metrics.pkl
│   ├── segment_labels.pkl
│   └── cluster_profiles.pkl
└── assets/
    └── correlation.png
```

---

## Dashboard Tabs

| Tab | What it does |
|-----|-------------|
| 🏠 Home & Overview | KPIs, model summary, platform intro |
| 📊 EDA & Distributions | Interactive distributions, bivariate plots |
| 🔗 Correlation Explorer | Heatmap, feature importance ranking |
| 🧩 Customer Segments | PCA cluster map, segment profiles, marketing playbook |
| 🤖 Model Performance | ROC curves, confusion matrix, feature importance |
| 🔮 Single Prediction | Form-based instant prediction with score gauge |
| 📂 Batch Upload & Score | Upload CSV → get scored output with marketing actions |

---

## Models Trained

### Classification (Loan Approval)
- Logistic Regression — baseline, explainable
- Random Forest Classifier — ensemble, feature importance
- Gradient Boosting — best model (~84% accuracy, ~90% AUC)

### Regression (Credit Score)
- Ridge Regression — best model (~91% R²)
- Random Forest Regressor (~85% R²)

### Clustering (Customer Segmentation)
- K-Means (k=4) → Prime, Emerging, Rebuilder, At-Risk

---

## Batch Upload Format
Minimum required columns:
```
Age, AnnualIncome_INR, DebtToIncomeRatio, CreditUtilization,
CreditHistoryYears, NumCreditAccounts, MissedPayments, Bankruptcies,
SavingsPct_Monthly, SavingsBalance_INR, ExistingDebt_INR,
FinLiteracyScore, UPI_TxnMonthly, EmergencyFundMonths
```
Use the "Download Sample Template" button inside Tab 7 to get a pre-formatted file.
