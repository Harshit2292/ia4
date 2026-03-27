"""Train all ML models and save as .pkl files for the Streamlit dashboard."""

import pandas as pd
import numpy as np
import pickle, os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
MODELS = '/home/claude/dashboard/models'

df = pd.read_csv('/home/claude/dashboard/data/cleaned_dataset.csv')

# ── Feature engineering (consistent with cleaning pipeline) ───────
num_fill = df.select_dtypes(include=np.number).columns
df[num_fill] = df[num_fill].fillna(df[num_fill].median())

# Encode repayment history ordinal
rep_map = {'Excellent': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
if 'RepaymentHistory' in df.columns:
    df['RepaymentHistory_Ord'] = df['RepaymentHistory'].map(rep_map).fillna(2)

# ── Feature sets ───────────────────────────────────────────────────
CLF_FEATURES = [
    'Age', 'AnnualIncome_INR', 'DebtToIncomeRatio', 'CreditUtilization',
    'CreditHistoryYears', 'NumCreditAccounts', 'NumHardInquiries',
    'MissedPayments', 'Bankruptcies', 'SavingsPct_Monthly',
    'SavingsBalance_INR', 'ExistingDebt_INR', 'UPI_TxnMonthly',
    'FinLiteracyScore', 'EmergencyFundMonths', 'FamilyBurdenScore',
    'HasProperty', 'HasVehicle', 'HasCreditCard', 'HasFinancialPlan',
    'IncomeShockLast3Yr', 'AppBasedLoan', 'RepaymentHistory_Ord',
    'RiskScore', 'EMI_to_Income', 'SavingsToDebt', 'YearsEmployed',
    'NumDependents'
]
REG_FEATURES  = CLF_FEATURES.copy()
CLU_FEATURES  = [
    'AnnualIncome_INR', 'CreditScore', 'DebtToIncomeRatio',
    'CreditUtilization', 'MissedPayments', 'SavingsBalance_INR',
    'RiskScore', 'FinLiteracyScore', 'EmergencyFundMonths', 'UPI_TxnMonthly'
]

# filter to existing cols
CLF_FEATURES = [c for c in CLF_FEATURES if c in df.columns]
REG_FEATURES = [c for c in REG_FEATURES if c in df.columns]
CLU_FEATURES = [c for c in CLU_FEATURES if c in df.columns]

X_clf = df[CLF_FEATURES].fillna(0)
y_clf = df['LoanApproved'].astype(int)

X_reg = df[REG_FEATURES].fillna(0)
y_reg = df['CreditScore'].astype(float)

X_clu = df[CLU_FEATURES].fillna(0)

# ── Scalers ────────────────────────────────────────────────────────
scaler_clf = StandardScaler()
X_clf_s = scaler_clf.fit_transform(X_clf)

scaler_reg = StandardScaler()
X_reg_s = scaler_reg.fit_transform(X_reg)

scaler_clu = StandardScaler()
X_clu_s = scaler_clu.fit_transform(X_clu)

X_train, X_test, y_train, y_test = train_test_split(X_clf_s, y_clf, test_size=0.2, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg_s, y_reg, test_size=0.2, random_state=42)

metrics = {}

# ── 1. Logistic Regression ─────────────────────────────────────────
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_prob = lr.predict_proba(X_test)[:, 1]
metrics['logistic_regression'] = {
    'accuracy': round(accuracy_score(y_test, y_pred)*100, 2),
    'roc_auc':  round(roc_auc_score(y_test, y_prob)*100, 2),
    'f1':       round(f1_score(y_test, y_pred)*100, 2),
}
print(f"  LR → Acc={metrics['logistic_regression']['accuracy']}%  AUC={metrics['logistic_regression']['roc_auc']}%")

# ── 2. Random Forest Classifier ────────────────────────────────────
print("Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42,
                                 class_weight='balanced', n_jobs=-1)
rf_clf.fit(X_train, y_train)
y_pred2 = rf_clf.predict(X_test)
y_prob2 = rf_clf.predict_proba(X_test)[:, 1]
metrics['random_forest'] = {
    'accuracy': round(accuracy_score(y_test, y_pred2)*100, 2),
    'roc_auc':  round(roc_auc_score(y_test, y_prob2)*100, 2),
    'f1':       round(f1_score(y_test, y_pred2)*100, 2),
    'feature_importance': dict(zip(CLF_FEATURES,
                                    rf_clf.feature_importances_.round(4).tolist()))
}
print(f"  RF  → Acc={metrics['random_forest']['accuracy']}%  AUC={metrics['random_forest']['roc_auc']}%")

# ── 3. Gradient Boosting (XGBoost-style) ───────────────────────────
print("Training Gradient Boosting Classifier...")
gb_clf = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                                      random_state=42)
gb_clf.fit(X_train, y_train)
y_pred3 = gb_clf.predict(X_test)
y_prob3 = gb_clf.predict_proba(X_test)[:, 1]
metrics['gradient_boosting'] = {
    'accuracy': round(accuracy_score(y_test, y_pred3)*100, 2),
    'roc_auc':  round(roc_auc_score(y_test, y_prob3)*100, 2),
    'f1':       round(f1_score(y_test, y_pred3)*100, 2),
}
print(f"  GB  → Acc={metrics['gradient_boosting']['accuracy']}%  AUC={metrics['gradient_boosting']['roc_auc']}%")

# ── 4. Ridge Regression (Credit Score) ────────────────────────────
print("Training Ridge Regression (Credit Score)...")
ridge = Ridge(alpha=1.0)
ridge.fit(Xr_train, yr_train)
yr_pred = ridge.predict(Xr_test)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
metrics['ridge_regression'] = {
    'rmse': round(rmse, 2),
    'r2':   round(r2_score(yr_test, yr_pred)*100, 2),
    'mae':  round(np.mean(np.abs(yr_test - yr_pred)), 2),
}
print(f"  Ridge → RMSE={metrics['ridge_regression']['rmse']}  R²={metrics['ridge_regression']['r2']}%")

# ── 5. RF Regressor (Credit Score) ────────────────────────────────
print("Training Random Forest Regressor (Credit Score)...")
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_reg.fit(Xr_train, yr_train)
yr_pred2 = rf_reg.predict(Xr_test)
rmse2 = np.sqrt(mean_squared_error(yr_test, yr_pred2))
metrics['rf_regressor'] = {
    'rmse': round(rmse2, 2),
    'r2':   round(r2_score(yr_test, yr_pred2)*100, 2),
    'mae':  round(np.mean(np.abs(yr_test - yr_pred2)), 2),
}
print(f"  RF-Reg → RMSE={metrics['rf_regressor']['rmse']}  R²={metrics['rf_regressor']['r2']}%")

# ── 6. K-Means Clustering ─────────────────────────────────────────
print("Training K-Means Clustering (k=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
kmeans.fit(X_clu_s)
df['Cluster'] = kmeans.labels_
cluster_profiles = df.groupby('Cluster')[CLU_FEATURES + ['LoanApproved']].mean().round(2)
cluster_profiles['CreditScore_mean'] = df.groupby('Cluster')['CreditScore'].mean().round(2)

seg_labels = {}
cs_series = cluster_profiles['CreditScore_mean'].squeeze()
cs_order = cs_series.sort_values(ascending=False).index.tolist()
label_names = ['Prime', 'Emerging', 'Rebuilder', 'At-Risk']
for i, cl in enumerate(cs_order):
    seg_labels[int(cl)] = label_names[i]
metrics['kmeans'] = {
    'n_clusters': 4,
    'segment_labels': seg_labels,
    'cluster_sizes': df['Cluster'].value_counts().sort_index().to_dict(),
}
print(f"  KMeans → Clusters: {df['Cluster'].value_counts().sort_index().to_dict()}")

# ── Save everything ────────────────────────────────────────────────
objects = {
    'model_lr.pkl':        lr,
    'model_rf_clf.pkl':    rf_clf,
    'model_gb_clf.pkl':    gb_clf,
    'model_ridge.pkl':     ridge,
    'model_rf_reg.pkl':    rf_reg,
    'model_kmeans.pkl':    kmeans,
    'scaler_clf.pkl':      scaler_clf,
    'scaler_reg.pkl':      scaler_reg,
    'scaler_clu.pkl':      scaler_clu,
    'clf_features.pkl':    CLF_FEATURES,
    'reg_features.pkl':    REG_FEATURES,
    'clu_features.pkl':    CLU_FEATURES,
    'metrics.pkl':         metrics,
    'segment_labels.pkl':  seg_labels,
    'cluster_profiles.pkl': cluster_profiles,
}
for fname, obj in objects.items():
    with open(f'{MODELS}/{fname}', 'wb') as f:
        pickle.dump(obj, f)
print(f"\nAll {len(objects)} model files saved to {MODELS}/")
print("\nTraining complete!")
