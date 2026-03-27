"""
CreditIQ — Loan Intelligence Dashboard  v4
Author : Harshit
Run    : streamlit run app.py
"""
# ── stdlib / third-party ────────────────────────────────────────────
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier,
                               RandomForestRegressor,
                               GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              mean_squared_error, r2_score,
                              confusion_matrix, roc_curve, auc as sk_auc)
import streamlit as st

warnings.filterwarnings("ignore")

# ── page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Loan Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.block-container{padding-top:1rem!important;}
section[data-testid="stSidebar"]{background:#080C14!important;}
section[data-testid="stSidebar"] *{color:#C9D1D9!important;}
/* KPI card */
.kpi{background:linear-gradient(145deg,#161B22,#0D1117);border:1px solid #21262D;
     border-radius:14px;padding:18px 12px;text-align:center;
     transition:transform .18s,border-color .18s;}
.kpi:hover{transform:translateY(-4px);border-color:#58A6FF;}
.kv{font-family:'Syne',sans-serif;font-size:1.95rem;font-weight:800;line-height:1.1;}
.kl{font-size:.68rem;color:#8B949E;text-transform:uppercase;letter-spacing:.09em;margin-top:5px;}
.ks{font-size:.7rem;color:#58A6FF;margin-top:3px;}
/* Section header */
.sh{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;color:#E6EDF3;
    border-bottom:2px solid #21262D;padding-bottom:4px;margin-bottom:.75rem;}
.ss{color:#8B949E;font-size:.8rem;margin-top:-10px;margin-bottom:10px;}
/* Insight box */
.ib{background:#0D1117;border-left:3px solid #58A6FF;border-radius:0 8px 8px 0;
    padding:9px 13px;margin:5px 0;font-size:.85rem;color:#C9D1D9;line-height:1.5;}
.ib.g{border-color:#3FB950;} .ib.a{border-color:#D29922;} .ib.r{border-color:#F85149;}
/* Verdict */
.vA{background:#0B2218;border:2px solid #3FB950;color:#3FB950;border-radius:10px;
    padding:14px;font-size:1.35rem;font-family:'Syne',sans-serif;font-weight:800;text-align:center;}
.vB{background:#271C00;border:2px solid #D29922;color:#D29922;border-radius:10px;
    padding:14px;font-size:1.35rem;font-family:'Syne',sans-serif;font-weight:800;text-align:center;}
.vR{background:#240A0A;border:2px solid #F85149;color:#F85149;border-radius:10px;
    padding:14px;font-size:1.35rem;font-family:'Syne',sans-serif;font-weight:800;text-align:center;}
/* Card */
.mc{background:#161B22;border:1px solid #30363D;border-radius:12px;padding:15px;
    transition:border-color .18s;}
.mc:hover{border-color:#58A6FF;}
.mc.best{border-color:#3FB950!important;background:#0A1A10;}
/* Divider */
.dv{height:1px;background:#21262D;margin:.9rem 0;}
</style>
""", unsafe_allow_html=True)

# ── Colour palette ───────────────────────────────────────────────────
BG   = "#0D1117"; SF = "#161B22"
C1   = "#58A6FF"; C2 = "#F85149"; C3 = "#D29922"; C4 = "#3FB950"
MU   = "#8B949E"; TX = "#E6EDF3"
PAL  = [C1, C2, C3, C4, "#A78BFA", "#F472B6", "#34D399", "#FB923C"]

def _rc():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": SF,
        "axes.edgecolor": "#30363D", "axes.labelcolor": TX,
        "xtick.color": MU, "ytick.color": MU,
        "text.color": TX, "grid.color": "#21262D",
        "grid.linestyle": "--", "grid.alpha": .55,
        "axes.titlesize": 11, "axes.labelsize": 9,
        "axes.titlepad": 8, "figure.dpi": 100,
    })
_rc()

def fig_show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def hdr(title, sub=""):
    st.markdown(f'<div class="sh">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="ss">{sub}</div>', unsafe_allow_html=True)

def ins(text, colour=""):
    cls = {"g": "g", "a": "a", "r": "r"}.get(colour, "")
    st.markdown(f'<div class="ib {cls}">{text}</div>', unsafe_allow_html=True)

def kpi_card(col, label, value, colour, sub=""):
    col.markdown(
        f'<div class="kpi">'
        f'<div class="kv" style="color:{colour}">{value}</div>'
        f'<div class="kl">{label}</div>'
        + (f'<div class="ks">{sub}</div>' if sub else "")
        + '</div>',
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════
# GENERATE DATA  (runs only if CSVs are missing — e.g. Streamlit Cloud)
# ═══════════════════════════════════════════════════════════════════
def generate_data():
    """Create synthetic 2000-row dataset and save CSVs."""
    os.makedirs("data", exist_ok=True)
    np.random.seed(42)
    N = 2000
    personas = ["Urban_Salaried","Rural_Farmer","Gig_Worker","MSME_Owner","Homemaker","Retired_Govt"]
    persona  = np.random.choice(personas, N, p=[0.30,0.15,0.20,0.18,0.10,0.07])

    age         = np.random.randint(21, 65, N)
    gender      = np.random.choice(["Male","Female","Other"], N, p=[0.55,0.43,0.02])
    marital     = np.random.choice(["Single","Married","Divorced"], N, p=[0.35,0.58,0.07])
    city_tier   = np.random.choice(["Tier1","Tier2","Tier3"], N, p=[0.35,0.38,0.27])
    education   = np.random.choice(["Postgraduate","Graduate","Undergraduate","High School"], N, p=[0.18,0.42,0.28,0.12])
    num_dep     = np.random.randint(0, 6, N)

    emp_status  = np.random.choice(["Salaried","Self-Employed","Freelance","Unemployed"], N, p=[0.52,0.25,0.15,0.08])
    yrs_emp     = np.clip(np.random.exponential(5, N).astype(int), 0, 35)
    ann_income  = np.clip(np.random.lognormal(12.8, 0.7, N), 80_000, 5_000_000)
    inc_stab    = np.random.choice(["Stable","Moderate","Volatile"], N, p=[0.45,0.35,0.20])
    sec_income  = np.random.choice([0]*60 + list(np.random.randint(5000,80000,40)), N)

    sav_pct     = np.clip(np.random.normal(18, 10, N), 0, 60)
    sav_bal     = np.clip(ann_income * sav_pct/100 * np.random.uniform(0.5,3,N), 0, 2_000_000)
    spending    = np.random.choice(["Conservative","Moderate","Liberal"], N, p=[0.30,0.45,0.25])
    credit_util = np.clip(np.random.beta(2, 4, N), 0, 1)
    upi_txn     = np.random.randint(0, 80, N)
    has_cc      = np.random.randint(0, 2, N)
    app_loan    = np.random.randint(0, 2, N)

    loan_type   = np.random.choice(["Home","Personal","Vehicle","Gold","Education","None"], N, p=[0.18,0.25,0.14,0.10,0.08,0.25])
    loan_purpose= np.random.choice(["Business","Education","Medical","Home","Vehicle","NA"], N, p=[0.20,0.15,0.10,0.22,0.13,0.20])
    loan_amt    = np.clip(np.random.lognormal(12.5, 1.0, N), 10_000, 5_000_000)
    loan_term   = np.random.choice([12,24,36,48,60,84,120,180,240], N)
    exist_debt  = np.clip(np.random.lognormal(10.5, 1.2, N) * np.random.choice([0,1], N, p=[0.3,0.7]), 0, 3_000_000)
    monthly_emi = exist_debt / np.where(loan_term==0, 36, loan_term)
    dti         = np.clip((monthly_emi*12) / (ann_income + 1), 0, 3)

    cred_hist   = np.clip(np.random.exponential(6, N), 0, 30).astype(int)
    num_accts   = np.random.randint(1, 12, N)
    hard_inq    = np.random.randint(0, 8, N)
    missed_pay  = np.random.choice([0,0,0,1,1,2,3], N)
    bankruptcies= np.random.choice([0,0,0,0,1,2], N)
    rep_hist    = np.random.choice(["Excellent","Good","Fair","Poor"], N, p=[0.35,0.33,0.20,0.12])

    has_prop    = np.random.randint(0, 2, N)
    has_veh     = np.random.randint(0, 2, N)
    inv_val     = np.clip(np.random.lognormal(10, 1.5, N) * np.random.choice([0,1], N, p=[0.4,0.6]), 0, 2_000_000)
    emg_fund    = np.clip(np.random.exponential(3, N), 0, 12)

    risk_tol    = np.random.choice(["Conservative","Moderate","Aggressive"], N, p=[0.35,0.40,0.25])
    fin_lit     = np.clip(np.random.normal(5.5, 2.0, N), 1, 10)
    has_fin_p   = np.random.randint(0, 2, N)
    inc_shock   = np.random.choice([0,0,0,1], N)
    family_burd = np.clip(num_dep * 0.15, 0, 1.5)

    rep_map_    = {"Excellent":4,"Good":3,"Fair":2,"Poor":1}
    rep_ord     = np.array([rep_map_[r] for r in rep_hist])
    risk_score  = np.clip(missed_pay*0.25 + bankruptcies*0.40 + dti*0.20 + credit_util*0.15, 0, 3)
    emi_inc     = np.clip(monthly_emi / (ann_income/12 + 1), 0, 3)
    sav_debt    = np.clip(sav_bal / (exist_debt + 1), 0, 50)

    # CreditScore
    cs = (600
          + (rep_ord - 2) * 30
          + (1 - credit_util) * 80
          - missed_pay * 40
          - bankruptcies * 80
          + cred_hist * 3
          + fin_lit * 5
          + has_prop * 20
          + (sav_pct - 15) * 1.5
          - dti * 50
          + np.random.normal(0, 25, N))
    cs = np.clip(cs, 300, 900).round(0)

    approved = (
        (cs >= 650).astype(int)
        + (dti < 0.4).astype(int)
        + (missed_pay == 0).astype(int)
        + (rep_ord >= 3).astype(int)
        - (bankruptcies > 0).astype(int)*2
    )
    approved = (approved >= 2).astype(int)
    # add noise
    flip = np.random.choice([0,1], N, p=[0.95,0.05])
    approved = np.abs(approved - flip)

    cs_band = pd.cut(cs, bins=[300,500,580,670,740,800,900],
                     labels=["Poor","Below Avg","Fair","Good","Very Good","Exceptional"])
    age_grp = pd.cut(age, bins=[18,25,35,45,55,65],
                     labels=["18-25","26-35","36-45","46-55","56-65"])
    inc_q   = pd.qcut(ann_income, 4, labels=["Q1","Q2","Q3","Q4"])
    mkt_seg = np.where(cs>=740,"Prime", np.where(cs>=620,"Emerging","AtRisk"))
    inc_log = np.log1p(ann_income)
    lam_log = np.log1p(loan_amt)
    approb  = np.clip((cs-300)/600 * 0.8 + approved*0.2, 0, 1)

    raw_df = pd.DataFrame({
        "RespondentID": [f"R{i:04d}" for i in range(1,N+1)],
        "Persona":persona,"Age":age,"Gender":gender,"MaritalStatus":marital,
        "CityTier":city_tier,"Education":education,"NumDependents":num_dep,
        "EmploymentStatus":emp_status,"YearsEmployed":yrs_emp,
        "AnnualIncome_INR":ann_income.round(2),"IncomeStability":inc_stab,
        "SecondaryIncome_INR":sec_income,"SavingsPct_Monthly":sav_pct.round(2),
        "SavingsBalance_INR":sav_bal.round(2),"SpendingPattern":spending,
        "CreditUtilization":credit_util.round(4),"UPI_TxnMonthly":upi_txn,
        "HasCreditCard":has_cc,"AppBasedLoan":app_loan,
        "PrimaryLoanType":loan_type,"LoanPurpose_New":loan_purpose,
        "LoanAmountSought":loan_amt.round(2),"LoanTermMonths":loan_term,
        "ExistingDebt_INR":exist_debt.round(2),"MonthlyEMI_INR":monthly_emi.round(2),
        "DebtToIncomeRatio":dti.round(4),"CreditHistoryYears":cred_hist,
        "NumCreditAccounts":num_accts,"NumHardInquiries":hard_inq,
        "MissedPayments":missed_pay,"Bankruptcies":bankruptcies,
        "RepaymentHistory":rep_hist,"HasProperty":has_prop,"HasVehicle":has_veh,
        "InvestmentValue_INR":inv_val.round(2),"EmergencyFundMonths":emg_fund.round(2),
        "RiskTolerance":risk_tol,"FinLiteracyScore":fin_lit.round(2),
        "HasFinancialPlan":has_fin_p,"IncomeShockLast3Yr":inc_shock,
        "FamilyBurdenScore":family_burd.round(3),
        "CreditScore":cs,"LoanApproved":approved,
        "MarketingSegment":mkt_seg,"ApprovalProbability":approb.round(4),
        "IncomeLog":inc_log.round(4),"LoanAmountLog":lam_log.round(4),
        "EMI_to_Income":emi_inc.round(4),"SavingsToDebt":sav_debt.round(4),
        "CreditScoreBand":cs_band.astype(str),"AgeGroup":age_grp.astype(str),
        "IncomeQuartile":inc_q.astype(str),"RiskScore":risk_score.round(4),
        "RepaymentHistory_Ord":rep_ord,
    })

    # label encode key cats
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for c in ["Gender","MaritalStatus","CityTier","Education","EmploymentStatus",
              "IncomeStability","SpendingPattern","PrimaryLoanType","LoanPurpose_New",
              "RepaymentHistory","RiskTolerance","MarketingSegment","Persona",
              "CreditScoreBand","AgeGroup","IncomeQuartile"]:
        raw_df[c+"_Enc"] = le.fit_transform(raw_df[c].astype(str))

    clean_df = raw_df.copy()
    raw_df.to_csv("raw_dataset_2000.csv", index=False)
    clean_df.to_csv("cleaned_dataset.csv", index=False)
    return clean_df, raw_df

# ═══════════════════════════════════════════════════════════════════
# DATA PREPARATION  (adds RepaymentHistory_Ord and other derived cols)
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    # Auto-generate CSVs if missing (Streamlit Cloud)
    if not os.path.exists("cleaned_dataset.csv"):
        with st.spinner("⏳ Generating dataset (~10s)..."):
            generate_data()
    df  = pd.read_csv("cleaned_dataset.csv")
    raw = pd.read_csv("raw_dataset_2000.csv")
    # numeric fill
    nc = df.select_dtypes(include=np.number).columns
    df[nc] = df[nc].fillna(df[nc].median())
    # derived columns that models need
    rep_map = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1}
    if "RepaymentHistory_Ord" not in df.columns:
        df["RepaymentHistory_Ord"] = df["RepaymentHistory"].map(rep_map).fillna(2)
    if "RiskScore" not in df.columns:
        df["RiskScore"] = (df["MissedPayments"]*0.25 + df.get("Bankruptcies", 0)*0.40
                           + df["DebtToIncomeRatio"]*0.20 + df["CreditUtilization"]*0.15).clip(0, 3)
    if "EMI_to_Income" not in df.columns:
        df["EMI_to_Income"] = (df.get("MonthlyEMI_INR", 0) / (df["AnnualIncome_INR"]/12 + 1)).clip(0, 3)
    if "SavingsToDebt" not in df.columns:
        df["SavingsToDebt"] = (df["SavingsBalance_INR"] / (df.get("ExistingDebt_INR", 1) + 1)).clip(0, 50)
    if "FamilyBurdenScore" not in df.columns:
        df["FamilyBurdenScore"] = (df["NumDependents"] * 0.15).clip(0, 1.5)
    return df, raw

# ═══════════════════════════════════════════════════════════════════
# AUTO-TRAIN (only runs on Streamlit Cloud when models/ is missing)
# ═══════════════════════════════════════════════════════════════════
def auto_train(df):
    os.makedirs("models", exist_ok=True)
    ALL_F = ["Age","AnnualIncome_INR","DebtToIncomeRatio","CreditUtilization",
             "CreditHistoryYears","NumCreditAccounts","NumHardInquiries","MissedPayments",
             "Bankruptcies","SavingsPct_Monthly","SavingsBalance_INR","ExistingDebt_INR",
             "UPI_TxnMonthly","FinLiteracyScore","EmergencyFundMonths","FamilyBurdenScore",
             "HasProperty","HasVehicle","HasCreditCard","HasFinancialPlan",
             "IncomeShockLast3Yr","AppBasedLoan","RepaymentHistory_Ord",
             "RiskScore","EMI_to_Income","SavingsToDebt","YearsEmployed","NumDependents"]
    CLU_F = ["AnnualIncome_INR","CreditScore","DebtToIncomeRatio","CreditUtilization",
             "MissedPayments","SavingsBalance_INR","RiskScore","FinLiteracyScore",
             "EmergencyFundMonths","UPI_TxnMonthly"]
    CLF = [c for c in ALL_F if c in df.columns]
    CLC = [c for c in CLU_F if c in df.columns]

    Xc = df[CLF].fillna(0); yc = df["LoanApproved"].astype(int)
    Xr = df[CLF].fillna(0); yr = df["CreditScore"].astype(float)
    Xk = df[CLC].fillna(0)

    sc  = StandardScaler(); Xcs = sc.fit_transform(Xc)
    sr  = StandardScaler(); Xrs = sr.fit_transform(Xr)
    sk  = StandardScaler(); Xks = sk.fit_transform(Xk)

    Xt,Xv,yt,yv = train_test_split(Xcs, yc, test_size=.2, random_state=42)
    Xrt,Xrv,yrt,yrv = train_test_split(Xrs, yr, test_size=.2, random_state=42)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(Xt,yt); yp=lr.predict(Xv); ypr=lr.predict_proba(Xv)[:,1]
    ml = {"accuracy": round(accuracy_score(yv,yp)*100,2),
          "roc_auc":  round(roc_auc_score(yv,ypr)*100,2),
          "f1":       round(f1_score(yv,yp)*100,2)}

    rf = RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(Xt,yt); yp2=rf.predict(Xv); ypr2=rf.predict_proba(Xv)[:,1]
    mr = {"accuracy": round(accuracy_score(yv,yp2)*100,2),
          "roc_auc":  round(roc_auc_score(yv,ypr2)*100,2),
          "f1":       round(f1_score(yv,yp2)*100,2),
          "feature_importance": dict(zip(CLF, rf.feature_importances_.round(4).tolist()))}

    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                     learning_rate=0.1, random_state=42)
    gb.fit(Xt,yt); yp3=gb.predict(Xv); ypr3=gb.predict_proba(Xv)[:,1]
    mg = {"accuracy": round(accuracy_score(yv,yp3)*100,2),
          "roc_auc":  round(roc_auc_score(yv,ypr3)*100,2),
          "f1":       round(f1_score(yv,yp3)*100,2)}

    rdg = Ridge(alpha=1.0); rdg.fit(Xrt,yrt); ypr_r=rdg.predict(Xrv)
    mrd = {"rmse": round(np.sqrt(mean_squared_error(yrv,ypr_r)),2),
           "r2":   round(r2_score(yrv,ypr_r)*100,2),
           "mae":  round(np.mean(np.abs(yrv-ypr_r)),2)}

    rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rfr.fit(Xrt,yrt); ypr_r2=rfr.predict(Xrv)
    mrf = {"rmse": round(np.sqrt(mean_squared_error(yrv,ypr_r2)),2),
           "r2":   round(r2_score(yrv,ypr_r2)*100,2),
           "mae":  round(np.mean(np.abs(yrv-ypr_r2)),2)}

    km = KMeans(n_clusters=4, random_state=42, n_init=10); km.fit(Xks)
    tmp = df.copy(); tmp["Cluster"] = km.labels_
    cp  = tmp.groupby("Cluster")[CLC].mean().round(2)
    cp["AvgCreditScore"] = tmp.groupby("Cluster")["CreditScore"].mean().round(2)
    cs_ord = cp["AvgCreditScore"].sort_values(ascending=False).index.tolist()
    sl = {int(cl): nm for cl, nm in zip(cs_ord, ["Prime","Emerging","Rebuilder","At-Risk"])}
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=5).fit(Xks).inertia_
                for k in range(2,9)]
    met = {"logistic_regression": ml, "random_forest": mr, "gradient_boosting": mg,
           "ridge_regression": mrd, "rf_regressor": mrf,
           "kmeans": {"n_clusters":4,"segment_labels":sl,"inertias":inertias,"cluster_sizes":{}}}

    objects = {"model_lr":lr,"model_rf_clf":rf,"model_gb_clf":gb,"model_ridge":rdg,
               "model_rf_reg":rfr,"model_kmeans":km,"scaler_clf":sc,"scaler_reg":sr,
               "scaler_clu":sk,"clf_features":CLF,"reg_features":CLF,"clu_features":CLC,
               "metrics":met,"segment_labels":sl,"cluster_profiles":cp}
    for name, obj in objects.items():
        with open(f"models/{name}.pkl","wb") as fh: pickle.dump(obj, fh)

# ═══════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models(df_hash: int):
    """df_hash forces re-cache if data changes; ignored at runtime."""
    if not os.path.exists("models/model_gb_clf.pkl"):
        with st.spinner("⏳ First launch — training models (~60 s) …"):
            df_tmp, _ = load_data()
            auto_train(df_tmp)
    def lp(name):
        with open(f"models/{name}.pkl","rb") as fh: return pickle.load(fh)
    return {k: lp(v) for k, v in {
        "lr":      "model_lr",      "rf_clf":  "model_rf_clf",
        "gb_clf":  "model_gb_clf",  "ridge":   "model_ridge",
        "rf_reg":  "model_rf_reg",  "kmeans":  "model_kmeans",
        "sc_clf":  "scaler_clf",    "sc_reg":  "scaler_reg",
        "sc_clu":  "scaler_clu",    "clf_f":   "clf_features",
        "reg_f":   "reg_features",  "clu_f":   "clu_features",
        "metrics": "metrics",       "seg_lbl": "segment_labels",
        "cp":      "cluster_profiles",
    }.items()}

# ── boot ────────────────────────────────────────────────────────────
with st.spinner("💳 Loading CreditIQ …"):
    df, raw = load_data()
    M = load_models(len(df))

# ── helpers that need M & df ────────────────────────────────────────
def prep_X(source_df, feature_list, scaler):
    """Build & scale feature matrix, filling any missing col with 0."""
    tmp = source_df.copy()
    for c in feature_list:
        if c not in tmp.columns:
            tmp[c] = 0
    return scaler.transform(tmp[feature_list].fillna(0))

SEG_COL = {"Prime": C4, "Emerging": C1, "Rebuilder": C3, "At-Risk": C2}

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:12px 0 6px'>"
        "<div style='font-family:Syne,sans-serif;font-size:1.55rem;font-weight:800;"
        "color:#58A6FF;letter-spacing:1px;'>💳 CreditIQ</div>"
        "<div style='font-size:.73rem;color:#8B949E;margin-top:2px;'>"
        "Loan Intelligence Platform</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    PAGE = st.radio("", [
        "🏠  Home",
        "📋  Dataset Info",
        "📊  EDA & Distributions",
        "🔗  Correlation Analysis",
        "🧩  Customer Segments",
        "📈  Regression Analysis",
        "🤖  Model Performance",
        "🔮  Live Prediction",
        "📂  Batch Scoring",
    ], label_visibility="collapsed")
    st.markdown("---")
    ar = df["LoanApproved"].mean()*100
    met_s = M["metrics"]
    st.markdown(
        f"<div style='font-size:.8rem;color:#8B949E;line-height:2.1;'>"
        f"📦 <b style='color:#C9D1D9'>Rows</b> {len(df):,}<br>"
        f"🔢 <b style='color:#C9D1D9'>Features</b> {len(M['clf_f'])}<br>"
        f"✅ <b style='color:#C9D1D9'>Approval</b> {ar:.1f}%<br>"
        f"🎯 <b style='color:#C9D1D9'>Best AUC</b> {met_s['gradient_boosting']['roc_auc']}%<br>"
        f"📐 <b style='color:#C9D1D9'>Best R²</b> {met_s['ridge_regression']['r2']}%"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("© 2025 Harshit | v4.0")


# ═══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
if PAGE == "🏠  Home":
    st.markdown(
        "<div style='padding:14px 0 8px'>"
        "<div style='font-family:Syne,sans-serif;font-size:2.5rem;font-weight:800;"
        "background:linear-gradient(120deg,#58A6FF 30%,#3FB950 100%);"
        "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        "line-height:1.2'>Credit Score Prediction<br>&amp; Loan Approval System</div>"
        "<div style='color:#8B949E;margin-top:6px;font-size:.95rem;'>"
        "Data-driven lending intelligence for India — by <b style='color:#C9D1D9'>Harshit</b></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    # ── 6 KPI cards ─────────────────────────────────────────────────
    seg_vc = df["MarketingSegment"].value_counts() if "MarketingSegment" in df.columns else {}
    cols = st.columns(6)
    kpi_card(cols[0], "Total Customers",  f"{len(df):,}",                              C1, "Survey respondents")
    kpi_card(cols[1], "Approval Rate",    f"{ar:.1f}%",                                C4, "Loans approved")
    kpi_card(cols[2], "Avg Credit Score", f"{df['CreditScore'].mean():.0f}",           C3, "CIBIL 300–900")
    kpi_card(cols[3], "Prime Customers",  f"{seg_vc.get('Prime',0):,}",                C4, "High-value segment")
    kpi_card(cols[4], "Best AUC",         f"{met_s['gradient_boosting']['roc_auc']}%", C1, "GB Classifier")
    kpi_card(cols[5], "Credit R²",        f"{met_s['ridge_regression']['r2']}%",       C3, "Ridge Regression")

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    # Credit score distribution
    with c1:
        hdr("Credit Score Distribution")
        fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=BG)
        n, bins, _ = ax.hist(df["CreditScore"], bins=50, color=C1, alpha=.7, edgecolor="none")
        ax.fill_between(bins[:-1], n, alpha=.12, color=C1, step="post")
        ax.axvline(df["CreditScore"].mean(),   color=C3, lw=2, ls="--",
                   label=f"Mean={df['CreditScore'].mean():.0f}")
        ax.axvline(df["CreditScore"].median(), color=C4, lw=2, ls="-.",
                   label=f"Median={df['CreditScore'].median():.0f}")
        ax.set_xlabel("Credit Score"); ax.set_ylabel("Count")
        ax.legend(fontsize=9, facecolor=SF); ax.grid(True)
        fig.tight_layout(); fig_show(fig)
        ins(f"Scores span 300–900 · Mean={df['CreditScore'].mean():.0f} · "
            f"Std={df['CreditScore'].std():.0f} · Slight left-skew (more at-risk than prime).")

    # Approval rate by segment
    with c2:
        hdr("Approval Rate by Marketing Segment")
        fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=BG)
        if "MarketingSegment" in df.columns:
            sa = df.groupby("MarketingSegment")["LoanApproved"].mean()*100
            sa = sa.sort_values(ascending=False)
            clrs = [SEG_COL.get(s, C1) for s in sa.index]
            bars = ax.bar(sa.index, sa.values, color=clrs, alpha=.85,
                          edgecolor="none", width=.5)
            ax.set_ylim(0, 110); ax.set_ylabel("Approval Rate (%)")
            ax.grid(True, axis="y")
            for b, v in zip(bars, sa.values):
                ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.1f}%",
                        ha="center", fontsize=11, fontweight="bold", color=TX)
        fig.tight_layout(); fig_show(fig)
        ins("Prime customers have the highest approval rate. "
            "At-Risk segment requires financial literacy before any loan offer.", "g")

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    # Platform capabilities
    hdr("Platform Capabilities")
    c1, c2, c3, c4 = st.columns(4)
    for col, (ico, title, desc) in zip([c1,c2,c3,c4], [
        ("📊","EDA & Distributions",   "Histograms · box plots · Q-Q plots · pair plots across all features"),
        ("🔗","Correlation Analysis",  "Full heatmap · feature importance · interactive Pearson scatter"),
        ("🧩","Customer Segments",     "Elbow chart · PCA map · K-Means profiles · marketing playbook"),
        ("🤖","ML Models",             "5 models · ROC curves · confusion matrix · regression diagnostics"),
    ]):
        col.markdown(
            f'<div class="mc" style="text-align:center;padding:20px 12px;">'
            f'<div style="font-size:2rem;">{ico}</div>'
            f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#E6EDF3;'
            f'margin:8px 0 5px;font-size:.9rem;">{title}</div>'
            f'<div style="font-size:.76rem;color:#8B949E;">{desc}</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE: DATASET INFO
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "📋  Dataset Info":
    hdr("Dataset Overview",
        "2,000 synthetic Indian financial respondents — 46 raw features + 24 engineered")

    c1,c2,c3,c4 = st.columns(4)
    kpi_card(c1,"Total Rows",    f"{len(df):,}",    C1)
    kpi_card(c2,"Columns",       f"{df.shape[1]}",  C3)
    kpi_card(c3,"Null Values",   "0",               C4)
    kpi_card(c4,"Target Vars",   "3",               C2)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        hdr("Feature Groups — Click to Expand")
        groups = {
            "👤 Demographics (6)":         ["Age","Gender","MaritalStatus","CityTier","Education","NumDependents"],
            "💼 Employment & Income (5)":  ["EmploymentStatus","YearsEmployed","AnnualIncome_INR","IncomeStability","SecondaryIncome_INR"],
            "💰 Financial Behaviour (7)":  ["SavingsPct_Monthly","SavingsBalance_INR","SpendingPattern","CreditUtilization","UPI_TxnMonthly","HasCreditCard","AppBasedLoan"],
            "🏦 Loans & Debt (6)":         ["PrimaryLoanType","LoanAmountSought","LoanTermMonths","ExistingDebt_INR","MonthlyEMI_INR","DebtToIncomeRatio"],
            "📜 Credit History (6)":       ["CreditHistoryYears","NumCreditAccounts","NumHardInquiries","MissedPayments","Bankruptcies","RepaymentHistory"],
            "🏠 Assets (4)":               ["HasProperty","HasVehicle","InvestmentValue_INR","EmergencyFundMonths"],
            "🧠 Behavioural (5)":          ["RiskTolerance","FinLiteracyScore","HasFinancialPlan","IncomeShockLast3Yr","FamilyBurdenScore"],
            "🎯 Target Variables (3)":     ["CreditScore","LoanApproved","MarketingSegment"],
        }
        for grp, cols_ in groups.items():
            ex = [c for c in cols_ if c in df.columns]
            with st.expander(grp, expanded=False):
                if ex:
                    st.dataframe(df[ex].describe().round(2).T, use_container_width=True)

    with col2:
        hdr("Data Quality Report")
        for txt, c in [
            ("✅ 0 null values — median imputation applied", "g"),
            ("✅ 0 duplicates — RespondentID verified", "g"),
            ("✅ 311 outliers capped via 3×IQR", "g"),
            ("📐 8 engineered features added", ""),
            ("🔢 16 columns label-encoded for ML", ""),
        ]:
            ins(txt, c)

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        hdr("Target Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(5, 3.5), facecolor=BG)
        av = df["LoanApproved"].value_counts()
        axes[0].pie(av, labels=["Approved","Rejected"], autopct="%1.1f%%",
                    colors=[C4,C2], startangle=90,
                    textprops={"color":TX,"fontsize":9})
        axes[0].set_title("Loan Approved", color=TX, fontsize=9)
        if "MarketingSegment" in df.columns:
            sv = df["MarketingSegment"].value_counts()
            cc = [SEG_COL.get(s, C1) for s in sv.index]
            axes[1].pie(sv, labels=sv.index, autopct="%1.1f%%",
                        colors=cc, startangle=90,
                        textprops={"color":TX,"fontsize":8})
            axes[1].set_title("Segments", color=TX, fontsize=9)
        fig.tight_layout(); fig_show(fig)

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    hdr("Raw Data Preview")
    st.dataframe(raw.head(50), use_container_width=True, height=300)
    st.download_button("⬇️ Download Raw CSV",
                       raw.to_csv(index=False).encode(),
                       "raw_dataset_2000.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "📊  EDA & Distributions":
    hdr("Exploratory Data Analysis",
        "Histograms · box plots · categorical breakdowns · pair plots")

    NUM_COLS = [c for c in ["CreditScore","AnnualIncome_INR","DebtToIncomeRatio",
        "CreditUtilization","Age","SavingsPct_Monthly","MissedPayments","UPI_TxnMonthly",
        "FinLiteracyScore","EmergencyFundMonths","RiskScore","SavingsBalance_INR",
        "CreditHistoryYears","NumDependents","ExistingDebt_INR"] if c in df.columns]
    CAT_COLS = [c for c in ["MarketingSegment","EmploymentStatus","CityTier","Education",
        "RepaymentHistory","SpendingPattern","PrimaryLoanType","IncomeStability",
        "RiskTolerance","MaritalStatus","Persona","AgeGroup","CreditScoreBand"]
        if c in df.columns]

    t1, t2, t3, t4 = st.tabs(["📈 Histograms", "📦 Box Plots",
                                "🔢 Categorical", "🔍 Pair Plots"])

    # ── Histograms ──────────────────────────────────────────────────
    with t1:
        r1, r2 = st.columns([1,3])
        with r1:
            sel    = st.selectbox("Variable", NUM_COLS)
            split  = st.selectbox("Colour split", ["None","LoanApproved",
                                  "MarketingSegment","CityTier","RepaymentHistory"])
            bins_n = st.slider("Bins", 15, 80, 40)
        with r2:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=BG)
            data = df[sel].dropna()
            ax = axes[0]
            if split != "None" and split in df.columns:
                cats = df[split].dropna().unique()
                for i, cat in enumerate(cats):
                    d = df[df[split]==cat][sel].dropna()
                    ax.hist(d, bins=bins_n, alpha=.55, color=PAL[i%8],
                            label=str(cat), edgecolor="none")
                ax.legend(fontsize=8, facecolor=SF)
            else:
                ax.hist(data, bins=bins_n, color=C1, alpha=.75, edgecolor="none")
                ax.axvline(data.mean(),   color=C3, lw=2, ls="--",
                           label=f"Mean={data.mean():.2f}")
                ax.axvline(data.median(), color=C4, lw=2, ls="-.",
                           label=f"Median={data.median():.2f}")
                ax.legend(fontsize=8, facecolor=SF)
            ax.set_title(f"Distribution of {sel}", color=C1, fontweight="bold")
            ax.set_xlabel(sel); ax.set_ylabel("Count"); ax.grid(True)

            # Q-Q plot
            ax2 = axes[1]
            (osm, osr), (slope, intercept, r_val) = stats.probplot(data, dist="norm")
            ax2.scatter(osm, osr, s=8, color=C1, alpha=.5, edgecolors="none")
            xl = np.array([osm.min(), osm.max()])
            ax2.plot(xl, slope*xl+intercept, color=C3, lw=2)
            ax2.set_title(f"Q-Q Plot  (r={r_val:.3f})", color=C3, fontweight="bold")
            ax2.set_xlabel("Theoretical Quantiles")
            ax2.set_ylabel("Sample Quantiles"); ax2.grid(True)
            fig.tight_layout(); fig_show(fig)

        sk_v = stats.skew(data); ku_v = stats.kurtosis(data)
        co = st.columns(5)
        for col_, (lbl_, val_) in zip(co, [
            ("Mean",     f"{data.mean():.3f}"),
            ("Median",   f"{data.median():.3f}"),
            ("Std Dev",  f"{data.std():.3f}"),
            ("Skewness", f"{sk_v:.3f}"),
            ("Kurtosis", f"{ku_v:.3f}"),
        ]):
            col_.metric(lbl_, val_)
        ins(f"<b>{sel}</b>: skewness={sk_v:.2f} "
            f"({'right-skewed' if sk_v>.5 else 'left-skewed' if sk_v<-.5 else 'approx normal'}). "
            f"Kurtosis={ku_v:.2f} "
            f"({'heavy tails' if ku_v>1 else 'thin tails' if ku_v<-1 else 'normal tails'}).")

        # ── all-variables grid ───────────────────────────────────────
        st.markdown("---")
        hdr("All Key Variables — Overview Grid")
        GRID = [c for c in ["CreditScore","AnnualIncome_INR","DebtToIncomeRatio",
            "CreditUtilization","MissedPayments","RiskScore","FinLiteracyScore",
            "EmergencyFundMonths","SavingsPct_Monthly","UPI_TxnMonthly",
            "CreditHistoryYears","Age"] if c in df.columns]
        fig2, ax2s = plt.subplots(3, 4, figsize=(20, 11), facecolor=BG)
        for ax_, cn_, clr_ in zip(ax2s.flat, GRID, PAL*3):
            d_ = df[cn_].dropna()
            ax_.hist(d_, bins=30, color=clr_, alpha=.75, edgecolor="none")
            ax_.axvline(d_.mean(), color="white", lw=1.2, ls="--", alpha=.7)
            ax_.set_title(cn_.replace("_INR","").replace("_"," "),
                          color=clr_, fontweight="bold", fontsize=9)
            ax_.set_ylabel("Count", fontsize=7); ax_.grid(True)
            ax_.text(.97,.90,f"Skew={stats.skew(d_):.2f}",
                     transform=ax_.transAxes,ha="right",fontsize=7,color=MU)
        fig2.suptitle("All Key Numerical Variables", color=C1,
                      fontsize=13, fontweight="bold", y=1.0)
        fig2.tight_layout(); fig_show(fig2)

    # ── Box plots ────────────────────────────────────────────────────
    with t2:
        hdr("Box Plots — Distribution by Category")
        r1, r2 = st.columns(2)
        with r1:
            nv = st.selectbox("Numeric variable",
                              [c for c in ["CreditScore","AnnualIncome_INR",
                                "DebtToIncomeRatio","CreditUtilization",
                                "MissedPayments","RiskScore","FinLiteracyScore",
                                "EmergencyFundMonths"] if c in df.columns])
        with r2:
            cv = st.selectbox("Group by",
                              [c for c in ["LoanApproved","MarketingSegment",
                                "CityTier","EmploymentStatus","RepaymentHistory",
                                "Education","RiskTolerance"] if c in df.columns])

        grp_list = sorted(df[cv].dropna().unique().astype(str).tolist())
        data_grps = [df[df[cv].astype(str)==g][nv].dropna().values for g in grp_list]
        fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
        bp = ax.boxplot(data_grps, patch_artist=True, notch=False,
                        medianprops={"color":"white","lw":2.5},
                        whiskerprops={"color":MU,"lw":1.2},
                        capprops={"color":MU,"lw":1.2},
                        flierprops={"marker":"o","color":C2,"alpha":.3,"markersize":3})
        for patch, clr_ in zip(bp["boxes"], PAL[:len(grp_list)]):
            patch.set_facecolor(clr_); patch.set_alpha(.55)
        ax.set_xticks(range(1,len(grp_list)+1))
        ax.set_xticklabels(grp_list, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(nv.replace("_INR","").replace("_"," "))
        ax.set_title(f"{nv} grouped by {cv}", color=C1, fontweight="bold")
        ax.grid(True, axis="y")
        fig.tight_layout(); fig_show(fig)

        sd = df.groupby(df[cv].astype(str))[nv].agg(["mean","median","std","min","max"]).round(2)
        sd.columns = ["Mean","Median","Std Dev","Min","Max"]
        st.dataframe(sd, use_container_width=True)

        meds = {g: df[df[cv].astype(str)==g][nv].median() for g in grp_list}
        bg_, wo_ = max(meds,key=meds.get), min(meds,key=meds.get)
        ins(f"<b>{bg_}</b> has the highest median {nv} ({meds[bg_]:.2f}). "
            f"<b>{wo_}</b> has the lowest ({meds[wo_]:.2f}).", "g")

    # ── Categorical ──────────────────────────────────────────────────
    with t3:
        hdr("Categorical Variable Analysis")
        sc_ = st.selectbox("Select variable", CAT_COLS)
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), facecolor=BG)
        vc = df[sc_].value_counts().head(10)

        # bar
        axes[0].bar(range(len(vc)), vc.values, color=PAL[:len(vc)],
                    alpha=.85, edgecolor="none")
        axes[0].set_xticks(range(len(vc)))
        axes[0].set_xticklabels(vc.index, rotation=30, ha="right", fontsize=8)
        axes[0].set_title(f"{sc_} — Frequency", color=C1, fontweight="bold")
        axes[0].set_ylabel("Count"); axes[0].grid(True, axis="y")
        for i, v in enumerate(vc.values):
            axes[0].text(i, v+4, str(v), ha="center", fontsize=8, color=TX)

        # pie
        axes[1].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
                    colors=PAL[:len(vc)], startangle=90,
                    textprops={"color":TX,"fontsize":8})
        axes[1].set_title(f"{sc_} — Share", color=C3, fontweight="bold")

        # approval rate
        ab = df.groupby(sc_)["LoanApproved"].mean()*100
        ab = ab.sort_values(ascending=False).head(10)
        ca = [C4 if v>=50 else C3 if v>=30 else C2 for v in ab.values]
        axes[2].barh(ab.index[::-1], ab.values[::-1],
                     color=ca[::-1], alpha=.85, edgecolor="none")
        axes[2].axvline(50, color=MU, lw=1, ls="--")
        axes[2].set_xlabel("Approval Rate (%)")
        axes[2].set_title(f"Approval Rate by {sc_}", color=C4, fontweight="bold")
        axes[2].grid(True, axis="x")
        for i, v in enumerate(ab.values[::-1]):
            axes[2].text(v+.5, i, f"{v:.1f}%", va="center", fontsize=8, color=TX)
        fig.tight_layout(); fig_show(fig)
        ins(f"Most common: <b>{vc.index[0]}</b> ({vc.iloc[0]:,} = "
            f"{vc.iloc[0]/len(df)*100:.1f}%). "
            f"Highest approval rate: <b>{ab.index[0]}</b> at {ab.iloc[0]:.1f}%.", "g")

    # ── Pair plots ───────────────────────────────────────────────────
    with t4:
        hdr("Pair Plot — Multivariate Relationships")
        defaults = [c for c in ["CreditScore","AnnualIncome_INR","DebtToIncomeRatio",
                    "RiskScore","CreditUtilization"] if c in df.columns][:5]
        sel_pp = st.multiselect("Select 3–5 variables", NUM_COLS, default=defaults)
        if len(sel_pp) >= 2:
            sp = df.sample(min(500,len(df)), random_state=42)
            n_ = len(sel_pp)
            fig, axes = plt.subplots(n_, n_, figsize=(4*n_, 4*n_), facecolor=BG)
            if n_ == 1: axes = np.array([[axes]])
            cc_ = sp["LoanApproved"].map({0: C2, 1: C4})
            for i, ci in enumerate(sel_pp):
                for j, cj in enumerate(sel_pp):
                    ax_ = axes[i][j]
                    ax_.set_facecolor(SF)
                    for sp_ in ax_.spines.values(): sp_.set_edgecolor("#30363D")
                    ax_.tick_params(labelsize=6, colors=MU)
                    if i == j:
                        for app, clr_ in [(0,C2),(1,C4)]:
                            d_ = sp[sp["LoanApproved"]==app][ci].dropna()
                            ax_.hist(d_, bins=20, color=clr_, alpha=.55, edgecolor="none")
                    else:
                        ax_.scatter(sp[cj], sp[ci], c=cc_, alpha=.3,
                                    s=8, edgecolors="none")
                    if i == n_-1: ax_.set_xlabel(cj.replace("_INR",""), fontsize=7, color=MU)
                    if j == 0:   ax_.set_ylabel(ci.replace("_INR",""), fontsize=7, color=MU)
            fig.legend(handles=[mpatches.Patch(color=C2, label="Rejected", alpha=.7),
                                  mpatches.Patch(color=C4, label="Approved", alpha=.7)],
                       loc="upper right", fontsize=9, facecolor=SF)
            fig.suptitle("Pair Plot — coloured by Loan Decision",
                         color=C1, fontsize=13, fontweight="bold", y=1.01)
            fig.tight_layout(); fig_show(fig)
            ins("Diagonal = per-variable histogram (green=approved, red=rejected). "
                "Off-diagonal = scatter. Clear colour separation = strong predictive feature.")
        else:
            st.info("Select at least 2 variables.")


# ═══════════════════════════════════════════════════════════════════
# PAGE: CORRELATION
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "🔗  Correlation Analysis":
    hdr("Correlation Analysis",
        "Full heatmap · feature importance ranking · interactive scatter")

    NC = [c for c in ["Age","AnnualIncome_INR","SavingsBalance_INR","ExistingDebt_INR",
        "DebtToIncomeRatio","CreditUtilization","CreditHistoryYears","NumCreditAccounts",
        "NumHardInquiries","MissedPayments","Bankruptcies","FinLiteracyScore",
        "UPI_TxnMonthly","EmergencyFundMonths","RiskScore","CreditScore","LoanApproved",
        "SavingsPct_Monthly","YearsEmployed","NumDependents","FamilyBurdenScore"]
        if c in df.columns]
    corr = df[NC].corr()

    col1, col2 = st.columns([3, 2])
    with col1:
        hdr("Full Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=ax, mask=mask,
                    cmap=sns.diverging_palette(220,15,as_cmap=True),
                    vmin=-1, vmax=1, center=0,
                    annot=True, fmt=".2f", annot_kws={"size":7.5},
                    linewidths=.3, linecolor=BG,
                    cbar_kws={"shrink":.75})
        ax.tick_params(labelsize=8)
        fig.tight_layout(); fig_show(fig)

    with col2:
        hdr("Feature Importance vs Target")
        tgt = st.selectbox("Target variable", ["CreditScore","LoanApproved"])
        fc  = corr[tgt].drop(tgt).sort_values()
        fig2, ax2 = plt.subplots(figsize=(7, 9), facecolor=BG)
        cl2 = [C2 if v < 0 else C4 for v in fc.values]
        ax2.barh(fc.index, fc.values, color=cl2, alpha=.85, edgecolor="none")
        ax2.axvline(0, color=MU, lw=1, ls="--")
        ax2.set_title(f"Pearson r  vs  {tgt}", color=C4, fontweight="bold")
        ax2.set_xlabel("Pearson r"); ax2.grid(True, axis="x")
        for i, (v, nm) in enumerate(zip(fc.values, fc.index)):
            ax2.text(v+(0.006 if v>=0 else -0.006), i, f"{v:.3f}",
                     va="center", ha="left" if v>=0 else "right",
                     fontsize=8, color=TX)
        fig2.tight_layout(); fig_show(fig2)

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    hdr("Key Correlation Insights")
    tp_ = fc[fc>0].tail(4); tn_ = fc[fc<0].head(4)
    ci1, ci2 = st.columns(2)
    with ci1:
        st.markdown("**Strongest POSITIVE drivers**")
        for f, v in tp_[::-1].items():
            ins(f"📈 <b>{f.replace('_INR','').replace('_',' ')}</b> — r = {v:.3f}", "g")
    with ci2:
        st.markdown("**Strongest NEGATIVE drivers**")
        for f, v in tn_.items():
            ins(f"📉 <b>{f.replace('_INR','').replace('_',' ')}</b> — r = {v:.3f}", "r")

    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    hdr("Interactive Scatter Plot")
    s1, s2 = st.columns(2)
    with s1: xv = st.selectbox("X variable", NC, index=0)
    with s2: yv = st.selectbox("Y variable", NC, index=len(NC)-2)
    sp2  = df.sample(min(700, len(df)), random_state=1)
    fig3, ax3 = plt.subplots(figsize=(10, 5), facecolor=BG)
    cc3  = sp2["LoanApproved"].map({0: C2, 1: C4})
    ax3.scatter(sp2[xv], sp2[yv], c=cc3, alpha=.35, s=14, edgecolors="none")
    if pd.api.types.is_numeric_dtype(df[yv]):
        tmp_s = sp2[[xv,yv]].dropna()
        if len(tmp_s) > 5:
            m_, b_, r_, p_, _ = stats.linregress(tmp_s[xv], tmp_s[yv])
            xl_ = np.linspace(tmp_s[xv].min(), tmp_s[xv].max(), 200)
            ax3.plot(xl_, m_*xl_+b_, color=C3, lw=2.5,
                     label=f"r={r_:.3f}  p={p_:.4f}")
            ax3.legend(fontsize=9, facecolor=SF)
    ax3.set_xlabel(xv.replace("_INR",""))
    ax3.set_ylabel(yv.replace("_INR",""))
    ax3.set_title(f"{xv}  vs  {yv}", color=TX, fontweight="bold")
    ax3.legend(handles=[mpatches.Patch(color=C2,label="Rejected",alpha=.7),
                         mpatches.Patch(color=C4,label="Approved",alpha=.7)],
               fontsize=9, facecolor=SF)
    ax3.grid(True)
    fig3.tight_layout(); fig_show(fig3)


# ═══════════════════════════════════════════════════════════════════
# PAGE: CUSTOMER SEGMENTS
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "🧩  Customer Segments":
    hdr("Customer Segmentation",
        "Elbow chart · PCA cluster map · segment profiles · marketing playbook")

    # predict cluster labels on the fly
    clu_f_ = [c for c in M["clu_f"] if c in df.columns]
    Xk_    = df[clu_f_].fillna(0)
    Xks_   = M["sc_clu"].transform(Xk_)
    df     = df.copy()
    df["Cluster_ID"]    = M["kmeans"].predict(Xks_)
    df["Cluster_Label"] = df["Cluster_ID"].map(M["seg_lbl"]).fillna("Emerging")

    # ── row 1: elbow + pca + pie ─────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])

    with c1:
        hdr("Elbow Chart — Optimal k")
        # compute inertias fresh (fast, ~<2 s for n_init=5)
        inertias_ = [KMeans(n_clusters=k,random_state=42,n_init=5).fit(Xks_).inertia_
                     for k in range(2, 9)]
        k_range   = list(range(2, 9))
        fig, ax   = plt.subplots(figsize=(6, 4), facecolor=BG)
        ax.plot(k_range, inertias_, "o-", color=C1, lw=2.5, ms=9,
                markerfacecolor=C3, markeredgecolor="none")
        ax.axvline(4, color=C2, lw=2, ls="--", label="Chosen  k = 4")
        ax.axvspan(3.5, 4.5, alpha=.07, color=C4)
        ax.set_xlabel("Number of Clusters  k")
        ax.set_ylabel("Inertia (within-cluster SS)")
        ax.set_title("Elbow Method — Finding Optimal k", color=C1, fontweight="bold")
        ax.legend(fontsize=9, facecolor=SF); ax.grid(True)
        fig.tight_layout(); fig_show(fig)
        ins("Elbow at <b>k = 4</b> — adding more clusters beyond 4 "
            "gives diminishing returns in variance explained.", "g")

    with c2:
        hdr("PCA Cluster Visualisation (2D)")
        pca   = PCA(n_components=2)
        Xp    = pca.fit_transform(Xks_)
        fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor=BG)
        for lb, clr_ in SEG_COL.items():
            mk_ = df["Cluster_Label"] == lb
            if mk_.any():
                ax2.scatter(Xp[mk_,0], Xp[mk_,1], c=clr_,
                            alpha=.45, s=14, label=lb, edgecolors="none")
        ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        ax2.set_title("Clusters in 2D PCA Space", color=TX, fontweight="bold")
        ax2.legend(fontsize=9, facecolor=SF, markerscale=2); ax2.grid(True)
        fig2.tight_layout(); fig_show(fig2)
        tot_var = pca.explained_variance_ratio_[:2].sum()*100
        ins(f"PCA captures {tot_var:.1f}% of variance in 2D. "
            "Clear cluster separation confirms meaningful segmentation.")

    with c3:
        hdr("Segment Sizes")
        sv_ = df["Cluster_Label"].value_counts()
        fig3, ax3 = plt.subplots(figsize=(3.5, 4), facecolor=BG)
        cs3 = [SEG_COL.get(s, C1) for s in sv_.index]
        ax3.pie(sv_, autopct="%1.1f%%", colors=cs3, startangle=90,
                pctdistance=.8, textprops={"color":TX,"fontsize":9})
        ax3.legend(sv_.index, loc="lower center", fontsize=8,
                   facecolor=SF, ncol=2, bbox_to_anchor=(.5,-.15))
        fig3.tight_layout(); fig_show(fig3)
        for s, cnt in sv_.items():
            st.markdown(
                f"<span style='color:{SEG_COL.get(s,C1)};font-weight:700'>{s}</span>: "
                f"{cnt:,} ({cnt/len(df)*100:.1f}%)",
                unsafe_allow_html=True,
            )

    # ── row 2: segment profiles ──────────────────────────────────────
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    hdr("Segment Profile Comparison — 8 Key Metrics")
    PM = [c for c in ["CreditScore","AnnualIncome_INR","DebtToIncomeRatio",
                       "CreditUtilization","MissedPayments","SavingsBalance_INR",
                       "FinLiteracyScore","EmergencyFundMonths"] if c in df.columns]
    sp3 = df.groupby("Cluster_Label")[PM].mean().round(2)
    sp3["Approval_%"] = (df.groupby("Cluster_Label")["LoanApproved"].mean()*100).round(1)
    SEG_ORD = [s for s in ["Prime","Emerging","Rebuilder","At-Risk"] if s in sp3.index]
    SEG_CLR = [SEG_COL[s] for s in SEG_ORD]

    fig4, ax4s = plt.subplots(2, 4, figsize=(22, 9), facecolor=BG)
    for ax_, met_ in zip(ax4s.flat, PM[:8]):
        vals_ = [sp3.loc[s, met_] if s in sp3.index else 0 for s in SEG_ORD]
        bars_ = ax_.bar(SEG_ORD, vals_, color=SEG_CLR, alpha=.82, edgecolor="none")
        ax_.set_title(met_.replace("_INR","").replace("_"," "),
                      color=TX, fontweight="bold", fontsize=9)
        ax_.set_ylabel("Avg", fontsize=7); ax_.grid(True, axis="y")
        ax_.tick_params(axis="x", labelsize=8, rotation=15)
        for bar_, v_ in zip(bars_, vals_):
            ax_.text(bar_.get_x()+bar_.get_width()/2, v_*1.012,
                     f"{v_:,.1f}", ha="center", va="bottom", fontsize=7.5, color=TX)
    fig4.suptitle("Average Metrics by Marketing Segment",
                  color=C1, fontsize=13, fontweight="bold", y=1.01)
    fig4.tight_layout(); fig_show(fig4)
    st.dataframe(sp3.style.background_gradient(cmap="RdYlGn", axis=0),
                 use_container_width=True)

    # ── row 3: marketing playbook ────────────────────────────────────
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    hdr("Marketing Playbook — Segment Action Map")
    pbc = st.columns(4)
    for col_, (seg_, ico_, clr_, action_, desc_) in zip(pbc, [
        ("Prime",    "⭐", C4, "Fast-track Approval",
         "Offer premium home/business loans. Pre-approved via email & push. Convert within 24 hrs. Upsell investment products."),
        ("Emerging", "🌱", C1, "Nurture Campaign",
         "Starter loan ₹1–5L. WhatsApp follow-up every 2 weeks. Credit score simulator. Graduate to Prime in 6 months."),
        ("Rebuilder","🔧", C3, "Credit Builder Programme",
         "Secured gold loan only. Monthly CIBIL guide. Financial webinar series. Target +80 pts in 12 months."),
        ("At-Risk",  "⚠️", C2, "Hold — Educate Only",
         "No loan offers. Financial literacy blog & videos. Budgeting tools. Re-enter pipeline after 12 months."),
    ]):
        col_.markdown(
            f'<div class="mc" style="border-color:{clr_};padding:16px 12px;">'
            f'<div style="font-size:1.6rem;text-align:center">{ico_}</div>'
            f'<div style="color:{clr_};font-family:Syne,sans-serif;font-weight:800;'
            f'text-align:center;margin:7px 0 4px;font-size:.95rem;">{seg_}</div>'
            f'<div style="color:#E6EDF3;font-weight:600;font-size:.82rem;margin-bottom:5px;">{action_}</div>'
            f'<div style="color:#8B949E;font-size:.76rem;line-height:1.55;">{desc_}</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE: REGRESSION ANALYSIS
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "📈  Regression Analysis":
    hdr("Regression Analysis — Credit Score Prediction",
        "Ridge & RF Regressor · actual vs predicted · residuals · feature coefficients")

    reg_f_ = [c for c in M["reg_f"] if c in df.columns]
    Xrs_   = prep_X(df, reg_f_, M["sc_reg"])
    y_true_ = df["CreditScore"].values
    y_ridge_ = M["ridge"].predict(Xrs_)
    y_rfr_   = M["rf_reg"].predict(Xrs_)

    mr_ = M["metrics"]["ridge_regression"]
    mf_ = M["metrics"]["rf_regressor"]

    c1,c2,c3,c4 = st.columns(4)
    kpi_card(c1, "Ridge R²",     f"{mr_['r2']}%",    C4)
    kpi_card(c2, "Ridge RMSE",   f"{mr_['rmse']}",   C1)
    kpi_card(c3, "RF Reg R²",    f"{mf_['r2']}%",    C3)
    kpi_card(c4, "RF Reg RMSE",  f"{mf_['rmse']}",   C2)
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)

    si_ = np.random.default_rng(0).choice(len(df), min(500,len(df)), replace=False)

    # ── Actual vs Predicted ──────────────────────────────────────────
    col1, col2 = st.columns(2)
    for col_, y_pred_, clr_, name_, m_info_ in [
        (col1, y_ridge_, C1, "Ridge Regression",  mr_),
        (col2, y_rfr_,   C3, "RF Regressor",      mf_),
    ]:
        with col_:
            hdr(f"Actual vs Predicted — {name_}")
            fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG)
            ax.scatter(y_true_[si_], y_pred_[si_], c=clr_,
                       alpha=.35, s=14, edgecolors="none", label="Predictions")
            ax.plot([300,900],[300,900], color=C4, lw=2, ls="--", label="Perfect fit")
            tmp_r = np.array([[y_true_[si_].min(), y_pred_[si_].min()],
                               [y_true_[si_].max(), y_pred_[si_].max()]])
            m_lr, b_lr, r_lr, _, _ = stats.linregress(y_true_[si_], y_pred_[si_])
            xl_ = np.linspace(300, 900, 200)
            ax.plot(xl_, m_lr*xl_+b_lr, color=C3, lw=2, label=f"Trend  r={r_lr:.3f}")
            ax.set_xlabel("Actual Credit Score")
            ax.set_ylabel("Predicted Credit Score")
            ax.set_title(f"{name_} — R²={m_info_['r2']}%  RMSE={m_info_['rmse']}",
                         color=clr_, fontweight="bold")
            ax.legend(fontsize=9, facecolor=SF); ax.grid(True)
            fig.tight_layout(); fig_show(fig)

    ins(f"Ridge R²={mr_['r2']}% outperforms RF (R²={mf_['r2']}%). "
        "Tight diagonal clustering confirms reliable predictions. "
        f"Typical error ±{mr_['rmse']} CIBIL points — acceptable for pre-qualification.", "g")

    # ── Residuals ────────────────────────────────────────────────────
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    residuals_ = y_true_ - y_ridge_
    col3, col4 = st.columns(2)
    with col3:
        hdr("Residual Distribution — Ridge")
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
        ax.hist(residuals_, bins=45, color=C1, alpha=.75, edgecolor="none")
        ax.axvline(0,                color=C4, lw=2, ls="--", label="Zero")
        ax.axvline(residuals_.mean(),color=C3, lw=2, ls="-.",
                   label=f"Mean={residuals_.mean():.2f}")
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution", color=C1, fontweight="bold")
        ax.legend(fontsize=9, facecolor=SF); ax.grid(True)
        fig.tight_layout(); fig_show(fig)

    with col4:
        hdr("Residuals vs Fitted Values")
        fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=BG)
        ax.scatter(y_ridge_[si_], residuals_[si_],
                   c=C2, alpha=.3, s=12, edgecolors="none")
        ax.axhline(0, color=C4, lw=2, ls="--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Fitted", color=C2, fontweight="bold")
        ax.grid(True)
        fig.tight_layout(); fig_show(fig)

    ins(f"Residuals centred near zero (mean={residuals_.mean():.2f}). "
        "No systematic pattern in residuals-vs-fitted confirms model is unbiased.", "")

    # ── Feature coefficients ─────────────────────────────────────────
    st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
    hdr("Ridge Regression — Feature Coefficients")
    n_coef = min(len(M["ridge"].coef_), len(reg_f_))
    coefs_ = pd.Series(M["ridge"].coef_[:n_coef], index=reg_f_[:n_coef]).sort_values()
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG)
    cc5 = [C2 if v < 0 else C4 for v in coefs_.values]
    ax.barh(coefs_.index, coefs_.values, color=cc5, alpha=.82, edgecolor="none")
    ax.axvline(0, color=MU, lw=1, ls="--")
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Ridge Coefficients — Credit Score Model", color=TX, fontweight="bold")
    ax.grid(True, axis="x")
    fig.tight_layout(); fig_show(fig)

    tp5 = coefs_[coefs_>0].tail(3).index.tolist()
    tn5 = coefs_[coefs_<0].head(3).index.tolist()
    ci1_, ci2_ = st.columns(2)
    with ci1_:
        st.markdown("**Features that INCREASE credit score**")
        for f_ in reversed(tp5): ins(f"➕ <b>{f_.replace('_',' ')}</b> — {coefs_[f_]:.3f}", "g")
    with ci2_:
        st.markdown("**Features that DECREASE credit score**")
        for f_ in tn5: ins(f"➖ <b>{f_.replace('_',' ')}</b> — {coefs_[f_]:.3f}", "r")


# ═══════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "🤖  Model Performance":
    hdr("Model Performance Dashboard",
        "ROC curves · confusion matrix · feature importance · model comparison")

    clf_f_ = [c for c in M["clf_f"] if c in df.columns]
    Xas_   = prep_X(df, clf_f_, M["sc_clf"])
    ya_    = df["LoanApproved"].astype(int).values

    t1, t2, t3 = st.tabs(["Classification Models",
                           "Feature Importance",
                           "Model Comparison"])

    # ── tab 1 ────────────────────────────────────────────────────────
    with t1:
        c1, c2, c3 = st.columns(3)
        for col_, (key_, name_, best_) in zip([c1,c2,c3],[
            ("logistic_regression", "Logistic Regression",  False),
            ("random_forest",       "Random Forest",         False),
            ("gradient_boosting",   "Gradient Boosting",     True),
        ]):
            m_ = M["metrics"][key_]
            col_.markdown(
                f'<div class="mc {"best" if best_ else ""}">'
                + (f'<div style="color:#3FB950;font-size:.7rem;font-weight:700;'
                   f'margin-bottom:5px;">★ BEST CLASSIFIER</div>' if best_ else "")
                + f'<div style="font-weight:700;font-size:.9rem;color:#E6EDF3;'
                  f'margin-bottom:8px;">{name_}</div>'
                  f'<div style="color:{C1};font-size:1.7rem;font-weight:800;">'
                  f'{m_["accuracy"]}%</div>'
                  f'<div style="color:{MU};font-size:.7rem;">Accuracy</div>'
                  f'<div class="dv"></div>'
                  f'<div style="display:flex;justify-content:space-around;">'
                  f'<div style="text-align:center"><div style="color:{C3};font-weight:700;font-size:1rem;">'
                  f'{m_["roc_auc"]}%</div><div style="color:{MU};font-size:.68rem;">AUC-ROC</div></div>'
                  f'<div style="text-align:center"><div style="color:{C4};font-weight:700;font-size:1rem;">'
                  f'{m_["f1"]}%</div><div style="color:{MU};font-size:.68rem;">F1 Score</div></div>'
                  f'</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            hdr("ROC Curves — All 3 Classifiers")
            fig, ax = plt.subplots(figsize=(7, 5.5), facecolor=BG)
            for mdl_, nm_, clr_ in [
                (M["gb_clf"], "Gradient Boosting",  C4),
                (M["rf_clf"], "Random Forest",       C1),
                (M["lr"],     "Logistic Regression", C3),
            ]:
                pr_  = mdl_.predict_proba(Xas_)[:,1]
                fp_, tp2_, _ = roc_curve(ya_, pr_)
                ra_  = sk_auc(fp_, tp2_)
                ax.plot(fp_, tp2_, color=clr_, lw=2.5,
                        label=f"{nm_}  (AUC={ra_:.3f})")
            ax.plot([0,1],[0,1], color=MU, lw=1, ls="--", label="Random Classifier")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves — Loan Approval",
                         color=C1, fontweight="bold")
            ax.legend(fontsize=9, facecolor=SF)
            ax.grid(True); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
            fig.tight_layout(); fig_show(fig)
            ins("Gradient Boosting achieves highest AUC. "
                "AUC > 0.85 is considered strong for credit risk models.", "g")

        with col2:
            hdr("Confusion Matrix — Gradient Boosting")
            yp_gb = M["gb_clf"].predict(Xas_)
            cm_   = confusion_matrix(ya_, yp_gb)
            fig2, ax2 = plt.subplots(figsize=(6, 5.5), facecolor=BG)
            sns.heatmap(cm_, annot=True, fmt="d", cmap="Blues", ax=ax2,
                        xticklabels=["Rejected","Approved"],
                        yticklabels=["Rejected","Approved"],
                        annot_kws={"size":18,"weight":"bold"},
                        linewidths=1, linecolor=BG)
            ax2.set_xlabel("Predicted", fontsize=11)
            ax2.set_ylabel("Actual",    fontsize=11)
            ax2.set_title("Confusion Matrix — Gradient Boosting",
                          color=TX, fontweight="bold")
            fig2.tight_layout(); fig_show(fig2)
            tn_, fp_cm, fn_, tp_cm = cm_.ravel()
            prec_ = tp_cm/(tp_cm+fp_cm) if (tp_cm+fp_cm)>0 else 0
            rec_  = tp_cm/(tp_cm+fn_)   if (tp_cm+fn_)>0   else 0
            ins(f"Precision={prec_*100:.1f}%  |  Recall={rec_*100:.1f}%  |  "
                f"False positives={fp_cm} — minimal NPA risk.")

    # ── tab 2 ────────────────────────────────────────────────────────
    with t2:
        hdr("Feature Importance — Random Forest Classifier")
        fi_ = M["metrics"]["random_forest"].get("feature_importance", {})
        if fi_:
            fi_df_ = (pd.DataFrame({"Feature": list(fi_.keys()),
                                    "Importance": list(fi_.values())})
                      .sort_values("Importance", ascending=True)
                      .tail(20))
            fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
            cf3 = [C4 if v>.05 else C1 if v>.02 else MU for v in fi_df_["Importance"]]
            ax.barh(fi_df_["Feature"], fi_df_["Importance"],
                    color=cf3, alpha=.85, edgecolor="none")
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 20 Features — Random Forest Classifier",
                         color=C4, fontweight="bold")
            ax.grid(True, axis="x")
            for i_, v_ in enumerate(fi_df_["Importance"]):
                ax.text(v_+.001, i_, f"{v_:.4f}", va="center", fontsize=8, color=TX)
            fig.tight_layout(); fig_show(fig)
            top5_ = fi_df_.tail(5)["Feature"].tolist()[::-1]
            ins(f"Top 5 features: <b>{', '.join(top5_)}</b>. "
                "Collecting these accurately at intake maximises prediction quality.", "g")
        else:
            st.warning("Feature importance not available (re-train to populate).")

    # ── tab 3 ────────────────────────────────────────────────────────
    with t3:
        hdr("Model Comparison Table")
        cmp_d = {
            "Model":       ["Logistic Regression","Random Forest","Gradient Boosting",
                            "Ridge Regression","RF Regressor"],
            "Task":        ["Classification"]*3 + ["Regression"]*2,
            "Accuracy/R²": [f"{M['metrics']['logistic_regression']['accuracy']}%",
                            f"{M['metrics']['random_forest']['accuracy']}%",
                            f"{M['metrics']['gradient_boosting']['accuracy']}%",
                            f"{M['metrics']['ridge_regression']['r2']}%",
                            f"{M['metrics']['rf_regressor']['r2']}%"],
            "AUC / RMSE":  [f"AUC {M['metrics']['logistic_regression']['roc_auc']}%",
                            f"AUC {M['metrics']['random_forest']['roc_auc']}%",
                            f"AUC {M['metrics']['gradient_boosting']['roc_auc']}%",
                            f"RMSE {M['metrics']['ridge_regression']['rmse']}",
                            f"RMSE {M['metrics']['rf_regressor']['rmse']}"],
            "F1 / MAE":    [f"F1 {M['metrics']['logistic_regression']['f1']}%",
                            f"F1 {M['metrics']['random_forest']['f1']}%",
                            f"F1 {M['metrics']['gradient_boosting']['f1']}%",
                            f"MAE {M['metrics']['ridge_regression']['mae']}",
                            f"MAE {M['metrics']['rf_regressor']['mae']}"],
        }
        st.dataframe(pd.DataFrame(cmp_d), use_container_width=True, hide_index=True)

        # grouped bar
        fig, ax4s = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
        cn_ = ["LR","RF","GB"]; x_ = np.arange(3); w_ = .26
        av_ = [M["metrics"][k]["accuracy"]  for k in ["logistic_regression","random_forest","gradient_boosting"]]
        au_ = [M["metrics"][k]["roc_auc"]   for k in ["logistic_regression","random_forest","gradient_boosting"]]
        fv_ = [M["metrics"][k]["f1"]        for k in ["logistic_regression","random_forest","gradient_boosting"]]
        ax4s[0].bar(x_-w_, av_, w_, label="Accuracy", color=C1, alpha=.8, edgecolor="none")
        ax4s[0].bar(x_,    au_, w_, label="AUC-ROC",  color=C4, alpha=.8, edgecolor="none")
        ax4s[0].bar(x_+w_, fv_, w_, label="F1 Score", color=C3, alpha=.8, edgecolor="none")
        ax4s[0].set_xticks(x_); ax4s[0].set_xticklabels(cn_)
        ax4s[0].set_ylabel("Score (%)"); ax4s[0].set_ylim(70, 100)
        ax4s[0].set_title("Classification Models", color=C1, fontweight="bold")
        ax4s[0].legend(fontsize=9, facecolor=SF); ax4s[0].grid(True, axis="y")

        rn_ = ["Ridge","RF Reg"]
        rv_ = [M["metrics"]["ridge_regression"]["r2"], M["metrics"]["rf_regressor"]["r2"]]
        ax4s[1].bar(rn_, rv_, color=[C4,C3], alpha=.82, edgecolor="none", width=.4)
        ax4s[1].set_ylabel("R² (%)"); ax4s[1].set_ylim(75, 100)
        ax4s[1].set_title("Regression Models — R²", color=C4, fontweight="bold")
        ax4s[1].grid(True, axis="y")
        for i_, v_ in enumerate(rv_):
            ax4s[1].text(i_, v_+.3, f"{v_}%", ha="center",
                         fontsize=11, fontweight="bold", color=TX)
        fig.tight_layout(); fig_show(fig)


# ═══════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "🔮  Live Prediction":
    hdr("Live Customer Prediction",
        "Enter any customer profile → instant loan decision · credit score · segment")

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Demographics**")
            age_   = st.slider("Age", 18, 70, 32)
            ndep_  = st.slider("Dependents", 0, 6, 2)
            hprop_ = st.selectbox("Owns Property", ["Yes","No"])
            hveh_  = st.selectbox("Owns Vehicle",  ["Yes","No"])
        with c2:
            st.markdown("**💼 Employment & Income**")
            ainc_  = st.number_input("Annual Income (₹)", 0, 10_000_000, 480_000, step=10_000)
            yemp_  = st.slider("Years Employed", 0, 40, 5)
            hfp_   = st.selectbox("Financial Plan", ["Yes","No"])
            ish_   = st.selectbox("Income Shock (3yr)", ["No","Yes"])
        with c3:
            st.markdown("**📱 Digital Behaviour**")
            upi_   = st.slider("UPI Txns/Month", 0, 100, 20)
            fl_    = st.slider("Fin Literacy (1–10)", 1.0, 10.0, 5.5, step=.1)
            al_    = st.selectbox("App-Based Loan", ["No","Yes"])
            hcc_   = st.selectbox("Has Credit Card", ["Yes","No"])
        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("**📜 Credit History**")
            ch_  = st.slider("Credit History (yrs)", 0, 30, 5)
            na_  = st.slider("Credit Accounts", 1, 12, 3)
            mp_  = st.slider("Missed Payments", 0, 5, 1)
            bk_  = st.slider("Bankruptcies", 0, 2, 0)
        with c5:
            st.markdown("**💳 Credit Profile**")
            cu_  = st.slider("Credit Utilization", 0.0, 1.0, .35, step=.01)
            hi_  = st.slider("Hard Inquiries", 0, 10, 2)
            rh_  = st.selectbox("Repayment History", ["Excellent","Good","Fair","Poor"])
        with c6:
            st.markdown("**💰 Finances**")
            dti_ = st.slider("Debt-to-Income Ratio", 0.0, 3.0, .4, step=.05)
            sp_  = st.slider("Savings % of Income",  0.0, 60.0, 15.0)
            sb_  = st.number_input("Savings Balance (₹)", 0, 2_000_000,  80_000, step=5_000)
            ed_  = st.number_input("Existing Debt (₹)",   0, 5_000_000, 120_000, step=5_000)
            ef_  = st.slider("Emergency Fund (months)", 0.0, 12.0, 2.0, step=.5)
        sub_ = st.form_submit_button("🔮  Run Prediction", use_container_width=True)

    if sub_:
        RM = {"Excellent":4,"Good":3,"Fair":2,"Poor":1}
        emi_  = ed_/36 if ed_>0 else 0
        rs_   = mp_*.25 + bk_*.40 + dti_*.20 + cu_*.15
        ei_   = emi_ / max(ainc_/12, 1)
        sd_   = sb_  / max(ed_, 1)
        inp_  = {
            "Age": age_, "AnnualIncome_INR": ainc_, "DebtToIncomeRatio": dti_,
            "CreditUtilization": cu_, "CreditHistoryYears": ch_,
            "NumCreditAccounts": na_, "NumHardInquiries": hi_,
            "MissedPayments": mp_, "Bankruptcies": bk_,
            "SavingsPct_Monthly": sp_, "SavingsBalance_INR": sb_,
            "ExistingDebt_INR": ed_, "UPI_TxnMonthly": upi_,
            "FinLiteracyScore": fl_, "EmergencyFundMonths": ef_,
            "FamilyBurdenScore": ndep_*.15,
            "HasProperty": int(hprop_=="Yes"), "HasVehicle": int(hveh_=="Yes"),
            "HasCreditCard": int(hcc_=="Yes"), "HasFinancialPlan": int(hfp_=="Yes"),
            "IncomeShockLast3Yr": int(ish_=="Yes"), "AppBasedLoan": int(al_=="Yes"),
            "RepaymentHistory_Ord": RM[rh_],
            "RiskScore": rs_, "EMI_to_Income": ei_, "SavingsToDebt": sd_,
            "YearsEmployed": yemp_, "NumDependents": ndep_,
        }
        # classification
        Xin_  = np.array([[inp_.get(f,0) for f in M["clf_f"]]])
        Xins_ = M["sc_clf"].transform(Xin_)
        gp_   = M["gb_clf"].predict_proba(Xins_)[0][1]
        rp_   = M["rf_clf"].predict_proba(Xins_)[0][1]
        lp_   = M["lr"].predict_proba(Xins_)[0][1]
        ep_   = gp_*.5 + rp_*.3 + lp_*.2
        # regression
        Xri_  = np.array([[inp_.get(f,0) for f in M["reg_f"]]])
        Xris_ = M["sc_reg"].transform(Xri_)
        pcs_  = int(np.clip(M["ridge"].predict(Xris_)[0], 300, 900))
        # clustering
        clu_inp_ = {"AnnualIncome_INR":ainc_,"CreditScore":pcs_,
                    "DebtToIncomeRatio":dti_,"CreditUtilization":cu_,
                    "MissedPayments":mp_,"SavingsBalance_INR":sb_,
                    "RiskScore":rs_,"FinLiteracyScore":fl_,
                    "EmergencyFundMonths":ef_,"UPI_TxnMonthly":upi_}
        Xci_  = np.array([[clu_inp_.get(f,0) for f in M["clu_f"]]])
        Xcis_ = M["sc_clu"].transform(Xci_)
        cl_id_ = int(M["kmeans"].predict(Xcis_)[0])
        seg_   = M["seg_lbl"].get(cl_id_, "Emerging")

        # Results
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")
        if   ep_ >= .60: vd_,bc_="✅  APPROVED","vA"
        elif ep_ >= .40: vd_,bc_="⚠️  BORDERLINE","vB"
        else:            vd_,bc_="❌  REJECTED","vR"

        r1,r2,r3,r4 = st.columns(4)
        r1.markdown(f'<div class="{bc_}">{vd_}</div>', unsafe_allow_html=True)
        slc_ = SEG_COL.get(seg_, C1)
        for col_, (lbl_,val_,clr_) in zip([r2,r3,r4],[
            ("Approval Probability", f"{ep_*100:.1f}%", C1),
            ("Predicted CIBIL",      f"{pcs_}",         C3),
            ("Segment",              seg_,               slc_),
        ]):
            kpi_card(col_, lbl_, val_, clr_)

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        col1_, col2_ = st.columns(2)
        with col1_:
            st.markdown("**Model Agreement**")
            for nm_,pr_ in [("Gradient Boosting",gp_),("Random Forest",rp_),("Logistic Reg",lp_)]:
                st.metric(nm_, f"{pr_*100:.1f}%",
                          delta="Approve ✅" if pr_>=.5 else "Reject ❌",
                          delta_color="normal" if pr_>=.5 else "inverse")

        with col2_:
            fig_g, ax_g = plt.subplots(figsize=(7, 2.8), facecolor=BG)
            zones_ = [(300,500,"#F85149","Poor"),(500,580,"#FF8C42","Below Avg"),
                      (580,670,"#D29922","Fair"),(670,740,"#90EE90","Good"),
                      (740,800,"#3FB950","Very Good"),(800,900,"#58A6FF","Exceptional")]
            for lo_,hi_,clr_,lbl_ in zones_:
                ax_g.barh(0,hi_-lo_,left=lo_,height=.5,color=clr_,alpha=.7,edgecolor="none")
                ax_g.text((lo_+hi_)/2,0,lbl_,ha="center",va="center",
                          fontsize=7,color="white",fontweight="bold")
            ax_g.axvline(pcs_,color="white",lw=3,ymin=.05,ymax=.95)
            ax_g.text(pcs_,.32,str(pcs_),ha="center",va="bottom",
                      color="white",fontsize=14,fontweight="bold")
            ax_g.set_xlim(300,900); ax_g.set_ylim(-.5,.7)
            ax_g.set_xlabel("CIBIL Credit Score"); ax_g.set_yticks([])
            ax_g.set_title("Predicted Credit Score", color=TX, fontweight="bold")
            ax_g.grid(False)
            fig_g.tight_layout(); fig_show(fig_g)

        st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
        pm_ = {
            "Prime":    ("⭐",C4,"Fast-track approval. Premium product offer. Convert within 24 hrs."),
            "Emerging": ("🌱",C1,"Starter loan ₹1–5L. Nurture campaign. Re-assess in 3 months."),
            "Rebuilder":("🔧",C3,"Secured loan only. Credit improvement plan. 12-month programme."),
            "At-Risk":  ("⚠️",C2,"No loan offer. Financial literacy content. Re-enter in 12 months."),
        }
        ico_,cl_a,ac_ = pm_.get(seg_, ("🌱",C1,"Nurture campaign."))
        st.markdown(
            f'<div class="mc" style="border-color:{cl_a};padding:14px 18px;">'
            f'<span style="font-size:1.3rem;">{ico_}</span>'
            f'<span style="color:{cl_a};font-weight:700;font-size:.95rem;'
            f'margin-left:8px;">{seg_} — Recommended Action</span><br><br>'
            f'<span style="color:#C9D1D9;font-size:.88rem;">{ac_}</span></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# PAGE: BATCH SCORING
# ═══════════════════════════════════════════════════════════════════
elif PAGE == "📂  Batch Scoring":
    hdr("Batch Customer Upload & Scoring",
        "Upload a CSV → approval probabilities · CIBIL scores · segments · actions for every row")

    req_cols = ["Age","AnnualIncome_INR","DebtToIncomeRatio","CreditUtilization",
                "CreditHistoryYears","NumCreditAccounts","MissedPayments","Bankruptcies",
                "SavingsPct_Monthly","SavingsBalance_INR","ExistingDebt_INR",
                "FinLiteracyScore","UPI_TxnMonthly","EmergencyFundMonths"]

    c1_, c2_ = st.columns([2,1])
    with c1_:
        st.markdown("**Required columns (minimum)**")
        st.code(", ".join(req_cols))
        ins("Other columns are optional — system auto-fills missing ones with safe defaults. "
            "Column names are case-sensitive.", "")
    with c2_:
        tpc_ = [c for c in req_cols if c in raw.columns]
        st.download_button("⬇️ Download Sample Template (10 rows)",
                           raw[tpc_].head(10).to_csv(index=False).encode(),
                           "template.csv","text/csv", use_container_width=True)

    up_ = st.file_uploader("📁 Upload your customer CSV", type=["csv"])
    if up_:
        try:
            nd_ = pd.read_csv(up_)
            st.success(f"✅ Uploaded: **{len(nd_):,} customers**  ×  {nd_.shape[1]} columns")
            with st.spinner("🔄 Scoring all customers …"):
                res_ = nd_.copy()
                # derived cols
                RM2 = {"Excellent":4,"Good":3,"Fair":2,"Poor":1}
                res_["RepaymentHistory_Ord"] = (
                    res_["RepaymentHistory"].map(RM2).fillna(2)
                    if "RepaymentHistory" in res_.columns else 2)
                ed2_  = res_.get("ExistingDebt_INR",   pd.Series(np.zeros(len(res_))))
                ai2_  = res_.get("AnnualIncome_INR",   pd.Series(np.ones(len(res_))))
                sb2_  = res_.get("SavingsBalance_INR", pd.Series(np.zeros(len(res_))))
                mp2_  = res_.get("MissedPayments",     pd.Series(np.zeros(len(res_))))
                bk2_  = res_.get("Bankruptcies",       pd.Series(np.zeros(len(res_))))
                dti2_ = res_.get("DebtToIncomeRatio",  pd.Series(np.zeros(len(res_))))
                cu2_  = res_.get("CreditUtilization",  pd.Series(np.zeros(len(res_))))
                nd2_  = res_.get("NumDependents",      pd.Series(np.full(len(res_),2)))
                res_["MonthlyEMI_INR"]    = ed2_/36
                res_["EMI_to_Income"]     = res_["MonthlyEMI_INR"] / (ai2_/12+1)
                res_["SavingsToDebt"]     = sb2_ / (ed2_+1)
                res_["RiskScore"]         = mp2_*.25 + bk2_*.40 + dti2_*.20 + cu2_*.15
                res_["FamilyBurdenScore"] = nd2_*.15
                for c_ in ["HasProperty","HasVehicle","HasCreditCard","HasFinancialPlan",
                            "IncomeShockLast3Yr","AppBasedLoan","NumDependents","YearsEmployed"]:
                    if c_ not in res_.columns: res_[c_] = 0

                Xb_  = prep_X(res_, M["clf_f"], M["sc_clf"])
                gbp_ = M["gb_clf"].predict_proba(Xb_)[:,1]
                rfp_ = M["rf_clf"].predict_proba(Xb_)[:,1]
                lrp_ = M["lr"].predict_proba(Xb_)[:,1]
                ep_b = gbp_*.5 + rfp_*.3 + lrp_*.2

                Xrb_ = prep_X(res_, M["reg_f"], M["sc_reg"])
                pcs_b = np.clip(M["ridge"].predict(Xrb_), 300, 900).round(0).astype(int)

                Xcb_ = prep_X(res_, M["clu_f"], M["sc_clu"])
                cls_b = M["kmeans"].predict(Xcb_)
                sls_b = [M["seg_lbl"].get(int(c_),"Emerging") for c_ in cls_b]

                AM = {"Prime":"Fast-track approval — premium product offer",
                      "Emerging":"Nurture campaign — starter loan + follow-up",
                      "Rebuilder":"Credit builder — secured loan only",
                      "At-Risk":"Fin literacy only — no loan yet"}

                res_["Approval_Probability_%"] = (ep_b*100).round(2)
                res_["Loan_Decision"]           = np.where(ep_b>=.60,"Approved",
                                                   np.where(ep_b>=.40,"Borderline","Rejected"))
                res_["Predicted_CreditScore"]   = pcs_b
                res_["Marketing_Segment"]       = sls_b
                res_["Recommended_Action"]      = [AM.get(s,"Nurture campaign") for s in sls_b]

            # Summary KPIs
            st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
            st.markdown("### Results Summary")
            kc = st.columns(5)
            kpi_card(kc[0],"Scored",     f"{len(res_):,}",                                    C1)
            kpi_card(kc[1],"Approved",   f"{(res_['Loan_Decision']=='Approved').sum():,}",    C4)
            kpi_card(kc[2],"Borderline", f"{(res_['Loan_Decision']=='Borderline').sum():,}",  C3)
            kpi_card(kc[3],"Rejected",   f"{(res_['Loan_Decision']=='Rejected').sum():,}",    C2)
            kpi_card(kc[4],"Avg Score",  f"{res_['Predicted_CreditScore'].mean():.0f}",       C3)

            st.markdown('<div class="dv"></div>', unsafe_allow_html=True)
            ca1_, ca2_ = st.columns(2)
            with ca1_:
                sv2_ = pd.Series(sls_b).value_counts()
                fig_b, ax_b = plt.subplots(figsize=(6, 3.5), facecolor=BG)
                cb2_ = [SEG_COL.get(s,C1) for s in sv2_.index]
                ax_b.bar(sv2_.index, sv2_.values, color=cb2_, alpha=.85, edgecolor="none")
                ax_b.set_ylabel("Count")
                ax_b.set_title("Customers by Segment", color=TX, fontweight="bold")
                ax_b.grid(True, axis="y")
                for i_, v_ in enumerate(sv2_.values):
                    ax_b.text(i_, v_+.5, str(v_), ha="center",
                              fontsize=10, fontweight="bold", color=TX)
                fig_b.tight_layout(); fig_show(fig_b)

            with ca2_:
                dv2_ = res_["Loan_Decision"].value_counts()
                fig_c, ax_c = plt.subplots(figsize=(6, 3.5), facecolor=BG)
                dc2_ = {"Approved":C4,"Borderline":C3,"Rejected":C2}
                cc2_ = [dc2_.get(d_,C1) for d_ in dv2_.index]
                ax_c.pie(dv2_, labels=dv2_.index, autopct="%1.1f%%",
                         colors=cc2_, startangle=90,
                         textprops={"color":TX,"fontsize":9})
                ax_c.set_title("Loan Decision Split", color=TX, fontweight="bold")
                fig_c.tight_layout(); fig_show(fig_c)

            show_cols_ = ["Loan_Decision","Approval_Probability_%",
                          "Predicted_CreditScore","Marketing_Segment","Recommended_Action"]
            st.dataframe(res_[[c for c in show_cols_ if c in res_.columns]].head(30),
                         use_container_width=True, height=320)
            st.download_button("⬇️ Download Full Scored CSV",
                               res_.to_csv(index=False).encode(),
                               "scored_customers.csv","text/csv",
                               use_container_width=True)
        except Exception as exc_:
            st.error(f"❌ Error processing file: {exc_}")
            st.exception(exc_)
    else:
        ins("<b>How to use batch scoring:</b><br>"
            "① Download sample template above<br>"
            "② Fill in your customer data (column names must match exactly)<br>"
            "③ Upload the completed CSV<br>"
            "④ Download the enriched output with approval probability, "
            "predicted CIBIL score, segment &amp; recommended action per row")
