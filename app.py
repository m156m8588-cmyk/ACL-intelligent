import io, os
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="كاشف الاحتيال الذكي", layout="wide")
st.title("🔍 نظام كشف الاحتيال الذكي - ACL")

ALIASES = {
    "date": ["date", "تاريخ"],
    "amount": ["amount", "total", "value", "مبلغ", "قيمة", "اجمالي", "إجمالي"],
    "vendor": ["vendor", "supplier", "name", "مورد", "عميل", "اسم"],
    "invoice": ["invoice", "ref", "فاتورة", "رقم"],
}
THRESHOLDS = [10000, 50000]
WEEKEND = {4, 5}

def read_file(name, data):
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(data))
    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(data))
    elif ext == "pdf":
        rows = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                for t in page.extract_tables() or []:
                    rows += t
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        st.error("الصيغة المدعومة: xlsx, xls, csv, pdf")
        return None
    mapping = {}
    for col in df.columns:
        c = str(col).strip().lower()
        for key, words in ALIASES.items():
            if key not in mapping.values() and any(w in c for w in words):
                mapping[col] = key
                break
    df = df.rename(columns=mapping)
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["amount"])
    df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
    return df

def analyze(df):
    n = len(df)
    score = np.zeros(n)
    why = [[] for _ in range(n)]
    def flag(mask, pts, text):
        for i in np.where(mask)[0]:
            score[i] += pts
            why[i].append(text)
    a = df["amount"].values
    if "invoice" in df.columns:
        flag(df.duplicated(["vendor", "invoice"], keep=False).values, 40, "فاتورة مكررة")
    flag(df.duplicated(["vendor", "amount", "date"], keep=False).values, 30, "نفس المورد والمبلغ والتاريخ")
    flag((a >= 1000) & (a % 1000 == 0), 15, "مبلغ مستدير مشبوه")
    for t in THRESHOLDS:
        flag((a >= t * 0.95) & (a < t), 25, f"أقل من حد الموافقة {t}")
    flag(df["date_parsed"].dt.weekday.isin(WEEKEND).values, 10, "عملية في عطلة")
    flag(a <= 0, 20, "مبلغ صفر أو سالب")
    if n >= 20:
        X = np.log1p(np.abs(a)).reshape(-1, 1)
        pred = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
        flag(pred == -1, 30, "قيمة شاذة (AI)")
    df["score"] = np.minimum(score, 100)
    df["reasons"] = ["، ".join(r) for r in why]
    return df

file = st.file_uploader("ارفعي ملف Excel / CSV / PDF", type=["xlsx","xls","csv","pdf"])
if file:
    df = read_file(file.name, file.read())
    if df is not None:
        result = analyze(df)
        st.success(f"تم تحليل {len(result)} صف، المشبوه: {int((result['score']>=40).sum())}")
        st.dataframe(result.sort_values("score", ascending=False))
    
