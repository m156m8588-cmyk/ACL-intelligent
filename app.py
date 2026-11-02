import io, os, math
from datetime import datetime
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

st.set_page_config(page_title="كاشف الاحتيال المالي", page_icon="🔍", layout="wide")

DB_URL = os.getenv("DATABASE_URL", "sqlite:///fraud.db")

@st.cache_resource
def get_engine():
    kwargs = {"connect_args": {"check_same_thread": False}} if DB_URL.startswith("sqlite") else {}
    return create_engine(DB_URL, **kwargs)

engine = get_engine()
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    filename = Column(String(255))
    created = Column(DateTime, default=datetime.utcnow)
    rows = Column(Integer)
    flagged = Column(Integer)
    note = Column(Text, default="")
    items = relationship("Item", cascade="all, delete")

class Item(Base):
    __tablename__ = "items"
    id = Column(Integer, primary_key=True)
    report_id = Column(Integer, ForeignKey("reports.id"))
    row_no = Column(Integer)
    date = Column(String(30))
    vendor = Column(String(255))
    invoice = Column(String(100))
    amount = Column(Float)
    score = Column(Float)
    reasons = Column(Text)

Base.metadata.create_all(engine)

class UserError(Exception): pass

ALIASES = {
    "date": ["date", "تاريخ"],
    "amount": ["amount", "total", "value", "مبلغ", "قيمة", "اجمالي", "إجمالي"],
    "vendor": ["vendor", "supplier", "name", "مورد", "عميل", "اسم"],
    "invoice": ["invoice", "ref", "فاتورة", "رقم"],
}
THRESHOLDS = [10000, 50000]

def get_first_digit(v):
    try:
        v = abs(float(v))
        if v == 0: return None
        s = ('%.10f' % v).lstrip('0').lstrip('.').lstrip('0')
        for ch in s:
            if ch.isdigit() and ch != '0':
                return int(ch)
        s2 = str(v).replace('.', '').replace(',', '').lstrip('0').lstrip('-')
        if s2 and s2[0].isdigit() and s2[0] != '0':
            return int(s2[0])
        return None
    except: return None

def analyze_benford(amounts):
    digits = [d for d in (get_first_digit(x) for x in amounts) if d is not None]
    if len(digits) < 50: return None
    obs = np.array([digits.count(d) for d in range(1,10)], dtype=float)
    exp_perc = np.array([math.log10(1+1/d) for d in range(1,10)])
    exp = exp_perc * len(digits)
    exp = np.maximum(exp, 1)
    try: chi2, p = chisquare(obs, exp)
    except: p = 1.0
    return {"observed": obs, "expected": exp, "percent_obs": obs/len(digits)*100, "percent_exp": exp_perc*100, "p_value": p, "is_fraud": p < 0.05, "count": len(digits)}

def read_file(name, data):
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"): df = pd.read_excel(io.BytesIO(data))
    elif ext == "csv": df = pd.read_csv(io.BytesIO(data))
    elif ext == "pdf":
        rows = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                for t in (page.extract_tables() or []): rows += t
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else: raise UserError("الصيغ المدعومة: xlsx, xls, csv, pdf")
    mapping = {}
    for col in df.columns:
        c = str(col).strip().lower()
        for key, words in ALIASES.items():
            if key not in mapping.values() and any(w in c for w in words):
                mapping[col] = key; break
    df = df.rename(columns=mapping)
    if "amount" not in df.columns: raise UserError("لم أجد عمود المبلغ")
    for k in ("date", "vendor", "invoice"):
        if k not in df.columns: df[k] = None
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["amount"]).reset_index(drop=True)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def analyze(df):
    n = len(df); score = np.zeros(n); why = [[] for _ in range(n)]
    def flag(mask, pts, text):
        for i in np.where(mask)[0]:
            score[i] += pts; why[i].append(text)
    a = df["amount"].values
    if df["invoice"].notna().any(): flag(df.duplicated(["vendor", "invoice"], keep=False).values, 40, "فاتورة مكررة")
    flag(df.duplicated(["vendor", "amount", "date"], keep=False).values, 30, "نفس المورد والمبلغ والتاريخ")
    flag((a >= 1000) & (a % 1000 == 0), 15, "مبلغ مستدير مشبوه")
    for t in THRESHOLDS: flag((a >= t*0.95) & (a < t), 25, f"أقل بقليل من حد الموافقة ({t:,.0f})")
    flag(a <= 0, 20, "مبلغ صفر أو سالب")
    benford = analyze_benford(a)
    note = ""
    if benford:
        note = f"توزيع الأرقام لا يطابق بنفورد (p={benford['p_value']:.4f})" if benford["is_fraud"] else f"التوزيع مطابق لبنفورد (p={benford['p_value']:.4f})"
        over = [d for d in range(1,10) if benford["percent_obs"][d-1] - benford["percent_exp"][d-1] > 5]
        if over:
            mask = np.array([get_first_digit(x) in over for x in a])
            flag(mask, 10, "رقم أول مبالغ في تكراره (بنفورد)")
    if n >= 20:
        X = np.log1p(np.abs(a)).reshape(-1, 1)
        pred = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
        flag(pred == -1, 30, "قيمة شاذة (AI)")
    df["score"] = np.minimum(score, 100)
    df["reasons"] = ["، ".join(r) for r in why]
    return df, note, benford

def save_report(filename, df, note):
    with Session() as s:
        rep = Report(filename=filename, rows=len(df), flagged=int((df["score"] >= 40).sum()), note=note)
        s.add(rep); s.flush()
        for i, r in df.iterrows():
            s.add(Item(report_id=rep.id, row_no=i+1, date=str(r["date"]), vendor=str(r["vendor"]), invoice=str(r["invoice"]), amount=float(r["amount"]), score=float(r["score"]), reasons=r["reasons"]))
        s.commit(); return rep.id

def list_reports(limit=50):
    with Session() as s:
        rs = s.query(Report).order_by(Report.id.desc()).limit(limit).all()
        return [{"id": r.id, "filename": r.filename, "rows": r.rows, "flagged": r.flagged} for r in rs]

def get_report_data(rid):
    with Session() as s:
        r = s.get(Report, rid)
        if not r: return None
        items = sorted(r.items, key=lambda i: -i.score)[:500]
        df = pd.DataFrame([{"السطر": i.row_no, "التاريخ": i.date, "المورد": i.vendor, "الفاتورة": i.invoice, "المبلغ": i.amount, "درجة الخطر": i.score, "الأسباب": i.reasons} for i in items])
        return {"id": r.id, "filename": r.filename, "rows": r.rows, "flagged": r.flagged, "note": r.note, "items": df}

def show_report(rid, prefix="app"):
    rep = get_report_data(rid)
    if rep is None: st.error("التقرير غير موجود"); return None, None
    st.subheader(f"تقرير #{rep['id']} — {rep['filename']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("عدد السطور", f"{rep['rows']:,}")
    c2.metric("مشبوهة (درجة ≥ 40)", f"{rep['flagged']:,}")
    c3.metric("نسبة الشبهة", f"{rep['flagged']/max(rep['rows'],1):.1%}")
    if rep["note"]: st.info(rep["note"])
    items = rep["items"]
    only_flagged = st.checkbox("عرض المشبوه فقط (درجة ≥ 40)", key=f"{prefix}_flagged_{rid}")
    view = items[items["درجة الخطر"] >= 40] if only_flagged else items
    st.dataframe(view, use_container_width=True, hide_index=True)
    buf = io.BytesIO(); items.to_excel(buf, index=False)
    st.download_button("⬇️ تحميل التقرير (Excel)", buf.getvalue(), file_name=f"report_{rid}.xlsx", key=f"{prefix}_dl_{rid}")
    return rep, items

st.title("🔍 كاشف الاحتيال المالي")
tab_upload, tab_reports = st.tabs(["📤 رفع وتحليل", "📚 التقارير"])
with tab_upload:
    file = st.file_uploader("ارفع Excel / CSV / PDF", type=["xlsx", "xls", "csv", "pdf"])
    if file is not None and st.button("تحليل وحفظ", type="primary"):
        if len(file.getvalue()) > 25*1024*1024: st.error("الملف أكبر من 25MB")
        else:
            try:
                with st.spinner("جاري التحليل..."):
                    df_raw = read_file(file.name, file.getvalue())
                    df_res, note, benford = analyze(df_raw)
                    rid = save_report(file.name, df_res, note)
                    st.session_state["last_report_id"] = rid
                    st.session_state["last_benford"] = benford
            except Exception as e: st.error(f"حدث خطأ: {e}")
    if "last_report_id" in st.session_state:
        st.success("تم التحليل ✅")
        rep, _ = show_report(st.session_state["last_report_id"], prefix="upload")
        benford = st.session_state.get("last_benford")
        if benford:
            st.divider()
            st.subheader("📊 تحليل قانون بنفورد (الصحيح)")
            chart_df = pd.DataFrame({"الرقم": list(range(1,10)), "الفعلي %": benford["percent_obs"].round(1), "المتوقع %": benford["percent_exp"].round(1)})
            st.bar_chart(chart_df, x="الرقم", y=["الفعلي %", "المتوقع %"])
            col1, col2 = st.columns(2)
            col1.metric("p-value", f"{benford['p_value']:.4f}")
            col2.metric("الحالة", "⚠️ تلاعب محتمل" if benford["is_fraud"] else "✅ طبيعي")

with tab_reports:
    reports = list_reports()
    if not reports: st.info("لا توجد تقارير محفوظة")
    else:
        st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
        chosen = st.selectbox("اختر تقريرا", options=[r["id"] for r in reports], format_func=lambda i: next(f"#{r['id']} — {r['filename']}" for r in reports if r["id"] == i))
        show_report(chosen, prefix="reports")
