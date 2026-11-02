import io, os
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
st.markdown("""
<style>
/* اخفاء قايمة ستريمليت */
#MainMenu, header, footer {visibility: hidden;}
.stApp {background: #f8f9fb;}
/* كارت زي التطبيق */
.app-card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    margin-bottom: 16px;
    border: 1px solid #f0f0f0;
}
.top-bar {
    background: linear-gradient(135deg, #6c5ce7, #a29bfe);
    color: white;
    padding: 18px 20px;
    border-radius: 0 0 25px 25px;
    margin: -80px -40px 20px -40px;
    text-align: center;
}
.bottom-nav {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: white;
    border-top: 1px solid #eee;
    display: flex;
    justify-content: space-around;
    padding: 10px 0 20px 0;
    z-index: 999;
    border-radius: 20px 20px 0 0;
}
.stButton>button {
    width: 100%;
    border-radius: 14px !important;
    height: 50px;
    font-weight: bold;
    background: #6c5ce7 !important;
    color: white !important;
    border: none !important;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
</style>
<div class="top-bar">
    <h2 style="margin:0">🔍 كاشف الاحتيال</h2>
    <p style="margin:4px 0 0 0; opacity:0.9; font-size:13px">الذكاء الاصطناعي للمراجعة المالية</p>
</div>
""", unsafe_allow_html=True)
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
THRESHOLDS = [float(x) for x in os.getenv("APPROVAL_LIMITS", "10000,50000").split(",")]
WEEKEND = {4, 5}
MAX_SIZE = 25 * 1024 * 1024

def read_file(name: str, data: bytes) -> pd.DataFrame:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(data))
    elif ext == "csv":
        df = pd.read_csv(io.BytesIO(data))
    elif ext == "pdf":
        rows = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                for t in (page.extract_tables() or []):
                    rows += t
        if len(rows) < 2:
            raise UserError("لم يتم العثور على جدول داخل ملف PDF")
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        raise UserError("الصيغ المدعومة: xlsx, xls, csv, pdf")
    mapping = {}
    for col in df.columns:
        c = str(col).strip().lower()
        for key, words in ALIASES.items():
            if key not in mapping.values() and any(w in c for w in words):
                mapping[col] = key
                break
    df = df.rename(columns=mapping)
    if "amount" not in df.columns:
        raise UserError("لم أجد عمود المبلغ (amount / المبلغ).")
    for k in ("date", "vendor", "invoice"):
        if k not in df.columns:
            df[k] = None
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["amount"]).reset_index(drop=True)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def analyze(df: pd.DataFrame):
    n = len(df)
    score = np.zeros(n)
    why = [[] for _ in range(n)]
    def flag(mask, pts, text):
        for i in np.where(mask)[0]:
            score[i] += pts
            why[i].append(text)
    if df["invoice"].notna().any():
        flag(df.duplicated(["vendor", "invoice"], keep=False).values, 40, "فاتورة مكررة (نفس المورد والرقم)")
    flag(df.duplicated(["vendor", "amount", "date"], keep=False).values, 30, "نفس المورد والمبلغ والتاريخ")
    a = df["amount"].values
    flag((a >= 1000) & (a % 1000 == 0), 15, "مبلغ مستدير بشكل مشبوه")
    for t in THRESHOLDS:
        flag((a >= t * 0.95) & (a < t), 25, f"أقل بقليل من حد الموافقة ({t:,.0f})")
    flag(df["date_parsed"].dt.weekday.isin(WEEKEND).values, 10, "عملية في يوم عطلة")
    flag(a <= 0, 20, "مبلغ صفر أو سالب")
    note = ""
    digits = pd.Series(np.abs(a)).astype(str).str.lstrip("0.").str[0]
    digits = pd.to_numeric(digits, errors="coerce")
    if n >= 100:
        obs = np.array([(digits == d).sum() for d in range(1, 10)], dtype=float)
        exp = np.log10(1 + 1 / np.arange(1, 10)) * obs.sum()
        p = chisquare(obs, exp).pvalue
        if p < 0.01:
            note = f"توزيع الأرقام الأولى لا يطابق قانون بنفورد (p={p:.4f})"
            over = [d for d in range(1, 10) if obs[d - 1] / obs.sum() - exp[d - 1] / obs.sum() > 0.05]
            flag(digits.isin(over).values, 10, "رقم أول مبالغ في تكراره (بنفورد)")
    if n >= 20:
        X = np.log1p(np.abs(a)).reshape(-1, 1)
        pred = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
        flag(pred == -1, 30, "قيمة شاذة إحصائيا (نموذج AI)")
    df["score"] = np.minimum(score, 100)
    df["reasons"] = ["، ".join(r) for r in why]
    return df, note

def save_report(filename: str, df: pd.DataFrame, note: str) -> int:
    with Session() as s:
        rep = Report(filename=filename, rows=len(df), flagged=int((df["score"] >= 40).sum()), note=note)
        s.add(rep)
        s.flush()
        s.add_all(Item(report_id=rep.id, row_no=i + 1, date=str(r["date"]), vendor=str(r["vendor"]), invoice=str(r["invoice"]), amount=float(r["amount"]), score=float(r["score"]), reasons=r["reasons"]) for i, r in df.iterrows())
        s.commit()
        return rep.id

def list_reports(limit: int = 50):
    with Session() as s:
        rs = s.query(Report).order_by(Report.id.desc()).limit(limit).all()
        return [{"id": r.id, "filename": r.filename, "created": r.created, "rows": r.rows, "flagged": r.flagged} for r in rs]

def get_report(rid: int):
    with Session() as s:
        r = s.get(Report, rid)
        if not r: return None
        items = sorted(r.items, key=lambda i: -i.score)[:500]
        return {"id": r.id, "filename": r.filename, "rows": r.rows, "flagged": r.flagged, "note": r.note, "items": pd.DataFrame([{"السطر": i.row_no, "التاريخ": i.date, "المورد": i.vendor, "الفاتورة": i.invoice, "المبلغ": i.amount, "درجة الخطر": i.score, "الأسباب": i.reasons} for i in items])}

def show_report(rid: int, prefix: str = ""):
    rep = get_report(rid)
    if rep is None:
        st.error("التقرير غير موجود"); return
    st.subheader(f"تقرير #{rep['id']} — {rep['filename']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("عدد السطور", f"{rep['rows']:,}")
    c2.metric("مشبوهة (درجة ≥ 40)", f"{rep['flagged']:,}")
    c3.metric("نسبة الشبهة", f"{rep['flagged'] / max(rep['rows'], 1):.1%}")
    if rep["note"]: st.warning(rep["note"])
    items = rep["items"]
    only_flagged = st.checkbox(f"عرض المشبوه فقط (درجة ≥ 40)", key=f"{prefix}_flagged_{rid}_{id(rep)}")
    view = items[items["درجة الخطر"] >= 40] if only_flagged else items
    st.dataframe(view, use_container_width=True, hide_index=True)
    buf = io.BytesIO()
    items.to_excel(buf, index=False)
    st.download_button("⬇️ تحميل التقرير (Excel)", buf.getvalue(), file_name=f"report_{rid}.xlsx", key=f"{prefix}_dl_{rid}_{id(rep)}")

st.title("🔍 كاشف الاحتيال المالي")
tab_upload, tab_reports = st.tabs(["📤 رفع وتحليل ملف", "📚 التقارير السابقة"])
with tab_upload:
    file = st.file_uploader("ارفع ملف Excel أو CSV أو PDF", type=["xlsx", "xls", "csv", "pdf"])
    if file is not None and st.button("تحليل وحفظ", type="primary"):
        data = file.getvalue()
        if len(data) > MAX_SIZE:
            st.error("الملف أكبر من 25MB")
        else:
            try:
                with st.spinner("جاري التحليل..."):
                    df, note = analyze(read_file(file.name, data))
                    st.session_state["last_report"] = save_report(file.name, df, note)
            except UserError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"حدث خطأ: {e}")
    if "last_report" in st.session_state:
        st.success("تم التحليل وحفظ التقرير ✅")
        show_report(st.session_state["last_report"], prefix="upload")

with tab_reports:
    reports = list_reports()
    if not reports:
        st.info("لا توجد تقارير محفوظة بعد.")
    else:
        st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
        chosen = st.selectbox("اختر تقريرا لعرضه", options=[r["id"] for r in reports], format_func=lambda i: next(f"#{r['id']} — {r['filename']}" for r in reports if r["id"] == i))
        show_report(chosen, prefix="reports")
