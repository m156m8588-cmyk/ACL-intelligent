import io, os, re, hmac, hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, inspect, text, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

import altair as alt
st.set_page_config(page_title="كاشف الاحتيال المالي", page_icon="🔍", layout="wide")

# ───────────────────────── الثيم الزجاجي (Glassmorphism) ─────────────────────────
ACCENT, ACCENT_2, GOOD, BAD = "#f59e0b", "#fb923c", "#34d399", "#f87171"
BG_IMAGE = os.getenv("BG_IMAGE", "")  # مسار صورة خلفية اختيارية (مثلا assets/bg.jpg)

GLASS_CSS = """
#MainMenu, header, footer {visibility: hidden;}
html, body, [class*="css"] {font-family: 'Inter','Segoe UI',Tahoma,sans-serif;}
.block-container {padding-top: 2rem; max-width: 1250px;}
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p {color: #f5efe8;}
[data-testid="stCaptionContainer"] {color: rgba(245,239,232,.6);}

/* كروت زجاجية: st.container(border=True) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, rgba(255,255,255,.09), rgba(255,255,255,.03));
    backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
    border: 1px solid rgba(255,255,255,.10) !important; border-radius: 22px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
    padding: .5rem .7rem;
}
/* الفورم جوه الكارت من غير اطار تاني */
[data-testid="stForm"] {border: none !important; padding: 0 !important; background: transparent;}

/* كروت الاحصائيات */
.g-card {
    background: linear-gradient(145deg, rgba(255,255,255,.10), rgba(255,255,255,.03));
    backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
    border: 1px solid rgba(255,255,255,.10); border-radius: 22px; padding: 18px 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.08);
    transition: transform .25s ease, border-color .25s ease; margin-bottom: 14px;
}
.g-card:hover {transform: translateY(-4px); border-color: rgba(245,158,11,.45);}
.g-card.accent {background: linear-gradient(145deg, rgba(245,158,11,.38), rgba(180,83,9,.18)); border-color: rgba(245,158,11,.35);}
.g-card.bad {background: linear-gradient(145deg, rgba(248,113,113,.22), rgba(127,29,29,.10)); border-color: rgba(248,113,113,.30);}
.g-row {display:flex; align-items:center; justify-content:space-between; gap:10px;}
.g-title {font-size:.85rem; opacity:.75; margin:0 0 8px;}
.g-value {font-size:1.9rem; font-weight:800; margin:0; line-height:1.1;}
.g-icon {width:36px; height:36px; border-radius:12px; display:grid; place-items:center; background:rgba(255,255,255,.08); font-size:1.1rem;}
.g-sub {font-size:.75rem; opacity:.55; margin-top:8px;}

/* هيدر وهيرو */
.g-header h1 {margin:0; font-size:1.9rem; color:#f5efe8;}
.g-header p {margin:.25rem 0 1.2rem; opacity:.6;}
.g-hero {text-align:center; margin: 3vh 0 1.4rem;}
.g-hero .logo {width:64px; height:64px; margin:0 auto 14px; border-radius:20px; display:grid; place-items:center; font-size:1.8rem;
    background: linear-gradient(135deg, rgba(245,158,11,.5), rgba(180,83,9,.25)); border:1px solid rgba(245,158,11,.4);
    box-shadow: 0 10px 30px rgba(245,158,11,.25);}
.g-hero h1 {margin:0; font-size:2rem;}
.g-hero p {opacity:.6; margin:.4rem 0 0;}

/* ازرار */
.stButton>button, .stDownloadButton>button, div[data-testid="stFormSubmitButton"]>button {
    width: 100%; height: 48px; border: none !important; border-radius: 999px !important; font-weight: 700;
    background: linear-gradient(135deg, #f59e0b, #fb923c) !important; color: #1a1004 !important;
    box-shadow: 0 6px 22px rgba(245,158,11,.35); transition: transform .2s ease, box-shadow .2s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover, div[data-testid="stFormSubmitButton"]>button:hover {
    transform: translateY(-2px); box-shadow: 0 10px 28px rgba(245,158,11,.5);
}

/* انبوتس */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"]>div {
    background: rgba(255,255,255,.92) !important;
    border: 1px solid rgba(255,255,255,.35) !important;
    border-radius: 14px !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea {
    background: transparent !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: #000000;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: rgba(0,0,0,.45) !important;
    -webkit-text-fill-color: rgba(0,0,0,.45) !important;
}
/* الـ Autofill */
input:-webkit-autofill, input:-webkit-autofill:focus {
    -webkit-box-shadow: 0 0 0 1000px #ffffff inset !important;
    -webkit-text-fill-color: #000000 !important;
}
/* زرار اظهار الباسورد (العين) */
[data-baseweb="input"] button svg {fill: #000000 !important;}

/* رفع الملفات */
[data-testid="stFileUploaderDropzone"] {background: rgba(255,255,255,.04); border: 1.5px dashed rgba(245,158,11,.5); border-radius: 20px;}

/* تابس */
.stTabs [data-baseweb="tab-list"] {gap:6px; background:rgba(255,255,255,.05); padding:6px; border-radius:999px; border:1px solid rgba(255,255,255,.08); width:fit-content;}
.stTabs [data-baseweb="tab"] {border-radius:999px; padding:6px 18px; height:auto;}
.stTabs [aria-selected="true"] {background: rgba(245,158,11,.25); color:#f59e0b;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none;}

/* جداول واكسباندر ورسايل */
[data-testid="stDataFrame"] {border-radius:18px; overflow:hidden; border:1px solid rgba(255,255,255,.08);}
[data-testid="stExpander"] {background: rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08) !important; border-radius:16px;}
[data-testid="stAlert"] {border-radius:16px; backdrop-filter: blur(12px);}
::-webkit-scrollbar {width:8px; height:8px;} ::-webkit-scrollbar-thumb {background:rgba(255,255,255,.18); border-radius:8px;}
"""

def _bg_css() -> str:
    if BG_IMAGE and os.path.exists(BG_IMAGE):
        import base64, mimetypes
        mime = mimetypes.guess_type(BG_IMAGE)[0] or "image/jpeg"
        b64 = base64.b64encode(open(BG_IMAGE, "rb").read()).decode()
        return f'.stApp {{background: linear-gradient(rgba(8,6,4,.55), rgba(8,6,4,.78)), url("data:{mime};base64,{b64}") center/cover fixed no-repeat;}}'
    return """.stApp {background:
        radial-gradient(900px 500px at 8% 8%, rgba(245,158,11,.28), transparent 60%),
        radial-gradient(800px 600px at 95% 20%, rgba(180,83,9,.30), transparent 60%),
        radial-gradient(900px 700px at 50% 105%, rgba(120,53,15,.35), transparent 60%), #0b0907;
        background-attachment: fixed;}"""

st.markdown(f"<style>{_bg_css()}{GLASS_CSS}</style>", unsafe_allow_html=True)

def section_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="g-header"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def stat_card(title: str, value: str, icon: str = "◔", variant: str = "", sub: str = ""):
    sub_html = f'<div class="g-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="g-card {variant}"><div class="g-row"><p class="g-title">{title}</p><div class="g-icon">{icon}</div></div>'
        f'<p class="g-value">{value}</p>{sub_html}</div>', unsafe_allow_html=True)

def _style_chart(ch, height=260):
    return (ch.properties(height=height, background="transparent")
              .configure_view(strokeWidth=0)
              .configure_axis(labelColor="#d9cfc4", gridColor="rgba(255,255,255,0.07)", domainColor="rgba(255,255,255,0.15)", tickColor="rgba(255,255,255,0.15)")
              .configure_legend(labelColor="#d9cfc4", titleColor="#d9cfc4", orient="bottom"))

def risk_charts(items: pd.DataFrame, rows: int, flagged: int):
    if items.empty:
        return
    left, right = st.columns([2, 1])
    with left, st.container(border=True):
        st.markdown("##### توزيع درجات الخطر")
        d = items[["درجة الخطر"]].rename(columns={"درجة الخطر": "score"})
        hist = alt.Chart(d).mark_bar(color=ACCENT, opacity=.9, cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=20), title=None), y=alt.Y("count():Q", title=None))
        st.altair_chart(_style_chart(hist), use_container_width=True, theme=None)
    with right, st.container(border=True):
        st.markdown("##### مشبوه مقابل سليم")
        pie = pd.DataFrame({"الحالة": ["مشبوه", "سليم"], "العدد": [flagged, max(rows - flagged, 0)]})
        donut = alt.Chart(pie).mark_arc(innerRadius=62, stroke="rgba(0,0,0,0)").encode(
            theta="العدد:Q", color=alt.Color("الحالة:N", scale=alt.Scale(domain=["مشبوه", "سليم"], range=[ACCENT, "#4b443c"]), legend=alt.Legend(title=None)))
        st.altair_chart(_style_chart(donut), use_container_width=True, theme=None)


# ───────────────────────── قاعدة البيانات ─────────────────────────
def _db_url() -> str:
    # الأولوية: متغير بيئة DATABASE_URL، بعدين Streamlit secrets، وآخر حاجة SQLite محلي
    url = os.getenv("DATABASE_URL")
    if not url:
        try:
            url = st.secrets["DATABASE_URL"]
        except Exception:
            url = "sqlite:///fraud.db"
    url = url.strip()
    if url.startswith("postgres://"):  # SQLAlchemy مش بيقبل postgres://
        url = "postgresql://" + url[len("postgres://"):]
    return url

DB_URL = _db_url()
IS_SQLITE = DB_URL.startswith("sqlite")

@st.cache_resource
def get_engine():
    if IS_SQLITE:
        return create_engine(DB_URL, connect_args={"check_same_thread": False})
    # Supabase (Postgres): اتصال مُجمّع مع فحص الاتصال قبل الاستخدام
    return create_engine(DB_URL, pool_pre_ping=True, pool_size=5, max_overflow=2, pool_recycle=300)

def _db_fail(e: Exception):
    st.error(
        f"تعذر الاتصال بقاعدة البيانات ({type(e).__name__}). "
        "راجع DATABASE_URL: استخدم رابط Session pooler من Supabase، "
        "وتأكد من تشفير الرموز الخاصة في كلمة السر، ومن إضافة psycopg2-binary في requirements.txt."
    )
    st.stop()

try:
    engine = get_engine()
except Exception as e:
    _db_fail(e)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(300), nullable=False)
    created = Column(DateTime, default=datetime.utcnow)

class Report(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
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

def _migrate():
    # لو جدول reports كان موجود من قبل من غير user_id، نضيف العمود
    cols = [c["name"] for c in inspect(engine).get_columns("reports")]
    if "user_id" not in cols:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE reports ADD COLUMN user_id INTEGER REFERENCES users(id)"))

try:
    Base.metadata.create_all(engine)
    _migrate()
except Exception as e:
    _db_fail(e)

class UserError(Exception): pass

# ───────────────────────── الحسابات ─────────────────────────
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PBKDF2_ITERS = 200_000

def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, PBKDF2_ITERS)
    return f"{salt.hex()}${h.hex()}"

def verify_password(pw: str, stored: str) -> bool:
    try:
        salt_hex, h_hex = stored.split("$")
        h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), PBKDF2_ITERS)
        return hmac.compare_digest(h.hex(), h_hex)
    except Exception:
        return False

def register_user(email: str, pw: str, pw2: str) -> dict:
    email = email.strip().lower()
    if not EMAIL_RE.match(email):
        raise UserError("البريد الإلكتروني غير صحيح")
    if len(pw) < 8:
        raise UserError("كلمة السر لازم تكون 8 حروف على الأقل")
    if pw != pw2:
        raise UserError("كلمتا السر غير متطابقتين")
    with Session() as s:
        if s.query(User).filter_by(email=email).first():
            raise UserError("هذا البريد مسجل بالفعل")
        u = User(email=email, password_hash=hash_password(pw))
        s.add(u)
        s.commit()
        return {"id": u.id, "email": u.email}

def login_user(email: str, pw: str) -> dict:
    email = email.strip().lower()
    with Session() as s:
        u = s.query(User).filter_by(email=email).first()
        if not u or not verify_password(pw, u.password_hash):
            raise UserError("البريد أو كلمة السر غير صحيحة")
        return {"id": u.id, "email": u.email}

def auth_screen():
    t_login, t_signup = st.tabs(["🔑 تسجيل الدخول", "🆕 إنشاء حساب"])
    with t_login:
        with st.form("login_form"):
            email = st.text_input("البريد الإلكتروني", key="li_email")
            pw = st.text_input("كلمة السر", type="password", key="li_pw")
            if st.form_submit_button("دخول"):
                try:
                    st.session_state["user"] = login_user(email, pw)
                    st.rerun()
                except UserError as e:
                    st.error(str(e))
    with t_signup:
        with st.form("signup_form"):
            email = st.text_input("البريد الإلكتروني", key="su_email")
            pw = st.text_input("كلمة السر (8 حروف على الأقل)", type="password", key="su_pw")
            pw2 = st.text_input("تأكيد كلمة السر", type="password", key="su_pw2")
            if st.form_submit_button("إنشاء الحساب"):
                try:
                    st.session_state["user"] = register_user(email, pw, pw2)
                    st.rerun()
                except UserError as e:
                    st.error(str(e))

# ───────────────────────── التحليل ─────────────────────────
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
    # أول رقم معنوي للمبالغ الموجبة فقط (الصفر والسالب مالهمش معنى في بنفورد)
    digits = pd.Series([int(f"{x:.10e}"[0]) if x > 0 else np.nan for x in a])
    valid = digits.dropna()
    m = len(valid)
    if m >= 100:
        obs = np.array([(valid == d).sum() for d in range(1, 10)], dtype=float)
        exp_p = np.log10(1 + 1 / np.arange(1, 10))
        obs_p = obs / m
        p = chisquare(obs, exp_p * m).pvalue
        mad = float(np.abs(obs_p - exp_p).mean())
        # مع الداتا الكبيرة كاي تربيع بيرفض أي فرق صغير، فبنشترط كمان MAD >= 0.012
        if p < 0.01 and mad >= 0.012:
            note = f"توزيع الأرقام الأولى لا يطابق قانون بنفورد (p={p:.4f}، MAD={mad:.4f})"
            over = [d for d in range(1, 10) if obs_p[d - 1] - exp_p[d - 1] > 0.03]
            if over:
                flag(digits.isin(over).values, 10, "رقم أول مبالغ في تكراره (بنفورد)")
    if n >= 20:
        X = np.log1p(np.abs(a)).reshape(-1, 1)
        pred = IsolationForest(contamination=0.05, random_state=42).fit_predict(X)
        flag(pred == -1, 30, "قيمة شاذة إحصائيا (نموذج AI)")
    df["score"] = np.minimum(score, 100)
    df["reasons"] = ["، ".join(r) for r in why]
    return df, note

# ───────────────────────── الحفظ والعرض (كله مقيّد بالمستخدم) ─────────────────────────
def save_report(user_id: int, filename: str, df: pd.DataFrame, note: str) -> int:
    with Session() as s:
        rep = Report(user_id=user_id, filename=filename, rows=len(df), flagged=int((df["score"] >= 40).sum()), note=note)
        s.add(rep)
        s.flush()
        s.add_all(Item(report_id=rep.id, row_no=i + 1, date=str(r["date"]), vendor=str(r["vendor"]), invoice=str(r["invoice"]), amount=float(r["amount"]), score=float(r["score"]), reasons=r["reasons"]) for i, r in df.iterrows())
        s.commit()
        return rep.id

def list_reports(user_id: int, limit: int = 50):
    with Session() as s:
        rs = s.query(Report).filter(Report.user_id == user_id).order_by(Report.id.desc()).limit(limit).all()
        return [{"id": r.id, "filename": r.filename, "created": r.created, "rows": r.rows, "flagged": r.flagged} for r in rs]

def extra_checks(rows):
    cols = ["السطر", "التاريخ", "المورد", "الفاتورة", "المبلغ"]
    df = pd.DataFrame([{"السطر": i.row_no, "التاريخ": i.date, "المورد": i.vendor, "الفاتورة": i.invoice, "المبلغ": i.amount} for i in rows], columns=cols)
    if df.empty:
        return df, [], df
    inv = df["الفاتورة"].astype(str).str.strip()
    has_inv = ~inv.str.lower().isin(["none", "nan", ""])
    withinv = df[has_inv]
    dups = withinv[withinv.duplicated(["المورد", "الفاتورة"], keep=False)].sort_values(["المورد", "الفاتورة"])
    nums = pd.to_numeric(inv[has_inv].str.extract(r"(\d+)\D*$")[0], errors="coerce").dropna().astype(int)
    gaps = []
    if len(nums) >= 2 and nums.max() - nums.min() <= 100000:
        gaps = sorted(set(range(int(nums.min()), int(nums.max()) + 1)) - set(nums))
    amt = df["المبلغ"].abs()
    rounds = df[(amt >= 1000) & (amt % 1000 == 0)]
    return dups, gaps, rounds

def get_report(rid: int, user_id: int):
    with Session() as s:
        r = s.get(Report, rid)
        if not r or r.user_id != user_id:  # مينفعش مستخدم يشوف تقرير غيره
            return None
        dups, gaps, rounds = extra_checks(r.items)
        items = sorted(r.items, key=lambda i: -i.score)[:500]
        return {"dups": dups, "gaps": gaps, "rounds": rounds, "id": r.id, "filename": r.filename, "rows": r.rows, "flagged": r.flagged, "note": r.note, "items": pd.DataFrame([{"السطر": i.row_no, "التاريخ": i.date, "المورد": i.vendor, "الفاتورة": i.invoice, "المبلغ": i.amount, "درجة الخطر": i.score, "الأسباب": i.reasons} for i in items])}

def show_report(rid: int, user_id: int, prefix: str = ""):
    rep = get_report(rid, user_id)
    if rep is None:
        st.error("التقرير غير موجود"); return
    ratio = rep["flagged"] / max(rep["rows"], 1)
    section_header(f"تقرير #{rep['id']}", rep["filename"])
    c1, c2, c3 = st.columns(3)
    with c1: stat_card("عدد السطور", f"{rep['rows']:,}", "📊", "accent")
    with c2: stat_card("مشبوهة (درجة ≥ 40)", f"{rep['flagged']:,}", "⚠️", "bad" if rep["flagged"] else "")
    with c3: stat_card("نسبة الشبهة", f"{ratio:.1%}", "🎯")
    d1, d2, d3 = st.columns(3)
    with d1: stat_card("فواتير متكررة", f"{len(rep['dups']):,}", "🔁")
    with d2: stat_card("أرقام فواتير ناقصة", f"{len(rep['gaps']):,}", "❓")
    with d3: stat_card("مبالغ مقفولة (000)", f"{len(rep['rounds']):,}", "🔢")
    risk_charts(rep["items"], rep["rows"], rep["flagged"])
    with st.expander("تفاصيل الفواتير المتكررة"):
        st.dataframe(rep["dups"], use_container_width=True, hide_index=True)
    with st.expander("تفاصيل أرقام الفواتير الناقصة"):
        st.write(rep["gaps"][:500] if rep["gaps"] else "لا توجد أرقام ناقصة")
    with st.expander("تفاصيل المبالغ المقفولة"):
        st.dataframe(rep["rounds"], use_container_width=True, hide_index=True)
    if rep["note"]: st.warning(rep["note"])
    items = rep["items"]
    only_flagged = st.checkbox("عرض المشبوه فقط (درجة ≥ 40)", key=f"{prefix}_flagged_{rid}")
    view = items[items["درجة الخطر"] >= 40] if only_flagged else items
    st.dataframe(view, use_container_width=True, hide_index=True)
    buf = io.BytesIO()
    items.to_excel(buf, index=False)
    st.download_button("⬇️ تحميل التقرير (Excel)", buf.getvalue(), file_name=f"report_{rid}.xlsx", key=f"{prefix}_dl_{rid}")

# ───────────────────────── الواجهة ─────────────────────────
user = st.session_state.get("user")
if not user:
    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.markdown(
            '<div class="g-hero"><div class="logo">🔍</div><h1>كاشف الاحتيال المالي</h1>'
            '<p>سجّل دخولك أو أنشئ حسابا جديدا علشان ترفع ملفاتك وتشوف تقاريرك.</p></div>',
            unsafe_allow_html=True)
        with st.container(border=True):
            auth_screen()
    st.stop()

h1, h2 = st.columns([4, 1])
with h1:
    section_header("🔍 كاشف الاحتيال المالي",
                   f"👤 {user['email']}  |  🗄️ قاعدة البيانات: " + ("SQLite محلي" if IS_SQLITE else "Supabase (PostgreSQL)"))
with h2:
    st.write("")
    if st.button("تسجيل الخروج"):
        st.session_state.clear()
        st.rerun()

tab_upload, tab_reports = st.tabs(["📤 رفع وتحليل ملف", "📚 التقارير السابقة"])
with tab_upload:
    with st.container(border=True):
        file = st.file_uploader("ارفع ملف Excel أو CSV أو PDF", type=["xlsx", "xls", "csv", "pdf"])
        run = file is not None and st.button("تحليل وحفظ", type="primary")
    if run:
        data = file.getvalue()
        if len(data) > MAX_SIZE:
            st.error("الملف أكبر من 25MB")
        else:
            try:
                with st.spinner("جاري التحليل..."):
                    df, note = analyze(read_file(file.name, data))
                    st.session_state["last_report"] = save_report(user["id"], file.name, df, note)
            except UserError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"حدث خطأ: {e}")
    if "last_report" in st.session_state:
        st.success("تم التحليل وحفظ التقرير ✅")
        show_report(st.session_state["last_report"], user["id"], prefix="upload")

with tab_reports:
    reports = list_reports(user["id"])
    if not reports:
        st.info("لا توجد تقارير محفوظة بعد.")
    else:
        with st.container(border=True):
            st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)
            chosen = st.selectbox("اختر تقريرا لعرضه", options=[r["id"] for r in reports], format_func=lambda i: next(f"#{r['id']} — {r['filename']}" for r in reports if r["id"] == i))
        show_report(chosen, user["id"], prefix="reports")
