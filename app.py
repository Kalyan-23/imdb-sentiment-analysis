import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ SentientGrid // Neural Sentiment OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Neon Cyberpunk CSS ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap');

:root {
    --neon-cyan:    #00ffe5;
    --neon-pink:    #ff2d78;
    --neon-yellow:  #f5ff00;
    --neon-purple:  #bf00ff;
    --neon-orange:  #ff6600;
    --bg-deep:      #020408;
    --bg-mid:       #080d14;
    --bg-panel:     #0b1120;
    --bg-card:      #0d1628;
    --grid-line:    rgba(0,255,229,0.06);
    --border-dim:   rgba(0,255,229,0.18);
    --border-hot:   rgba(0,255,229,0.55);
    --text-primary: #c8f0ff;
    --text-dim:     #4a7a8a;
    --text-muted:   #1e3a4a;
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: var(--bg-deep) !important;
    color: var(--text-primary);
}

/* Scrolling cyber grid background */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-deep) !important;
    background-image:
        linear-gradient(var(--grid-line) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
    background-size: 40px 40px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020810 0%, #060d1a 100%) !important;
    border-right: 1px solid var(--border-dim) !important;
    box-shadow: 4px 0 30px rgba(0,255,229,0.07) !important;
}

h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.08em !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    color: var(--neon-cyan);
    text-shadow:
        0 0 8px var(--neon-cyan),
        0 0 24px rgba(0,255,229,0.6),
        0 0 60px rgba(0,255,229,0.2);
    letter-spacing: 0.12em;
    line-height: 1.15;
    text-transform: uppercase;
    position: relative;
}

.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    color: var(--neon-pink);
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 2rem;
    text-shadow: 0 0 8px rgba(255,45,120,0.7);
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-top: 1px solid var(--neon-cyan);
    border-radius: 4px;
    padding: 1.8rem;
    margin-bottom: 1rem;
    clip-path: polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 0 100%);
    box-shadow:
        0 0 20px rgba(0,255,229,0.04),
        inset 0 0 30px rgba(0,255,229,0.02);
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px));
    position: relative;
    transition: box-shadow 0.2s ease;
}

.metric-card:hover {
    box-shadow: 0 0 20px rgba(0,255,229,0.15);
}

.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--neon-cyan);
    text-shadow: 0 0 12px rgba(0,255,229,0.8);
}

.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

/* ── Sentiment Result Boxes ── */
.positive-result {
    background: linear-gradient(135deg, #001a0d, #002a14);
    border: 1px solid #00ffe5;
    border-left: 3px solid #00ffe5;
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,255,229,0.2), inset 0 0 20px rgba(0,255,229,0.04);
    clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
}

.negative-result {
    background: linear-gradient(135deg, #1a0008, #2a0012);
    border: 1px solid #ff2d78;
    border-left: 3px solid #ff2d78;
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,45,120,0.2), inset 0 0 20px rgba(255,45,120,0.04);
    clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 0 100%);
}

.result-emoji { font-size: 3rem; }

.result-label {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.5rem 0;
    letter-spacing: 0.15em;
}

.result-conf {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-dim);
    font-size: 0.8rem;
    letter-spacing: 0.08em;
}

/* ── Clean text box ── */
.clean-box {
    background: #040c14;
    border: 1px solid var(--border-dim);
    border-left: 2px solid var(--neon-cyan);
    border-radius: 2px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: var(--neon-cyan);
    font-family: 'Share Tech Mono', monospace;
    min-height: 60px;
    word-break: break-word;
    text-shadow: 0 0 6px rgba(0,255,229,0.4);
}

/* ── Model winner box ── */
.model-winner {
    background: linear-gradient(135deg, #060e1a, #0a1628);
    border: 1px solid rgba(245,255,0,0.35);
    border-top: 2px solid var(--neon-yellow);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(245,255,0,0.1);
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 0 100%);
}

/* ── Sidebar info ── */
.sidebar-info {
    background: #050d18;
    border: 1px solid var(--border-dim);
    border-left: 2px solid var(--neon-pink);
    border-radius: 2px;
    padding: 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
}

.sidebar-info b { color: var(--neon-cyan); }

/* ── Textarea ── */
.stTextArea textarea {
    background: #040c14 !important;
    border: 1px solid var(--border-dim) !important;
    border-left: 2px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: var(--neon-cyan) !important;
}

.stTextArea textarea:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 12px rgba(0,255,229,0.25) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    color: var(--neon-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    border: 1px solid var(--neon-cyan) !important;
    border-radius: 2px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px)) !important;
    transition: all 0.15s ease !important;
    text-shadow: 0 0 8px rgba(0,255,229,0.6) !important;
    box-shadow: 0 0 12px rgba(0,255,229,0.1), inset 0 0 8px rgba(0,255,229,0.05) !important;
}

.stButton > button:hover {
    background: rgba(0,255,229,0.08) !important;
    box-shadow: 0 0 24px rgba(0,255,229,0.35), inset 0 0 12px rgba(0,255,229,0.08) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border-dim) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-dim) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 1.2rem !important;
    text-transform: uppercase !important;
    border-bottom: 2px solid transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    text-shadow: 0 0 8px rgba(0,255,229,0.7) !important;
    background: rgba(0,255,229,0.03) !important;
}

/* ── Progress bars ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple)) !important;
    box-shadow: 0 0 8px rgba(0,255,229,0.5) !important;
}

.stProgress > div {
    background: #0b1828 !important;
    border-radius: 0 !important;
    height: 6px !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #040c14 !important;
    border: 1px solid var(--border-dim) !important;
    color: var(--neon-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-dim) !important;
}

/* ── Success / warning / spinner ── */
.stSuccess {
    background: #001a0d !important;
    border: 1px solid #00ffe5 !important;
    color: var(--neon-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
}

.stWarning {
    background: #1a0d00 !important;
    border: 1px solid var(--neon-orange) !important;
    color: var(--neon-orange) !important;
    border-radius: 2px !important;
}

.stInfo {
    background: #050d18 !important;
    border: 1px solid var(--border-dim) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 2px !important;
}

/* ── Scanline overlay on the main content ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.08) 2px,
        rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Glitch animation for hero ── */
@keyframes glitch {
    0%, 90%, 100% { text-shadow: 0 0 8px var(--neon-cyan), 0 0 24px rgba(0,255,229,0.6); transform: none; }
    92%  { text-shadow: -2px 0 var(--neon-pink), 2px 0 var(--neon-cyan); transform: translateX(1px); }
    94%  { text-shadow: 2px 0 var(--neon-pink), -2px 0 var(--neon-cyan); transform: translateX(-1px); }
    96%  { text-shadow: 0 0 8px var(--neon-cyan); transform: none; }
}

.hero-title { animation: glitch 6s infinite; }

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

def get_intensity(prob_positive):
    return int(prob_positive * 100)

def intensity_label(score):
    if score >= 85: return "⚡ CRITICAL POSITIVE"
    elif score >= 65: return "▲ SIGNAL POSITIVE"
    elif score >= 45: return "◈ NEUTRAL // MIXED"
    elif score >= 25: return "▼ SIGNAL NEGATIVE"
    else: return "☠ CRITICAL NEGATIVE"

def intensity_color(score):
    if score >= 65: return "#00ffe5"
    elif score >= 45: return "#f5ff00"
    else: return "#ff2d78"


@st.cache_resource(show_spinner=False)
def train_all_models(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    review_col    = next((c for c in df.columns if 'review'    in c.lower()), df.columns[0])
    sentiment_col = next((c for c in df.columns if 'sentiment' in c.lower()), df.columns[1])
    df = df.rename(columns={review_col: 'review', sentiment_col: 'sentiment'})
    df['clean'] = df['review'].apply(clean_text)
    df['label'] = (df['sentiment'].str.lower() == 'positive').astype(int)
    df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label'], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_tr  = tfidf.fit_transform(X_train)
    X_te  = tfidf.transform(X_test)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    t0 = time.time()
    lr.fit(X_tr, y_train)
    lr_time = round(time.time() - t0, 2)
    lr_pred = lr.predict(X_te)
    results['Logistic Regression'] = {
        'model': lr, 'acc': accuracy_score(y_test, lr_pred),
        'f1': f1_score(y_test, lr_pred),
        'cm': confusion_matrix(y_test, lr_pred),
        'time': lr_time,
        'report': classification_report(y_test, lr_pred, target_names=['Negative','Positive'], output_dict=True)
    }

    # Naive Bayes
    nb = MultinomialNB(alpha=0.1)
    t0 = time.time()
    nb.fit(X_tr, y_train)
    nb_time = round(time.time() - t0, 2)
    nb_pred = nb.predict(X_te)
    results['Naive Bayes'] = {
        'model': nb, 'acc': accuracy_score(y_test, nb_pred),
        'f1': f1_score(y_test, nb_pred),
        'cm': confusion_matrix(y_test, nb_pred),
        'time': nb_time,
        'report': classification_report(y_test, nb_pred, target_names=['Negative','Positive'], output_dict=True)
    }

    # SVM (calibrated for probability)
    svm_base = LinearSVC(max_iter=2000)
    svm = CalibratedClassifierCV(svm_base)
    t0 = time.time()
    svm.fit(X_tr, y_train)
    svm_time = round(time.time() - t0, 2)
    svm_pred = svm.predict(X_te)
    results['SVM'] = {
        'model': svm, 'acc': accuracy_score(y_test, svm_pred),
        'f1': f1_score(y_test, svm_pred),
        'cm': confusion_matrix(y_test, svm_pred),
        'time': svm_time,
        'report': classification_report(y_test, svm_pred, target_names=['Negative','Positive'], output_dict=True)
    }

    best_model_name = max(results, key=lambda k: results[k]['acc'])
    return tfidf, results, best_model_name, df


def predict_with_model(text, tfidf, model):
    vec  = tfidf.transform([clean_text(text)])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return pred, prob


# ── Matplotlib Cyberpunk Style ────────────────────────────────
CYBER_BG    = '#030810'
CYBER_PANEL = '#0a1220'
CYAN        = '#00ffe5'
PINK        = '#ff2d78'
YELLOW      = '#f5ff00'
PURPLE      = '#bf00ff'
TEXT_DIM    = '#4a7a8a'
BORDER      = '#0d2233'

def cyber_fig(w=5, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CYBER_BG)
    ax.set_facecolor(CYBER_PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    # faint grid
    ax.grid(True, color='#0d2233', linewidth=0.5, alpha=0.7)
    return fig, ax


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:1rem;font-weight:900;
                color:#00ffe5;text-shadow:0 0 12px rgba(0,255,229,0.8);
                letter-spacing:0.12em;padding:0.5rem 0 0.2rem">
        ⚡ SENTIMENT ANALYZER
    </div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                color:#ff2d78;letter-spacing:0.2em;margin-bottom:1rem">
        NEURAL SENTIMENT OS v2.0
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                color:#4a7a8a;letter-spacing:0.12em;margin-bottom:0.5rem">
        // LOAD DATASET
    </div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload IMDB Dataset CSV", type=["csv"])
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
    <b>// DATASET SPECS</b><br><br>
    FILE &nbsp;› <b>IMDB Dataset.csv</b><br>
    ROWS &nbsp;› <b>50,000 reviews</b><br>
    COLS &nbsp;› <b>review · sentiment</b><br>
    LBLS &nbsp;› <b>positive / negative</b><br><br>
    <small style="color:#1e3a4a">kaggle.com/datasets/lakshmi25npathi/<br>imdb-dataset-of-50k-movie-reviews</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info" style="border-left-color:#f5ff00">
    <b style="color:#f5ff00">// ACTIVE MODULES</b><br><br>
    <span style="color:#00ffe5">▸</span> 3-Model Neural Comparison<br>
    <span style="color:#00ffe5">▸</span> Sentiment Intensity Grid<br>
    <span style="color:#00ffe5">▸</span> Review Length Analysis<br>
    <span style="color:#00ffe5">▸</span> Text Preprocessing Viz
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// LR ◈ NAIVE BAYES ◈ SVM &nbsp;·&nbsp; INTENSITY MATRIX &nbsp;·&nbsp; CORPUS ANALYSIS &nbsp;·&nbsp; PREPROCESSING NODE</div>', unsafe_allow_html=True)

if uploaded:
    tmp_path = os.path.join(tempfile.gettempdir(), "imdb_dataset.csv")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    with st.spinner("⚡ Initializing neural grid... training 3 models on 50,000 reviews"):
        tfidf, results, best_model_name, df = train_all_models(tmp_path)

    st.success(f"✅ GRID ONLINE — {len(df):,} reviews processed · DOMINANT ALGORITHM: {best_model_name}")

    # ════════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "▸ MODEL MATRIX",
        "▸ ANALYZE NODE",
        "▸ LENGTH GRID",
        "▸ PREPROCESS"
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — MODEL COMPARISON
    # ════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
        <div style="font-family:'Orbitron',monospace;font-size:0.85rem;
                    color:#00ffe5;letter-spacing:0.1em;margin-bottom:1.2rem">
            // MODEL PERFORMANCE MATRIX &nbsp;·&nbsp; LR vs NAIVE BAYES vs SVM
        </div>""", unsafe_allow_html=True)

        model_names  = list(results.keys())
        accs         = [results[m]['acc']  for m in model_names]
        f1s          = [results[m]['f1']   for m in model_names]
        times        = [results[m]['time'] for m in model_names]

        cols = st.columns(3)
        model_colors = {
            'Logistic Regression': '#00ffe5',
            'Naive Bayes':         '#ff2d78',
            'SVM':                 '#f5ff00'
        }
        for i, mname in enumerate(model_names):
            with cols[i]:
                color = model_colors[mname]
                crown = " ◈ DOMINANT" if mname == best_model_name else ""
                st.markdown(f"""
                <div class="metric-card" style="border-color:{color}35;border-top:2px solid {color}">
                    <div style="font-family:'Orbitron',monospace;font-size:0.6rem;
                                color:{color};letter-spacing:0.12em;margin-bottom:0.3rem">
                        {mname}{crown}</div>
                    <div style="font-family:'Orbitron',monospace;font-size:2rem;
                                font-weight:900;color:{color};
                                text-shadow:0 0 16px {color}99">
                        {results[mname]['acc']:.1%}</div>
                    <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                                color:#4a7a8a;letter-spacing:0.12em">ACCURACY</div>
                    <div style="border-top:1px solid #0d2233;margin:0.6rem 0"></div>
                    <div style="display:flex;justify-content:space-between;
                                font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:#4a7a8a">
                        <span>F1 <b style="color:{color}">{results[mname]['f1']:.3f}</b></span>
                        <span>⏱ {results[mname]['time']}s</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                color:#4a7a8a;letter-spacing:0.12em;margin-bottom:0.4rem">
                // ACCURACY & F1 SIGNAL</div>""", unsafe_allow_html=True)
            fig, ax = cyber_fig(5, 3.5)
            x     = np.arange(len(model_names))
            width = 0.35
            bar_cols = [CYAN, PINK, YELLOW]
            bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',
                           color=bar_cols, alpha=0.9, edgecolor='none')
            bars2 = ax.bar(x + width/2, f1s, width, label='F1 Score',
                           color=bar_cols, alpha=0.35, edgecolor='none')
            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                        f'{bar.get_height():.3f}', ha='center', color=CYAN, fontsize=7,
                        fontfamily='monospace')
            ax.set_xticks(x)
            ax.set_xticklabels(['LR', 'NB', 'SVM'], color=TEXT_DIM, fontsize=9)
            ax.set_ylim(0.8, 1.01)
            ax.spines[['top','right']].set_visible(False)
            legend = ax.legend(fontsize=7, labelcolor=TEXT_DIM,
                               facecolor=CYBER_BG, edgecolor=BORDER)
            ax.set_ylabel('Score', color=TEXT_DIM, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

        with col_r:
            st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                color:#4a7a8a;letter-spacing:0.12em;margin-bottom:0.4rem">
                // TRAINING CYCLE TIME</div>""", unsafe_allow_html=True)
            fig2, ax2 = cyber_fig(5, 3.5)
            bars = ax2.barh(model_names, times,
                            color=[CYAN, PINK, YELLOW], edgecolor='none', height=0.45)
            for bar in bars:
                ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                         f'{bar.get_width()}s', va='center', color=TEXT_DIM,
                         fontsize=8, fontfamily='monospace')
            ax2.tick_params(colors=TEXT_DIM, labelsize=8)
            ax2.spines[['top','right','bottom']].set_visible(False)
            ax2.xaxis.set_visible(False)
            for spine in ax2.spines.values():
                spine.set_color(BORDER)
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
            color:#4a7a8a;letter-spacing:0.12em;margin:1rem 0 0.4rem">
            // CONFUSION MATRICES</div>""", unsafe_allow_html=True)
        cm_cols = st.columns(3)
        cm_cmaps = ['Blues', 'RdPu', 'YlOrBr']  # per model color theme
        for i, mname in enumerate(model_names):
            with cm_cols[i]:
                st.markdown(f"<div style='font-family:Orbitron,monospace;font-size:0.65rem;"
                            f"color:{list(model_colors.values())[i]};letter-spacing:0.1em;"
                            f"margin-bottom:0.3rem'>{mname}</div>", unsafe_allow_html=True)
                fig3, ax3 = plt.subplots(figsize=(3.2, 2.8))
                fig3.patch.set_facecolor(CYBER_BG)
                ax3.set_facecolor(CYBER_PANEL)
                sns.heatmap(results[mname]['cm'], annot=True, fmt='d',
                            cmap=cm_cmaps[i],
                            xticklabels=['NEG','POS'], yticklabels=['NEG','POS'],
                            ax=ax3, linewidths=1, linecolor='#030810',
                            annot_kws={"size": 11, "weight": "bold",
                                       "color": "white", "family": "monospace"})
                ax3.tick_params(colors=TEXT_DIM, labelsize=8)
                ax3.set_xlabel('PREDICTED', color=TEXT_DIM, fontsize=7,
                               fontfamily='monospace', labelpad=6)
                ax3.set_ylabel('ACTUAL',    color=TEXT_DIM, fontsize=7,
                               fontfamily='monospace', labelpad=6)
                plt.tight_layout()
                st.pyplot(fig3)

        best = results[best_model_name]
        st.markdown(f"""
        <div class="model-winner">
            <div style="font-size:1.4rem">◈</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;
                        color:#f5ff00;font-weight:700;letter-spacing:0.1em;
                        text-shadow:0 0 12px rgba(245,255,0,0.6)">
                {best_model_name.upper()} // DOMINANT ALGORITHM</div>
            <div style="font-family:'Share Tech Mono',monospace;color:#4a7a8a;
                        font-size:0.78rem;margin-top:0.4rem;letter-spacing:0.08em">
                ACCURACY <b style="color:#f5ff00">{best['acc']:.2%}</b> &nbsp;·&nbsp;
                F1 <b style="color:#f5ff00">{best['f1']:.4f}</b> &nbsp;·&nbsp;
                CYCLE <b style="color:#f5ff00">{best['time']}s</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # TAB 2 — ANALYZE REVIEW
    # ════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:0.85rem;
            color:#00ffe5;letter-spacing:0.1em;margin-bottom:1.2rem">
            // SENTIMENT ANALYSIS NODE</div>""", unsafe_allow_html=True)

        selected_model = st.selectbox(
            "ALGORITHM SELECT:",
            options=list(results.keys()),
            index=list(results.keys()).index(best_model_name)
        )

        review_input = st.text_area(
            "INPUT STREAM:",
            height=130,
            placeholder="// pipe review text into the sentiment grid...\ne.g. This film was an absolute masterpiece — the acting, the score, everything was perfect...",
            label_visibility="visible"
        )

        if st.button("⚡ EXECUTE ANALYSIS"):
            if review_input.strip():
                model = results[selected_model]['model']
                pred, prob = predict_with_model(review_input, tfidf, model)
                pos_prob   = prob[1]
                neg_prob   = prob[0]
                intensity  = get_intensity(pos_prob)
                i_label    = intensity_label(intensity)
                i_color    = intensity_color(intensity)

                col_res, col_int = st.columns(2)

                with col_res:
                    if pred == 1:
                        st.markdown(f"""
                        <div class="positive-result">
                            <div class="result-emoji">▲</div>
                            <div class="result-label" style="color:#00ffe5;
                                text-shadow:0 0 18px rgba(0,255,229,0.8)">POSITIVE SIGNAL</div>
                            <div class="result-conf">ALGORITHM: {selected_model}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="negative-result">
                            <div class="result-emoji">▼</div>
                            <div class="result-label" style="color:#ff2d78;
                                text-shadow:0 0 18px rgba(255,45,120,0.8)">NEGATIVE SIGNAL</div>
                            <div class="result-conf">ALGORITHM: {selected_model}</div>
                        </div>""", unsafe_allow_html=True)

                with col_int:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color:{i_color}35;border-top:2px solid {i_color}">
                        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;
                                    color:#4a7a8a;letter-spacing:0.15em">
                            ◈ INTENSITY INDEX</div>
                        <div style="font-family:'Orbitron',monospace;font-size:2.8rem;
                                    font-weight:900;color:{i_color};
                                    text-shadow:0 0 18px {i_color}99">
                            {intensity}</div>
                        <div style="font-family:'Share Tech Mono',monospace;
                                    color:#1e3a4a;font-size:0.75rem">/ 100</div>
                        <div style="background:#030810;border-radius:0;
                                    height:8px;margin:0.8rem 0;overflow:hidden;
                                    border:1px solid {i_color}25">
                            <div style="width:{intensity}%;height:100%;
                                        background:linear-gradient(90deg,{i_color}55,{i_color});
                                        box-shadow:0 0 8px {i_color}88"></div>
                        </div>
                        <div style="font-family:'Orbitron',monospace;color:{i_color};
                                    font-weight:700;font-size:0.75rem;letter-spacing:0.1em">
                            {i_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""<div style="font-family:'Share Tech Mono',monospace;
                    font-size:0.72rem;color:#4a7a8a;letter-spacing:0.12em">
                    // PROBABILITY CHANNELS</div>""", unsafe_allow_html=True)
                st.markdown(f"<span style='font-family:monospace;color:#00ffe5'>▲ POSITIVE</span> — **{pos_prob:.1%}**")
                st.progress(float(pos_prob))
                st.markdown(f"<span style='font-family:monospace;color:#ff2d78'>▼ NEGATIVE</span> — **{neg_prob:.1%}**")
                st.progress(float(neg_prob))

                st.markdown("---")
                st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:0.75rem;
                    color:#00ffe5;letter-spacing:0.1em;margin-bottom:0.8rem">
                    // ALL ALGORITHMS ON THIS INPUT</div>""", unsafe_allow_html=True)
                m_cols = st.columns(3)
                for i, mname in enumerate(results.keys()):
                    m = results[mname]['model']
                    p, pb = predict_with_model(review_input, tfidf, m)
                    inten = get_intensity(pb[1])
                    col   = "#00ffe5" if p == 1 else "#ff2d78"
                    arrow = "▲" if p == 1 else "▼"
                    with m_cols[i]:
                        st.markdown(f"""
                        <div class="metric-card" style="border-color:{col}25;border-top:2px solid {col}">
                            <div style="font-family:'Orbitron',monospace;font-size:0.6rem;
                                        color:#4a7a8a;letter-spacing:0.1em">{mname}</div>
                            <div style="font-size:1.8rem;color:{col};
                                        text-shadow:0 0 12px {col}88">{arrow}</div>
                            <div style="font-family:'Orbitron',monospace;color:{col};
                                        font-weight:700;font-size:0.85rem;letter-spacing:0.1em">
                                {"POSITIVE" if p==1 else "NEGATIVE"}</div>
                            <div style="font-family:'Share Tech Mono',monospace;
                                        color:#1e3a4a;font-size:0.75rem">
                                IDX <b style="color:{col}">{inten}/100</b></div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.warning("// NULL INPUT — pipe review text into stream")

        st.markdown("---")
        st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:0.75rem;
            color:#4a7a8a;letter-spacing:0.1em;margin-bottom:0.8rem">
            // SAMPLE INPUT NODES</div>""", unsafe_allow_html=True)
        samples = [
            "An absolute masterpiece! The performances were breathtaking and the story deeply moving.",
            "Terrible. Boring script, wooden acting, and a plot that goes absolutely nowhere.",
            "Pretty average film. Some good moments but overall forgettable.",
            "One of the best films I've ever seen! Pure cinematic gold from start to finish.",
        ]
        s_cols = st.columns(2)
        for i, sample in enumerate(samples):
            with s_cols[i % 2]:
                if st.button(f"▸ NODE {i+1:02d}", key=f"s_{i}", use_container_width=True):
                    model = results[selected_model]['model']
                    pred, prob = predict_with_model(sample, tfidf, model)
                    label = "▲ POSITIVE" if pred == 1 else "▼ NEGATIVE"
                    inten = get_intensity(prob[1])
                    st.info(f"**{label}** · IDX: **{inten}/100**\n\n`{sample[:80]}...`")

    # ════════════════════════════════════════════════════════════
    # TAB 3 — REVIEW LENGTH ANALYSIS
    # ════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:0.85rem;
            color:#00ffe5;letter-spacing:0.1em;margin-bottom:1.2rem">
            // CORPUS LENGTH ANALYSIS GRID</div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                color:#4a7a8a;letter-spacing:0.12em;margin-bottom:0.4rem">
                // AVG TOKEN COUNT BY CLASS</div>""", unsafe_allow_html=True)
            avg_len = df.groupby('sentiment')['review_length'].mean()
            fig, ax = cyber_fig(5, 3.5)
            colors = [CYAN if s == 'positive' else PINK for s in avg_len.index]
            bars = ax.bar(avg_len.index, avg_len.values,
                          color=colors, edgecolor='none', width=0.45)
            for bar, clr in zip(bars, colors):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                        f'{bar.get_height():.0f}',
                        ha='center', color=clr, fontsize=9,
                        fontfamily='monospace',
                        fontweight='bold')
            ax.tick_params(colors=TEXT_DIM, labelsize=9)
            ax.spines[['top','right','left']].set_visible(False)
            ax.spines['bottom'].set_color(BORDER)
            ax.yaxis.set_visible(False)
            ax.set_xlabel('SENTIMENT CLASS', color=TEXT_DIM, fontsize=8,
                          fontfamily='monospace', labelpad=8)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                color:#4a7a8a;letter-spacing:0.12em;margin-bottom:0.4rem">
                // LENGTH DISTRIBUTION SIGNAL</div>""", unsafe_allow_html=True)
            fig2, ax2 = cyber_fig(5, 3.5)
            pos_lens = df[df['sentiment'].str.lower() == 'positive']['review_length']
            neg_lens = df[df['sentiment'].str.lower() == 'negative']['review_length']
            ax2.hist(pos_lens.clip(upper=600), bins=40, alpha=0.55,
                     color=CYAN, label='POSITIVE', edgecolor='none')
            ax2.hist(neg_lens.clip(upper=600), bins=40, alpha=0.55,
                     color=PINK, label='NEGATIVE', edgecolor='none')
            ax2.tick_params(colors=TEXT_DIM, labelsize=8)
            ax2.spines[['top','right']].set_visible(False)
            ax2.spines[['left','bottom']].set_color(BORDER)
            ax2.set_xlabel('TOKENS (WORDS)', color=TEXT_DIM, fontsize=8,
                           fontfamily='monospace')
            ax2.set_ylabel('FREQUENCY', color=TEXT_DIM, fontsize=8,
                           fontfamily='monospace')
            leg = ax2.legend(fontsize=7, labelcolor=TEXT_DIM,
                             facecolor=CYBER_BG, edgecolor=BORDER)
            for t in leg.get_texts():
                t.set_fontfamily('monospace')
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
            color:#4a7a8a;letter-spacing:0.12em;margin:1rem 0 0.4rem">
            // STATISTICAL BREAKDOWN</div>""", unsafe_allow_html=True)
        stats = df.groupby('sentiment')['review_length'].describe()[['mean','min','50%','max']]
        stats.columns = ['MEAN', 'MIN', 'MEDIAN', 'MAX']
        stats = stats.round(1)
        st.dataframe(stats, use_container_width=True)

        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
            color:#4a7a8a;letter-spacing:0.12em;margin:1rem 0 0.4rem">
            // LENGTH vs CONFIDENCE SCATTER (n=500)</div>""", unsafe_allow_html=True)
        sample_df = df.sample(500, random_state=42).copy()
        model = results[best_model_name]['model']
        vecs  = tfidf.transform(sample_df['clean'])
        probs = model.predict_proba(vecs)[:, 1]
        sample_df['confidence'] = probs
        sample_df['color']      = sample_df['sentiment'].str.lower().map(
            {'positive': CYAN, 'negative': PINK}
        )

        fig3, ax3 = cyber_fig(10, 3.5)
        ax3.scatter(sample_df['review_length'].clip(upper=600),
                    sample_df['confidence'],
                    c=sample_df['color'], alpha=0.45, s=12, edgecolors='none')
        ax3.axhline(0.5, color=YELLOW, linestyle='--', linewidth=0.8, alpha=0.6)
        ax3.tick_params(colors=TEXT_DIM, labelsize=8)
        ax3.spines[['top','right']].set_visible(False)
        ax3.spines[['left','bottom']].set_color(BORDER)
        ax3.set_xlabel('REVIEW LENGTH (TOKENS)', color=TEXT_DIM, fontsize=8,
                       fontfamily='monospace')
        ax3.set_ylabel('POSITIVE CONFIDENCE', color=TEXT_DIM, fontsize=8,
                       fontfamily='monospace')
        pos_patch = mpatches.Patch(color=CYAN, label='POSITIVE')
        neg_patch = mpatches.Patch(color=PINK, label='NEGATIVE')
        leg3 = ax3.legend(handles=[pos_patch, neg_patch], fontsize=7,
                          labelcolor=TEXT_DIM, facecolor=CYBER_BG, edgecolor=BORDER)
        for t in leg3.get_texts():
            t.set_fontfamily('monospace')
        plt.tight_layout()
        st.pyplot(fig3)

    # ════════════════════════════════════════════════════════════
    # TAB 4 — BEFORE / AFTER TEXT CLEANING
    # ════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:0.85rem;
            color:#00ffe5;letter-spacing:0.1em;margin-bottom:0.4rem">
            // TEXT PREPROCESSING PIPELINE</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
            color:#4a7a8a;letter-spacing:0.1em;margin-bottom:1.2rem">
            Visualize the NLP cleaning stages on raw input</div>""", unsafe_allow_html=True)

        clean_input = st.text_area(
            "RAW INPUT STREAM:",
            value='<b>Wow!!!</b> This movie was GREAT... or was it?? I\'m not 100% sure 😅 #cinema <br>',
            height=100,
            label_visibility="visible"
        )

        if clean_input:
            cleaned = clean_text(clean_input)
            word_count_before = len(clean_input.split())
            word_count_after  = len(cleaned.split()) if cleaned else 0

            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                    color:#ff2d78;letter-spacing:0.12em;margin-bottom:0.3rem">
                    ▼ PRE-PROCESS // RAW</div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="clean-box" style="color:#ff2d78;'
                            f'border-left-color:#ff2d78;text-shadow:0 0 6px rgba(255,45,120,0.4)">'
                            f'{clean_input}</div>', unsafe_allow_html=True)
                st.caption(f"CHARS: {len(clean_input)} · TOKENS: {word_count_before}")

            with col_a:
                st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;
                    color:#00ffe5;letter-spacing:0.12em;margin-bottom:0.3rem">
                    ▲ POST-PROCESS // CLEAN</div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="clean-box">{cleaned}</div>', unsafe_allow_html=True)
                st.caption(f"CHARS: {len(cleaned)} · TOKENS: {word_count_after}")

            st.markdown("---")
            st.markdown("""<div style="font-family:'Orbitron',monospace;font-size:0.75rem;
                color:#00ffe5;letter-spacing:0.1em;margin-bottom:0.8rem">
                // PIPELINE STAGES</div>""", unsafe_allow_html=True)

            stages = [
                ("01", "HTML STRIP",       "Removed `<b>`, `<br>` and all HTML markup tags"),
                ("02", "SPECIAL CHARS",    "Purged `!!!`, `??`, `100%`, `😅`, `#`, apostrophes"),
                ("03", "LOWERCASE NORM",   "All chars converted to lowercase encoding"),
                ("04", "WHITESPACE TRIM",  "Leading/trailing whitespace removed"),
            ]

            for num, title, desc in stages:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:1rem;
                            margin-bottom:0.6rem;padding:0.6rem;
                            background:#040c14;border:1px solid #0d2233;
                            border-left:2px solid #00ffe5;border-radius:2px">
                    <span style="font-family:'Orbitron',monospace;font-size:0.65rem;
                                 color:#00ffe5;opacity:0.5;min-width:24px">{num}</span>
                    <span style="font-family:'Orbitron',monospace;font-size:0.7rem;
                                 color:#00ffe5;min-width:140px;letter-spacing:0.08em">{title}</span>
                    <span style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;
                                 color:#4a7a8a">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                color:#4a7a8a;letter-spacing:0.12em;margin-bottom:0.5rem">
                // CORPUS SAMPLE — PRE vs POST</div>""", unsafe_allow_html=True)
            sample_rows = df.sample(5, random_state=7)[['review', 'clean', 'sentiment']].copy()
            sample_rows.columns = ['RAW REVIEW', 'CLEANED TEXT', 'CLASS']
            sample_rows['RAW REVIEW']   = sample_rows['RAW REVIEW'].str[:120] + '...'
            sample_rows['CLEANED TEXT'] = sample_rows['CLEANED TEXT'].str[:120] + '...'
            st.dataframe(sample_rows, use_container_width=True)

else:
    # ── Upload prompt ─────────────────────────────────────────
    st.markdown("""
    <div class="card" style="text-align:center; padding:4rem 2rem">
        <div style="font-family:'Orbitron',monospace;font-size:3rem;
                    color:#00ffe5;text-shadow:0 0 30px rgba(0,255,229,0.6)">◈</div>
        <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;
                    color:#00ffe5;letter-spacing:0.15em;margin:1rem 0 0.5rem;
                    text-shadow:0 0 15px rgba(0,255,229,0.5)">
            AWAITING DATA UPLINK</div>
        <div style="font-family:'Share Tech Mono',monospace;color:#4a7a8a;
                    max-width:480px;margin:0 auto;font-size:0.85rem;letter-spacing:0.05em;
                    line-height:1.8">
            Upload <b style="color:#00ffe5">IMDB Dataset.csv</b> (50,000 reviews)<br>
            via sidebar uplink to initialize the neural grid
        </div>
        <div style="display:flex;justify-content:center;gap:2rem;
                    flex-wrap:wrap;margin-top:1.8rem">
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                         color:#00ffe5">▸ MODEL MATRIX</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                         color:#ff2d78">▸ INTENSITY INDEX</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                         color:#f5ff00">▸ LENGTH GRID</span>
            <span style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                         color:#bf00ff">▸ PREPROCESS NODE</span>
        </div>
        <div style="font-family:'Share Tech Mono',monospace;color:#1e3a4a;
                    font-size:0.78rem;margin-top:1.5rem">
            // kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
        </div>
    </div>
    """, unsafe_allow_html=True)