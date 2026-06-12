import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from collections import Counter
import requests
import json

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    st.error("Library 'wordcloud' atau 'matplotlib' belum terinstal.")
    st.stop()

# =====================================================================
# 1. KONFIGURASI — DARK MODE
# =====================================================================
st.set_page_config(
    page_title="Executive CX Analytics - Bank XYZ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ── Dark Mode Base ── */
.stApp { background-color: #0F172A !important; }
.main .block-container { padding-top: 1rem; }

/* ── Sidebar Dark ── */
[data-testid="stSidebar"] {
    background-color: #1E293B !important;
    border-right: 1px solid #334155 !important;
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label { color: #94A3B8 !important; font-size: 12px; }

/* ── Semua teks global ── */
*, p, span, div, label, h1, h2, h3, h4, h5, h6 {
    color: #E2E8F0 !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #1E293B !important;
    border-radius: 10px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border: 1px solid #334155 !important;
    border-radius: 8px; padding: 8px 16px;
    color: #94A3B8 !important; font-weight: 600; font-size: 12px;
}
.stTabs [aria-selected="true"] {
    background-color: #3B82F6 !important;
    color: #FFFFFF !important; border-color: #3B82F6 !important;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(145deg, #1E293B, #0F172A);
    border: 1px solid #334155; padding: 16px; border-radius: 14px;
    text-align: center; transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card:hover {
    transform: translateY(-3px); border-color: #3B82F6;
    box-shadow: 0 8px 25px rgba(59,130,246,0.2);
}
.metric-title {
    color: #64748B !important; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;
}
.metric-value { font-size: 28px; font-weight: 900; line-height: 1; }
.metric-value.blue  { color: #60A5FA !important; }
.metric-value.green { color: #34D399 !important; }
.metric-value.red   { color: #F87171 !important; }
.metric-value.amber { color: #FBBF24 !important; }
.metric-value.white { color: #F1F5F9 !important; }
.metric-sub { color: #64748B !important; font-size: 10px; margin-top: 4px; }

/* ── Section Headers ── */
.section-header {
    font-size: 14px; font-weight: 800; color: #F1F5F9 !important;
    border-left: 4px solid #3B82F6; padding-left: 12px;
    margin: 16px 0 10px 0; letter-spacing: 0.3px;
}

/* ── Insight Boxes ── */
.insight-box {
    background: linear-gradient(135deg, #1E3A5F, #1E293B);
    border: 1px solid #3B82F6; border-radius: 10px;
    padding: 12px 16px; margin: 6px 0;
}
.insight-box p { color: #93C5FD !important; font-size: 12px; margin: 0; }

/* ── AI Chat ── */
.chat-container {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 12px; padding: 16px; margin: 8px 0;
    max-height: 400px; overflow-y: auto;
}
.chat-user {
    background: #1D4ED8; border-radius: 12px 12px 4px 12px;
    padding: 10px 14px; margin: 6px 0 6px auto;
    max-width: 80%; text-align: right;
    display: inline-block; float: right; clear: both;
}
.chat-user p { color: #DBEAFE !important; font-size: 13px; margin: 0; }
.chat-ai {
    background: #0F3460; border: 1px solid #1E40AF;
    border-radius: 12px 12px 12px 4px;
    padding: 10px 14px; margin: 6px 0;
    max-width: 85%; display: inline-block; float: left; clear: both;
}
.chat-ai p { color: #BAE6FD !important; font-size: 13px; margin: 0; }
.chat-clearfix { clear: both; }

/* ── Dataframe dark ── */
.stDataFrame { background: #1E293B !important; }
[data-testid="stDataFrame"] { background: #1E293B !important; }

/* ── Input fields ── */
.stTextInput input, .stSelectbox select {
    background: #1E293B !important; color: #E2E8F0 !important;
    border: 1px solid #334155 !important; border-radius: 8px;
}
.stTextArea textarea {
    background: #1E293B !important; color: #E2E8F0 !important;
    border: 1px solid #334155 !important;
}

/* ── Buttons ── */
.stButton button {
    background: #1D4ED8 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px;
    font-weight: 600; transition: all 0.2s;
}
.stButton button:hover {
    background: #2563EB !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4);
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1E293B !important; color: #E2E8F0 !important;
    border: 1px solid #334155 !important; border-radius: 8px;
}

/* ── Success/Info/Warning ── */
.stSuccess { background: rgba(16,185,129,0.15) !important; border-color: #10B981 !important; }
.stInfo    { background: rgba(59,130,246,0.15) !important; border-color: #3B82F6 !important; }
.stWarning { background: rgba(245,158,11,0.15) !important; border-color: #F59E0B !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }
</style>
""", unsafe_allow_html=True)

# ── Warna konstanta ────────────────────────────────────────────────────
NPS_COLORS  = {'Promoter':'#10B981','Passive':'#F59E0B','Detractor':'#EF4444'}
COLOR_XYZ   = '#3B82F6'
COLOR_KOMP  = '#F87171'
COLOR_IMPRT = '#64748B'
CHART_BG    = '#0F172A'
CHART_PAPER = '#0F172A'
GRID_COLOR  = '#1E293B'

def fmt_card(title, value, color="white", sub=""):
    return f"""<div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value {color}'>{value}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>"""

def insight_box(text):
    st.markdown(f"<div class='insight-box'><p>💡 {text}</p></div>",
                unsafe_allow_html=True)

def elite_layout(fig, title="", height=None):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#E2E8F0')),
        template="plotly_dark",
        plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER,
        margin=dict(t=45, b=15, l=10, r=10),
        font=dict(color='#CBD5E1', size=11),
        hoverlabel=dict(bgcolor="#1E293B", font_color="#E2E8F0",
                        font_size=12, bordercolor="#3B82F6"),
        legend=dict(
            font=dict(color="#CBD5E1"),
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="#334155",
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        **({"height": height} if height else {})
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     automargin=True, tickfont=dict(color='#94A3B8'),
                     title_font=dict(color='#94A3B8'))
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     automargin=True, tickfont=dict(color='#94A3B8'),
                     title_font=dict(color='#94A3B8'))
    return fig

def short_label(col, col_map, max_len=38):
    name = col_map.get(col, col)
    name = re.sub(r'\s*-\s*(XYZ|kompetitor)\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(.*?\)', '', name).strip()
    name = re.sub(r'\s+', ' ', name)
    return name[:max_len] + "…" if len(name) > max_len else name

# ── Parser komentar VoC ────────────────────────────────────────────────
def parse_comment(raw):
    """Ambil bagian akhir komentar yang paling spesifik."""
    if pd.isna(raw) or str(raw).strip() in ['', 'None', 'nan']:
        return None
    parts = str(raw).split(';')
    # Ambil item terakhir yang panjang > 5 karakter
    for p in reversed(parts):
        p = p.strip()
        if len(p) > 5 and p not in ['NET', 'Subnet']:
            return p
    return parts[-1].strip()

def clean_comments(series):
    return series.apply(parse_comment).dropna()

# =====================================================================
# 2. LOAD DATA
# =====================================================================
@st.cache_data
def load_data():
    df      = pd.read_csv('data/df_clean.csv', low_memory=False)
    col_map = pd.read_csv('data/col_mapping.csv').set_index('kode')['nama_panjang'].to_dict()
    summary = pd.read_csv('data/summary_scores.csv')
    return df, col_map, summary

try:
    df_raw, col_map, summary_df = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# =====================================================================
# 3. SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(
        "<h3 style='color:#60A5FA !important; font-weight:800; "
        "margin-bottom:16px;'>🎛️ Filter Analitik</h3>",
        unsafe_allow_html=True
    )

    # Kompetitor
    st.markdown("**🏦 Benchmark Kompetitor**")
    komp_list   = sorted(df_raw['KOMP'].dropna().unique().tolist())
    target_komp = st.selectbox(
        "Kompetitor:",
        ["Semua Kompetitor (Rata-rata)"] + komp_list,
        help="Pilih bank kompetitor spesifik"
    )

    st.markdown("---")

    # Lokasi Cascading
    st.markdown("**📍 Lokasi**")
    sel_prov = st.multiselect(
        "Provinsi", sorted(df_raw['PROV'].dropna().unique())
    )
    kota_pool = df_raw[df_raw['PROV'].isin(sel_prov)] if sel_prov else df_raw
    sel_kota  = st.multiselect(
        "Kab/Kota", sorted(kota_pool['KABKOTA'].dropna().unique())
    )
    cab_pool  = kota_pool[kota_pool['KABKOTA'].isin(sel_kota)] if sel_kota else kota_pool
    sel_cab   = st.multiselect(
        "Cabang", sorted(cab_pool['CABANG'].dropna().unique())
    )

    st.markdown("---")

    # Profil Responden
    with st.expander("👤 Profil Responden", expanded=False):
        sel_gender    = st.multiselect("Gender",
            sorted(df_raw['S1'].dropna().unique()))
        sel_usia      = st.multiselect("Rentang Usia",
            sorted(df_raw['S2_2'].dropna().unique()))
        sel_tenure    = st.multiselect("Lama Nasabah",
            sorted(df_raw['S4'].dropna().unique()))
        sel_frek      = st.multiselect("Frekuensi Transaksi",
            sorted(df_raw['S7'].dropna().unique()))
        sel_panel     = st.multiselect("Panel",
            sorted(df_raw['PANEL'].dropna().unique()))
        sel_pekerjaan = st.multiselect("Pekerjaan",
            sorted(df_raw['P4'].dropna().unique()))
        sel_pendidikan= st.multiselect("Pendidikan",
            sorted(df_raw['P3'].dropna().unique()))
        sel_p1        = st.multiselect("Status Nikah",
            sorted(df_raw['P1'].dropna().unique()))

    with st.expander("🏦 Perilaku Perbankan", expanded=False):
        sel_bank_simpan = st.multiselect("Bank Utama Simpan",
            sorted(df_raw['A1B'].dropna().unique()))
        sel_bank_trans  = st.multiselect("Bank Utama Transaksi",
            sorted(df_raw['A1C'].dropna().unique()))
        sel_nps_cat     = st.multiselect("Kategori NPS",
            ['Promoter','Passive','Detractor'])

    with st.expander("🎯 Filter Skor", expanded=False):
        nps_range = st.slider("Rentang NPS Score", 0, 10, (0, 10))
        sat_range = st.slider("Rentang Kepuasan", 1, 6, (1, 6))
        loy_range = st.slider("Rentang Loyalitas", 1, 6, (1, 6))

    st.markdown("---")

# ── Apply filter ──────────────────────────────────────────────────────
df = df_raw.copy()
if sel_prov:         df = df[df['PROV'].isin(sel_prov)]
if sel_kota:         df = df[df['KABKOTA'].isin(sel_kota)]
if sel_cab:          df = df[df['CABANG'].isin(sel_cab)]
if sel_gender:       df = df[df['S1'].isin(sel_gender)]
if sel_usia:         df = df[df['S2_2'].isin(sel_usia)]
if sel_tenure:       df = df[df['S4'].isin(sel_tenure)]
if sel_frek:         df = df[df['S7'].isin(sel_frek)]
if sel_panel:        df = df[df['PANEL'].isin(sel_panel)]
if sel_pekerjaan:    df = df[df['P4'].isin(sel_pekerjaan)]
if sel_pendidikan:   df = df[df['P3'].isin(sel_pendidikan)]
if sel_p1:           df = df[df['P1'].isin(sel_p1)]
if sel_bank_simpan:  df = df[df['A1B'].isin(sel_bank_simpan)]
if sel_bank_trans:   df = df[df['A1C'].isin(sel_bank_trans)]
if sel_nps_cat:      df = df[df['G1A_CAT'].isin(sel_nps_cat)]
df = df[df['G1A'].between(nps_range[0], nps_range[1])]
df = df[df['E1A'].between(sat_range[0], sat_range[1])]
df = df[df['F1A'].between(loy_range[0], loy_range[1])]

df_komp     = df.copy() if target_komp == "Semua Kompetitor (Rata-rata)" \
              else df[df['KOMP'] == target_komp]
df_has_komp = df_komp[df_komp['KOMP'].notna()]

with st.sidebar:
    st.success(f"📊 Responden: **{len(df):,}**")
    st.info(f"🏦 Dgn Kompetitor: **{len(df_has_komp):,}**")
    if len(df) < 30:
        st.warning("⚠️ Sampel < 30, hasil mungkin tidak representatif.")

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 12px 0 20px 0;'>
    <h1 style='font-weight:900; letter-spacing:-1px; color:#F1F5F9 !important;
    font-size:26px; margin:0;'>
    🏦 BANK XYZ — EXECUTIVE CX INTELLIGENCE DASHBOARD
    </h1>
    <p style='color:#64748B !important; font-size:12px; margin:4px 0 0 0;'>
    Real-time Customer Experience Analytics Platform
    </p>
</div>""", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ Data kosong. Sesuaikan filter Anda.")
    st.stop()

# =====================================================================
# HELPERS
# =====================================================================
def calc_nps(series_cat):
    total = series_cat.notna().sum()
    if total == 0: return 0, 0, 0, 0
    prom  = (series_cat=='Promoter').sum()
    pasv  = (series_cat=='Passive').sum()
    detr  = (series_cat=='Detractor').sum()
    return round((prom-detr)/total*100,1), round(prom/total*100,1), \
           round(pasv/total*100,1), round(detr/total*100,1)

def get_dim_map(df_ref):
    return {
        "Kantor Cabang": {
            "imp":  [f"T_KC1_{i}" for i in range(1,36) if f"T_KC1_{i}" in df_ref.columns],
            "xyz":  [c for c in df_ref.columns if c.startswith("T_KC2_") and
                     c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116",
                                "T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and
                     int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df_ref.columns if c.startswith("T_KC2_") and
                     c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116",
                                "T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and
                     int(c.split("_")[-1]) % 3 == 0],
        },
        "Sekuriti": {
            "imp":  [f"T_SC1_{i}" for i in range(1,16) if f"T_SC1_{i}" in df_ref.columns],
            "xyz":  [c for c in df_ref.columns if c.startswith("T_SC2_") and
                     c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df_ref.columns if c.startswith("T_SC2_") and
                     c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Teller": {
            "imp":  [f"T_TL2_{i}" for i in range(1,20) if f"T_TL2_{i}" in df_ref.columns],
            "xyz":  [c for c in df_ref.columns if c.startswith("T_TL3_") and
                     c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df_ref.columns if c.startswith("T_TL3_") and
                     c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Customer Service": {
            "imp":  [f"T_CS2_{i}" for i in range(1,24) if f"T_CS2_{i}" in df_ref.columns],
            "xyz":  [c for c in df_ref.columns if c.startswith("T_CS3_") and
                     c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df_ref.columns if c.startswith("T_CS3_") and
                     c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Customer Advisor": {
            "imp":  [f"T_CA1_{i}" for i in range(1,20) if f"T_CA1_{i}" in df_ref.columns],
            "xyz":  [c for c in df_ref.columns if c.startswith("T_CA2_") and c != "T_CA2_20"],
            "komp": [],
        },
        "ATM": {
            "imp":  [f"T_AT2_{i}" for i in range(1,19) if f"T_AT2_{i}" in df_ref.columns],
            "xyz":  [c for c in df_ref.columns if c.startswith("T_AT3_") and
                     c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df_ref.columns if c.startswith("T_AT3_") and
                     c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1]) % 3 == 0],
        },
    }

DIMENSI_MAP = get_dim_map(df_raw)

stopwords_id = {
    'yang','untuk','dengan','pada','dari','sebagai','tidak','karena','sangat',
    'lebih','sudah','saya','bank','bisa','dan','di','ke','ini','itu','ada',
    'juga','net','subnet','positive','negative','comments','dalam','oleh',
    'akan','telah','dapat','kami','anda','nya','atau','jadi','baru','lagi',
    'saat','pernah','masih','serta','yaitu','namun','jika','agar','bagi',
    'atas','antara','setiap','para','mereka','kita','xyz','nasabah','layanan',
    'cabang','rekening','lainnya','none','lain','sudah','belum','karena',
    'baik','bagus','cukup','sudah','masih','sangat','sekali','paling','makin'
}

# =====================================================================
# AI CHAT HELPER
# =====================================================================
def build_context(df_ctx, df_komp_ctx):
    """Buat ringkasan konteks data untuk AI."""
    nps_s, pp, _, pd_ = calc_nps(df_ctx['G1A_CAT'])
    nps_k, pk, _, dk  = calc_nps(df_komp_ctx['G1C_CAT']) \
                         if len(df_komp_ctx) > 0 else (0,0,0,0)
    ovr_cols = [c for c in df_ctx.columns if c.startswith("OVR_") and "_XYZ" in c]
    ovr_means = {c.replace("OVR_","").replace("_XYZ",""): round(df_ctx[c].mean(),2)
                 for c in ovr_cols if c in df_ctx.columns}

    top_cab_col = [c for c in df_ctx.columns if c.startswith("OVR_") and "_XYZ" in c
                   and c not in ["OVR_KC_OPERASIONAL_XYZ","OVR_KC_PARKIR_XYZ",
                                  "OVR_KC_BANKINGHALL_XYZ","OVR_KC_TOILET_XYZ"]]
    cab_score = df_ctx.groupby('CABANG')[top_cab_col].mean().mean(axis=1) \
                if top_cab_col else pd.Series(dtype=float)

    top3  = cab_score.nlargest(3).index.tolist()  if len(cab_score) > 0 else []
    bot3  = cab_score.nsmallest(3).index.tolist() if len(cab_score) > 0 else []

    # Sample komentar Promoter & Detractor
    prom_comments = clean_comments(
        df_ctx[df_ctx['G1A_CAT']=='Promoter']['G1B']
    ).head(5).tolist()
    detr_comments = clean_comments(
        df_ctx[df_ctx['G1A_CAT']=='Detractor']['G1B']
    ).head(5).tolist()

    ctx = f"""
DATA RINGKASAN BANK XYZ (berdasarkan filter aktif):
- Total responden: {len(df_ctx):,}
- NPS Score XYZ: {nps_s} (Promoter {pp}%, Detractor {pd_}%)
- NPS Score Kompetitor: {nps_k} (Promoter {pk}%, Detractor {dk}%)
- Gap NPS: {round(nps_s - nps_k, 1)} poin
- Kepuasan XYZ (1-6): {round(df_ctx['E1A'].mean(), 2)}
- Loyalitas XYZ (1-6): {round(df_ctx['F1A'].mean(), 2)}
- Skor per dimensi: {json.dumps(ovr_means)}
- Top 3 cabang terbaik: {', '.join(top3)}
- Bottom 3 cabang: {', '.join(bot3)}
- Sample alasan Promoter: {prom_comments}
- Sample alasan Detractor: {detr_comments}
- Provinsi yang disurvei: {sorted(df_ctx['PROV'].dropna().unique().tolist())}
- Kompetitor yang dibandingkan: {target_komp}
"""
    return ctx

def call_claude_api(messages, context):
    """Panggil Anthropic API."""
    try:
        system_prompt = f"""Kamu adalah analis CX (Customer Experience) senior yang ahli dalam data perbankan. 
Kamu memiliki akses ke data survei kepuasan nasabah Bank XYZ. 
Jawab pertanyaan dengan ringkas, insightful, dan actionable dalam Bahasa Indonesia.
Gunakan data berikut sebagai konteks:

{context}

Berikan jawaban yang:
1. Berdasarkan data yang ada
2. Disertai angka spesifik
3. Ada rekomendasi actionable
4. Maksimal 3-4 paragraf pendek"""

        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": messages
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data['content'][0]['text']
        else:
            return f"Error API: {response.status_code}"
    except Exception as e:
        return f"Gagal menghubungi AI: {str(e)}"

# =====================================================================
# TABS
# =====================================================================
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9 = st.tabs([
    "🌟 Executive",
    "🏢 Kinerja Layanan",
    "🏆 Brand & Komp.",
    "🎯 Touchpoint",
    "💡 Emosi & Loyalitas",
    "📱 Digitalisasi",
    "👥 Profil & Segmen",
    "💬 Voice of Customer",
    "🤖 AI Analyst"
])

# =====================================================================
# TAB 1 — EXECUTIVE SUMMARY
# =====================================================================
with tab1:
    nps_score,pct_p,pct_pasv,pct_d = calc_nps(df['G1A_CAT'])
    nps_k,pct_pk,pct_pasvk,pct_dk  = calc_nps(df_has_komp['G1C_CAT']) \
                                       if len(df_has_komp)>0 else (0,0,0,0)
    sat_xyz = df['E1A'].mean()
    sat_kom = df_has_komp['E1B'].mean() if len(df_has_komp)>0 else np.nan
    loy_xyz = df['F1A'].mean()
    loy_kom = df_has_komp['F1B'].mean() if len(df_has_komp)>0 else np.nan
    gap_nps = nps_score - nps_k

    # ── KPI Row ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📌 Key Performance Indicators</div>",
                unsafe_allow_html=True)
    k = st.columns(8)
    k[0].markdown(fmt_card("NPS XYZ", f"{nps_score:.0f}", "blue",
        f"P:{pct_p:.0f}% D:{pct_d:.0f}%"), unsafe_allow_html=True)
    k[1].markdown(fmt_card("NPS Kompetitor", f"{nps_k:.0f}", "red",
        f"P:{pct_pk:.0f}% D:{pct_dk:.0f}%"), unsafe_allow_html=True)
    k[2].markdown(fmt_card("Gap NPS",
        f"+{gap_nps:.0f}" if gap_nps>=0 else f"{gap_nps:.0f}",
        "green" if gap_nps>=0 else "red", "XYZ − Komp"),
        unsafe_allow_html=True)
    k[3].markdown(fmt_card("Kepuasan XYZ", f"{sat_xyz:.2f}", "blue", "Skala 1–6"),
        unsafe_allow_html=True)
    k[4].markdown(fmt_card("Kepuasan Komp",
        f"{sat_kom:.2f}" if not np.isnan(sat_kom) else "N/A", "red", "Skala 1–6"),
        unsafe_allow_html=True)
    k[5].markdown(fmt_card("Loyalitas XYZ", f"{loy_xyz:.2f}", "green", "Skala 1–6"),
        unsafe_allow_html=True)
    k[6].markdown(fmt_card("Promoter XYZ", f"{pct_p:.0f}%", "green",
        f"{int(df['G1A_CAT'].eq('Promoter').sum())} orang"), unsafe_allow_html=True)
    k[7].markdown(fmt_card("Total Resp.", f"{len(df):,}", "white",
        f"{len(df_has_komp):,} dgn komp"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Auto Insight ─────────────────────────────────────────────────
    if gap_nps > 50:
        insight_box(f"XYZ unggul sangat signifikan dengan gap NPS {gap_nps:.0f} poin. "
                    f"Promoter XYZ ({pct_p:.0f}%) jauh lebih tinggi dari kompetitor ({pct_pk:.0f}%). "
                    f"Pertahankan keunggulan di semua dimensi layanan.")
    elif gap_nps > 20:
        insight_box(f"XYZ unggul {gap_nps:.0f} poin NPS di atas kompetitor. "
                    f"Fokus pada dimensi dengan gap terkecil untuk memperlebar keunggulan.")
    else:
        insight_box(f"Gap NPS XYZ vs kompetitor hanya {gap_nps:.0f} poin — persaingan ketat. "
                    f"Perlu strategi diferensiasi yang lebih kuat di touchpoint utama.")

    # ── Row 2: Donut + Scorecard ──────────────────────────────────────
    r2a, r2b = st.columns([1, 2.3])

    with r2a:
        st.markdown("<div class='section-header'>NPS Composition</div>",
                    unsafe_allow_html=True)
        for cat_data, title_d, nps_val in [
            (df['G1A_CAT'], "Bank XYZ", nps_score),
            (df_has_komp['G1C_CAT'] if len(df_has_komp)>0 else pd.Series(dtype=str),
             target_komp, nps_k)
        ]:
            if len(cat_data) > 0:
                comp = cat_data.value_counts().reset_index()
                comp.columns = ['Kategori','Jumlah']
                fig_d = px.pie(comp, values='Jumlah', names='Kategori', hole=0.62,
                    color='Kategori', color_discrete_map=NPS_COLORS)
                fig_d.update_traces(
                    textposition='outside', textinfo='percent+label',
                    marker=dict(line=dict(color='#0F172A', width=2))
                )
                fig_d.add_annotation(text=f"{title_d}<br><b>{nps_val:.0f}</b>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=13, color='#E2E8F0'))
                fig_d.update_layout(height=220, margin=dict(t=30,b=10,l=0,r=0))
                st.plotly_chart(elite_layout(fig_d), use_container_width=True)

    with r2b:
        st.markdown("<div class='section-header'>Scorecard Dimensi XYZ vs Kompetitor</div>",
                    unsafe_allow_html=True)
        sc_rows = [
            ("Kantor Cabang",    "OVR_KC_XYZ",           "OVR_KC_KOM"),
            ("  ↳ Operasional",  "OVR_KC_OPERASIONAL_XYZ","OVR_KC_OPERASIONAL_KOM"),
            ("  ↳ Parkir",       "OVR_KC_PARKIR_XYZ",    "OVR_KC_PARKIR_KOM"),
            ("  ↳ Banking Hall", "OVR_KC_BANKINGHALL_XYZ","OVR_KC_BANKINGHALL_KOM"),
            ("  ↳ Toilet",       "OVR_KC_TOILET_XYZ",    "OVR_KC_TOILET_KOM"),
            ("Sekuriti",         "OVR_SEKURITI_XYZ",     "OVR_SEKURITI_KOM"),
            ("Teller",           "OVR_TELLER_XYZ",       "OVR_TELLER_KOM"),
            ("Customer Service", "OVR_CS_XYZ",           "OVR_CS_KOM"),
            ("Customer Advisor", "OVR_CA_XYZ",           None),
            ("Sarana Elektronik","OVR_SARANA_XYZ",       None),
            ("ATM",              "OVR_ATM_XYZ",          "OVR_ATM_KOM"),
        ]
        sc_data = []
        for label, cx, ck in sc_rows:
            xv = df[cx].mean() if cx in df.columns else np.nan
            kv = df_has_komp[ck].mean() if ck and ck in df_has_komp.columns \
                 and len(df_has_komp)>0 else np.nan
            gp = xv - kv if not (np.isnan(xv) or np.isnan(kv)) else np.nan
            sc_data.append({"Dimensi":label,"XYZ":round(xv,2),
                "Kompetitor":round(kv,2) if not np.isnan(kv) else None,
                "Gap":round(gp,2) if not np.isnan(gp) else None})
        sc_df = pd.DataFrame(sc_data)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(name='Bank XYZ', y=sc_df['Dimensi'], x=sc_df['XYZ'],
            orientation='h', marker_color=COLOR_XYZ,
            text=sc_df['XYZ'], texttemplate='%{x:.2f}', textposition='outside'))
        fig_sc.add_trace(go.Bar(name='Kompetitor', y=sc_df['Dimensi'],
            x=pd.to_numeric(sc_df['Kompetitor'], errors='coerce'),
            orientation='h', marker_color=COLOR_KOMP, opacity=0.8,
            text=sc_df['Kompetitor'], texttemplate='%{x}', textposition='outside'))
        fig_sc.update_layout(barmode='group', xaxis_range=[4,6.6], height=400)
        st.plotly_chart(elite_layout(fig_sc), use_container_width=True)

    # ── Row 3: Top/Bottom + Korelasi ─────────────────────────────────
    r3a, r3b = st.columns(2)

    with r3a:
        st.markdown("<div class='section-header'>🏆 Top & Bottom Cabang</div>",
                    unsafe_allow_html=True)
        ovr_main = [c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c
                    and c not in ["OVR_KC_OPERASIONAL_XYZ","OVR_KC_PARKIR_XYZ",
                                   "OVR_KC_BANKINGHALL_XYZ","OVR_KC_TOILET_XYZ"]]
        if ovr_main:
            cs = df.groupby('CABANG')[ovr_main].mean().mean(axis=1).reset_index()
            cs.columns = ['CABANG','Skor']
            tb = pd.concat([
                cs.nlargest(5,'Skor').assign(Status='🌟 Top 5'),
                cs.nsmallest(5,'Skor').assign(Status='⚠️ Bottom 5')
            ])
            fig_tb = px.bar(tb, x='Skor', y='CABANG', color='Status', orientation='h',
                text='Skor',
                color_discrete_map={'🌟 Top 5':COLOR_XYZ,'⚠️ Bottom 5':COLOR_KOMP})
            fig_tb.update_traces(texttemplate='%{x:.2f}', textposition='outside')
            fig_tb.update_xaxes(range=[4.5,6.4])
            st.plotly_chart(elite_layout(fig_tb, height=380), use_container_width=True)

    with r3b:
        st.markdown("<div class='section-header'>📊 Matriks Korelasi</div>",
                    unsafe_allow_html=True)
        corr_map = {'NPS':'G1A','Kepuasan':'E1A','Loyalitas':'F1A',
                    'Teller':'OVR_TELLER_XYZ','CS':'OVR_CS_XYZ',
                    'ATM':'OVR_ATM_XYZ','Sekuriti':'OVR_SEKURITI_XYZ',
                    'Kantor Cabang':'OVR_KC_XYZ'}
        valid = {k:v for k,v in corr_map.items() if v in df.columns}
        cdf   = df[list(valid.values())].copy()
        cdf.columns = list(valid.keys())
        fig_c = px.imshow(cdf.corr(), text_auto=".2f",
            color_continuous_scale='RdBu', aspect='auto', zmin=-1, zmax=1)
        st.plotly_chart(elite_layout(fig_c, height=380), use_container_width=True)

    # ── Row 4: NPS per Provinsi ───────────────────────────────────────
    st.markdown("<div class='section-header'>🗺️ NPS per Provinsi</div>",
                unsafe_allow_html=True)
    prov_nps = df.groupby('PROV').agg(
        NPS=('G1A','mean'), Kepuasan=('E1A','mean'),
        Loyalitas=('F1A','mean'), Count=('SERIAL','count')
    ).reset_index().sort_values('NPS', ascending=True)

    fig_prov = px.bar(prov_nps, x='NPS', y='PROV', orientation='h',
        color='NPS', color_continuous_scale='Blues',
        text='NPS', hover_data=['Kepuasan','Loyalitas','Count'])
    fig_prov.update_traces(texttemplate='%{x:.1f}', textposition='outside')
    fig_prov.update_xaxes(range=[7,10.5])
    st.plotly_chart(elite_layout(fig_prov, "Rata-rata NPS per Provinsi", height=400),
        use_container_width=True)

# =====================================================================
# TAB 2 — KINERJA LAYANAN
# =====================================================================
with tab2:
    # ── Heatmap ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔥 Heatmap Kinerja Cabang</div>",
                unsafe_allow_html=True)
    hm_c1, hm_c2 = st.columns([2,1])
    with hm_c1:
        ovr_all = [c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c]
        ovr_lbl = {c: c.replace("OVR_","").replace("_XYZ","").replace("_"," ").title()
                   for c in ovr_all}
        sel_hm  = st.multiselect("Dimensi heatmap:", list(ovr_lbl.keys()),
            default=ovr_all, format_func=lambda x: ovr_lbl[x])
    with hm_c2:
        min_resp_hm = st.slider("Min. responden per cabang:", 3, 30, 8)
        sort_hm     = st.selectbox("Urutkan berdasarkan:",
            ["Rata-rata keseluruhan"] + [ovr_lbl[c] for c in sel_hm if c in ovr_lbl])

    if sel_hm:
        hm_filtered = df.groupby('CABANG').filter(lambda x: len(x) >= min_resp_hm)
        hm_data     = hm_filtered.groupby('CABANG')[sel_hm].mean().round(2)
        hm_data.columns = [ovr_lbl[c] for c in hm_data.columns]

        if sort_hm == "Rata-rata keseluruhan":
            hm_data = hm_data.loc[hm_data.mean(axis=1).sort_values(ascending=False).index]
        elif sort_hm in hm_data.columns:
            hm_data = hm_data.sort_values(sort_hm, ascending=False)

        fig_hm = px.imshow(hm_data, text_auto=".2f",
            color_continuous_scale='RdYlGn', aspect='auto', zmin=4, zmax=6)
        fig_hm.update_layout(height=max(400, len(hm_data) * 20))
        st.plotly_chart(elite_layout(fig_hm,
            "Heatmap Skor OVR per Cabang (Merah=Rendah, Hijau=Tinggi)"),
            use_container_width=True)

        if len(hm_data) > 0:
            wb = hm_data.mean(axis=1).idxmax()
            ww = hm_data.mean(axis=1).idxmin()
            insight_box(
                f"Performa terbaik: **{wb}** (avg {hm_data.loc[wb].mean():.2f}). "
                f"Perlu perhatian: **{ww}** (avg {hm_data.loc[ww].mean():.2f}). "
                f"Gap antara best & worst: {hm_data.loc[wb].mean()-hm_data.loc[ww].mean():.2f} poin."
            )

    st.markdown("---")

    # ── Drill-down ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Drill-Down Item Level</div>",
                unsafe_allow_html=True)
    dc1, dc2, dc3 = st.columns(3)
    with dc1: sel_dim = st.selectbox("Dimensi:", list(DIMENSI_MAP.keys()))
    with dc2: drill_mode = st.radio("Tampilan:",["Bar Chart","Scatter IPA"],horizontal=True)
    with dc3:
        sel_cab_drill = st.selectbox("Filter Cabang:",
            ["Semua"] + sorted(df['CABANG'].dropna().unique().tolist()))

    df_dr   = df if sel_cab_drill=="Semua" else df[df['CABANG']==sel_cab_drill]
    df_dr_k = df_has_komp if sel_cab_drill=="Semua" \
              else df_has_komp[df_has_komp['CABANG']==sel_cab_drill]

    di      = DIMENSI_MAP[sel_dim]
    ic      = [c for c in di["imp"]  if c in df_dr.columns]
    xc      = [c for c in di["xyz"]  if c in df_dr.columns]
    kc      = [c for c in di["komp"] if c in df_dr_k.columns]
    mn      = min(len(ic), len(xc))

    if mn > 0:
        lbs  = [short_label(c, col_map) for c in ic[:mn]]
        iv   = [df_dr[c].mean()  for c in ic[:mn]]
        xv   = [df_dr[c].mean()  for c in xc[:mn]]
        kv   = [df_dr_k[c].mean() if c in df_dr_k.columns else np.nan
                for c in kc[:mn]] if kc else []

        if drill_mode == "Bar Chart":
            dd1, dd2 = st.columns([2.5,1])
            with dd1:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Bar(name='Importance', x=iv, y=lbs,
                    orientation='h', marker_color=COLOR_IMPRT, opacity=0.65))
                fig_dd.add_trace(go.Bar(name='Sat. XYZ', x=xv, y=lbs,
                    orientation='h', marker_color=COLOR_XYZ))
                if kv:
                    fig_dd.add_trace(go.Bar(name=f'Sat. {target_komp}', x=kv, y=lbs,
                        orientation='h', marker_color=COLOR_KOMP, opacity=0.8))
                fig_dd.update_layout(barmode='group', xaxis_range=[3,6.6],
                    height=max(350, mn*26))
                st.plotly_chart(elite_layout(fig_dd,f"Importance vs Satisfaction — {sel_dim}"),
                    use_container_width=True)
            with dd2:
                gdf = pd.DataFrame({'Item':lbs,
                    'Gap':[x-i for x,i in zip(xv,iv)]}).sort_values('Gap')
                gdf['Warna'] = np.where(gdf['Gap']<0, COLOR_KOMP, COLOR_XYZ)
                fig_g = px.bar(gdf, x='Gap', y='Item', orientation='h', text='Gap')
                fig_g.update_traces(marker_color=gdf['Warna'],
                    texttemplate='%{x:.2f}', textposition='outside')
                fig_g.update_layout(height=max(350, mn*26))
                st.plotly_chart(elite_layout(fig_g, "Gap: Sat − Imp"),
                    use_container_width=True)
                worst = gdf.iloc[0]
                if worst['Gap'] < 0:
                    insight_box(f"Item paling kritis: **{worst['Item']}** "
                                f"(gap {worst['Gap']:.2f}). High importance, low satisfaction.")
        else:
            idf = pd.DataFrame({'Item':lbs,'Importance':iv,'Satisfaction':xv})
            mi, ms = np.nanmean(iv), np.nanmean(xv)
            def q_label(r):
                if r['Importance']>=mi and r['Satisfaction']<ms: return "⚠️ Perbaiki"
                elif r['Importance']>=mi and r['Satisfaction']>=ms: return "🌟 Pertahankan"
                elif r['Importance']<mi and r['Satisfaction']<ms: return "💤 Rendah"
                else: return "✅ Berlebihan"
            idf['Kuadran'] = idf.apply(q_label, axis=1)
            qc = {"⚠️ Perbaiki":COLOR_KOMP,"🌟 Pertahankan":"#34D399",
                  "💤 Rendah":COLOR_IMPRT,"✅ Berlebihan":"#FBBF24"}
            fig_ipa = px.scatter(idf, x='Satisfaction', y='Importance',
                text='Item', color='Kuadran', color_discrete_map=qc)
            if kv:
                fig_ipa.add_trace(go.Scatter(x=kv, y=iv, mode='markers',
                    name=target_komp, marker=dict(size=10, symbol='x',
                    color=COLOR_KOMP, line=dict(width=2))))
            fig_ipa.update_traces(marker=dict(size=11), textposition='top center',
                textfont=dict(size=9), selector=dict(mode='markers+text'))
            fig_ipa.add_vline(x=ms, line_dash="dash", line_color="#334155")
            fig_ipa.add_hline(y=mi, line_dash="dash", line_color="#334155")
            fig_ipa.update_layout(height=480)
            st.plotly_chart(elite_layout(fig_ipa, f"IPA Matrix — {sel_dim}"),
                use_container_width=True)

    st.markdown("---")

    # ── Waktu Tunggu ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>⏱️ Analisis Waktu Tunggu</div>",
                unsafe_allow_html=True)
    wt1, wt2, wt3 = st.columns(3)

    with wt1:
        df_tl = df[df['PANEL']=='Teller'].dropna(subset=['TL5','TL6'])
        if len(df_tl) > 0:
            wtd = pd.DataFrame({'Metrik':['Aktual','Toleransi'],
                'Menit':[df_tl['TL5'].mean(), df_tl['TL6'].mean()]})
            fig_wt = px.bar(wtd, x='Metrik', y='Menit', color='Metrik', text='Menit',
                color_discrete_map={'Aktual':COLOR_KOMP,'Toleransi':COLOR_XYZ})
            fig_wt.update_traces(texttemplate='%{y:.1f} mnt', textposition='outside')
            fig_wt.update_yaxes(range=[0, df_tl['TL6'].mean()*1.6])
            st.plotly_chart(elite_layout(fig_wt,"⏳ Teller: Aktual vs Toleransi"),
                use_container_width=True)
            insight_box(
                f"Rata-rata tunggu Teller: **{df_tl['TL5'].mean():.1f} mnt** "
                f"(toleransi: {df_tl['TL6'].mean():.1f} mnt). "
                f"Buffer aman: {df_tl['TL6'].mean()-df_tl['TL5'].mean():.1f} mnt."
            )

    with wt2:
        df_cswt = df[df['PANEL']=='CS'].dropna(subset=['CS5','CS6'])
        if len(df_cswt) > 0:
            wtd2 = pd.DataFrame({'Metrik':['Aktual','Toleransi'],
                'Menit':[df_cswt['CS5'].mean(), df_cswt['CS6'].mean()]})
            fig_wt2 = px.bar(wtd2, x='Metrik', y='Menit', color='Metrik', text='Menit',
                color_discrete_map={'Aktual':COLOR_KOMP,'Toleransi':COLOR_XYZ})
            fig_wt2.update_traces(texttemplate='%{y:.1f} mnt', textposition='outside')
            fig_wt2.update_yaxes(range=[0, df_cswt['CS6'].mean()*1.6])
            st.plotly_chart(elite_layout(fig_wt2,"⏳ CS: Aktual vs Toleransi"),
                use_container_width=True)

    with wt3:
        # Jam sibuk
        jt = df['TL1'].dropna().value_counts().reset_index()
        jt.columns = ['Jam','Teller']
        jcs = df['CS1'].dropna().value_counts().reset_index()
        jcs.columns = ['Jam','CS']
        jam_df = pd.merge(jt, jcs, on='Jam', how='outer').fillna(0)
        fig_jam = go.Figure()
        fig_jam.add_trace(go.Bar(name='Teller', x=jam_df['Jam'], y=jam_df['Teller'],
            marker_color=COLOR_XYZ))
        fig_jam.add_trace(go.Bar(name='CS', x=jam_df['Jam'], y=jam_df['CS'],
            marker_color=COLOR_KOMP))
        fig_jam.update_layout(barmode='group', xaxis_tickangle=-25, height=280)
        st.plotly_chart(elite_layout(fig_jam,"⏰ Jam Paling Sibuk"),
            use_container_width=True)
        if len(jt) > 0:
            insight_box(f"Jam tersibuk Teller: **{jt.iloc[0]['Jam']}**. "
                        f"Pertimbangkan penambahan staf di jam ini.")

    # ── Waktu Tunggu per Cabang ───────────────────────────────────────
    st.markdown("<div class='section-header'>⏱️ Waktu Tunggu per Cabang</div>",
                unsafe_allow_html=True)
    wt_panel = st.radio("Panel:", ["Teller","CS"], horizontal=True)
    if wt_panel == "Teller":
        wt_cab = df[df['PANEL']=='Teller'].groupby('CABANG').agg(
            Aktual=('TL5','mean'), Toleransi=('TL6','mean'), Count=('SERIAL','count')
        ).reset_index()
        wt_cab = wt_cab[wt_cab['Count'] >= 5].sort_values('Aktual', ascending=False)
    else:
        wt_cab = df[df['PANEL']=='CS'].groupby('CABANG').agg(
            Aktual=('CS5','mean'), Toleransi=('CS6','mean'), Count=('SERIAL','count')
        ).reset_index()
        wt_cab = wt_cab[wt_cab['Count'] >= 5].sort_values('Aktual', ascending=False)

    if len(wt_cab) > 0:
        wt_cab['Melebihi Toleransi'] = wt_cab['Aktual'] > wt_cab['Toleransi']
        fig_wtc = go.Figure()
        fig_wtc.add_trace(go.Bar(name='Aktual', x=wt_cab['CABANG'],
            y=wt_cab['Aktual'], marker_color=COLOR_KOMP))
        fig_wtc.add_trace(go.Bar(name='Toleransi', x=wt_cab['CABANG'],
            y=wt_cab['Toleransi'], marker_color=COLOR_XYZ, opacity=0.7))
        fig_wtc.update_layout(barmode='overlay', xaxis_tickangle=-30, height=380)
        st.plotly_chart(elite_layout(fig_wtc, f"Waktu Tunggu {wt_panel} per Cabang"),
            use_container_width=True)
        over_tol = wt_cab[wt_cab['Melebihi Toleransi']]
        if len(over_tol) > 0:
            insight_box(
                f"**{len(over_tol)} cabang** waktu tunggu {wt_panel} melebihi toleransi: "
                f"{', '.join(over_tol['CABANG'].tolist()[:5])}."
            )

# =====================================================================
# TAB 3 — BRAND & KOMPETITOR
# =====================================================================
with tab3:
    brand_imp_cols = [f"T_C1A_{i}" for i in range(1,25) if f"T_C1A_{i}" in df.columns]
    brand_xyz_cols = sorted([c for c in df.columns if c.startswith("T_C1B_") and
        int(c.split("_")[-1]) % 3 == 2], key=lambda x: int(x.split("_")[-1]))
    brand_kom_cols = sorted([c for c in df.columns if c.startswith("T_C1B_") and
        int(c.split("_")[-1]) % 3 == 0], key=lambda x: int(x.split("_")[-1]))

    blabels = [short_label(c, col_map) for c in brand_imp_cols]
    bimp    = [df[c].mean() for c in brand_imp_cols]
    bxyz    = [df[c].mean() for c in brand_xyz_cols[:len(brand_imp_cols)]]
    bkom    = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
               for c in brand_kom_cols[:len(brand_imp_cols)]]

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("<div class='section-header'>🕸️ Radar Brand</div>",
                    unsafe_allow_html=True)
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=bxyz, theta=blabels, fill='toself',
            name='Bank XYZ', line_color=COLOR_XYZ,
            fillcolor='rgba(59,130,246,0.12)'))
        fig_r.add_trace(go.Scatterpolar(r=bkom, theta=blabels, fill='toself',
            name=target_komp, line_color=COLOR_KOMP,
            fillcolor='rgba(248,113,113,0.08)'))
        fig_r.update_layout(
            polar=dict(
                bgcolor='#1E293B',
                radialaxis=dict(visible=True, range=[3,6],
                    gridcolor='#334155', tickfont=dict(color='#94A3B8')),
                angularaxis=dict(tickfont=dict(color='#CBD5E1', size=10))
            ),
            legend=dict(orientation="h", y=-0.12), height=480
        )
        st.plotly_chart(elite_layout(fig_r), use_container_width=True)

    with b2:
        st.markdown("<div class='section-header'>🎯 IPA Matrix Brand</div>",
                    unsafe_allow_html=True)
        bdf = pd.DataFrame({'Label':blabels,'Importance':bimp,'Satisfaction':bxyz})
        mi_b, ms_b = np.nanmean(bimp), np.nanmean(bxyz)
        def bq(r):
            if r['Importance']>=mi_b and r['Satisfaction']<ms_b: return "⚠️ Perbaiki"
            elif r['Importance']>=mi_b and r['Satisfaction']>=ms_b: return "🌟 Pertahankan"
            elif r['Importance']<mi_b and r['Satisfaction']<ms_b: return "💤 Rendah"
            else: return "✅ Berlebihan"
        bdf['Kuadran'] = bdf.apply(bq, axis=1)
        qcm = {"⚠️ Perbaiki":COLOR_KOMP,"🌟 Pertahankan":"#34D399",
               "💤 Rendah":COLOR_IMPRT,"✅ Berlebihan":"#FBBF24"}
        fig_ib = px.scatter(bdf, x='Satisfaction', y='Importance', text='Label',
            color='Kuadran', color_discrete_map=qcm, hover_data=['Kuadran'])
        fig_ib.update_traces(marker=dict(size=10), textposition='top center',
            textfont=dict(size=8))
        fig_ib.add_vline(x=ms_b, line_dash="dash", line_color="#334155")
        fig_ib.add_hline(y=mi_b, line_dash="dash", line_color="#334155")
        fig_ib.update_layout(height=480)
        st.plotly_chart(elite_layout(fig_ib), use_container_width=True)

    # ── Gap bar ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Gap Kompetitif per Atribut</div>",
                unsafe_allow_html=True)
    gb = pd.DataFrame({'Atribut':blabels,'XYZ':bxyz,'Komp':bkom,
        'Gap':[x-k for x,k in zip(bxyz,bkom)]}).sort_values('Gap')
    gb['Warna'] = np.where(gb['Gap']<0, COLOR_KOMP, COLOR_XYZ)
    fig_gb = px.bar(gb, x='Gap', y='Atribut', orientation='h', text='Gap',
        hover_data=['XYZ','Komp'])
    fig_gb.update_traces(marker_color=gb['Warna'],
        texttemplate='%{x:.2f}', textposition='outside')
    st.plotly_chart(elite_layout(fig_gb,
        f"Gap Brand XYZ vs {target_komp} (+= XYZ unggul)", height=520),
        use_container_width=True)

    tg = gb.nlargest(1,'Gap').iloc[0]
    bg = gb.nsmallest(1,'Gap').iloc[0]
    insight_box(f"Keunggulan terbesar XYZ: **{tg['Atribut']}** (+{tg['Gap']:.2f}). "
                f"Perlu ditingkatkan: **{bg['Atribut']}** ({bg['Gap']:.2f}).")

    # ── Share of Wallet ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏦 Share of Wallet</div>",
                unsafe_allow_html=True)
    sw1, sw2, sw3 = st.columns(3)
    with sw1:
        bo = df['A1AX'].dropna().str.split(';').explode().str.strip()
        bo = bo[bo != ''].value_counts().head(8).reset_index()
        bo.columns = ['Bank','Jumlah']
        fig_bk = px.bar(bo, x='Jumlah', y='Bank', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_bk.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_bk,"Bank Lain yang Aktif"),
            use_container_width=True)
    with sw2:
        sp = df['A1B'].value_counts().reset_index()
        sp.columns = ['Bank','Jumlah']
        fig_sp = px.pie(sp, values='Jumlah', names='Bank', hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(elite_layout(fig_sp,"Bank Utama Simpan Dana"),
            use_container_width=True)
    with sw3:
        tr = df['A1C'].value_counts().reset_index()
        tr.columns = ['Bank','Jumlah']
        fig_tr = px.pie(tr, values='Jumlah', names='Bank', hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(elite_layout(fig_tr,"Bank Utama Bertransaksi"),
            use_container_width=True)

    xyz_sp = (df['A1B']=='Bank XYZ').mean()*100
    xyz_tr = (df['A1C']=='Bank XYZ').mean()*100
    riv    = bo.iloc[0]['Bank'] if len(bo)>0 else "N/A"
    insight_box(f"XYZ sebagai rekening utama simpan: **{xyz_sp:.0f}%**, "
                f"utama transaksi: **{xyz_tr:.0f}%**. Rival terbesar: **{riv}**.")

# =====================================================================
# TAB 4 — TOUCHPOINT & IPA
# =====================================================================
with tab4:
    st.markdown("<div class='section-header'>🎯 Analisis Touchpoint Interaktif</div>",
                unsafe_allow_html=True)
    tp_c1, tp_c2, tp_c3 = st.columns(3)
    with tp_c1: sel_tp = st.selectbox("Touchpoint:", list(DIMENSI_MAP.keys()))
    with tp_c2: tp_view = st.radio("Tampilan:",["IPA Scatter","Bar Comparison"],horizontal=True)
    with tp_c3:
        tp_cab = st.selectbox("Filter Cabang:",
            ["Semua"] + sorted(df['CABANG'].unique().tolist()))

    dft  = df if tp_cab=="Semua" else df[df['CABANG']==tp_cab]
    dftk = df_has_komp if tp_cab=="Semua" \
           else df_has_komp[df_has_komp['CABANG']==tp_cab]

    dtp  = DIMENSI_MAP[sel_tp]
    itpc = [c for c in dtp["imp"]  if c in dft.columns]
    xtpc = [c for c in dtp["xyz"]  if c in dft.columns]
    ktpc = [c for c in dtp["komp"] if c in dftk.columns]
    mtp  = min(len(itpc), len(xtpc))

    if mtp > 0:
        ltps = [short_label(c,col_map) for c in itpc[:mtp]]
        ivtp = [dft[c].mean()  for c in itpc[:mtp]]
        xvtp = [dft[c].mean()  for c in xtpc[:mtp]]
        kvtp = [dftk[c].mean() if c in dftk.columns else np.nan
                for c in ktpc[:mtp]] if ktpc else []

        if tp_view == "IPA Scatter":
            tp1, tp2 = st.columns([1.6,1])
            with tp1:
                itpdf = pd.DataFrame({'Item':ltps,'Importance':ivtp,'Satisfaction':xvtp})
                mip, msp = np.nanmean(ivtp), np.nanmean(xvtp)
                def tpq(r):
                    if r['Importance']>=mip and r['Satisfaction']<msp: return "⚠️ Perbaiki"
                    elif r['Importance']>=mip and r['Satisfaction']>=msp: return "🌟 Pertahankan"
                    elif r['Importance']<mip and r['Satisfaction']<msp: return "💤 Rendah"
                    else: return "✅ Berlebihan"
                itpdf['Kuadran'] = itpdf.apply(tpq, axis=1)
                qtp = {"⚠️ Perbaiki":COLOR_KOMP,"🌟 Pertahankan":"#34D399",
                       "💤 Rendah":COLOR_IMPRT,"✅ Berlebihan":"#FBBF24"}
                fig_itp = px.scatter(itpdf, x='Satisfaction', y='Importance',
                    text='Item', color='Kuadran', color_discrete_map=qtp)
                if kvtp:
                    fig_itp.add_trace(go.Scatter(x=kvtp, y=ivtp, mode='markers',
                        name=target_komp, marker=dict(size=10, symbol='x',
                        color=COLOR_KOMP, line=dict(width=2))))
                fig_itp.update_traces(marker=dict(size=11), textposition='top center',
                    textfont=dict(size=9), selector=dict(mode='markers+text'))
                fig_itp.add_vline(x=msp, line_dash="dash", line_color="#334155")
                fig_itp.add_hline(y=mip, line_dash="dash", line_color="#334155")
                fig_itp.update_layout(height=460)
                st.plotly_chart(elite_layout(fig_itp,f"IPA — {sel_tp}"),
                    use_container_width=True)
            with tp2:
                st.markdown("**📋 Prioritas per Kuadran:**")
                for qn, qcv in qtp.items():
                    items_q = itpdf[itpdf['Kuadran']==qn]['Item'].tolist()
                    if items_q:
                        st.markdown(f"<span style='color:{qcv};font-weight:700;'>{qn}</span>",
                            unsafe_allow_html=True)
                        for it in items_q:
                            st.markdown(f"<span style='color:#94A3B8;font-size:12px;'>• {it}</span>",
                                unsafe_allow_html=True)
        else:
            fig_btp = go.Figure()
            fig_btp.add_trace(go.Bar(name='Importance', x=ltps, y=ivtp,
                marker_color=COLOR_IMPRT, opacity=0.7))
            fig_btp.add_trace(go.Bar(name='Sat. XYZ', x=ltps, y=xvtp,
                marker_color=COLOR_XYZ))
            if kvtp:
                fig_btp.add_trace(go.Bar(name=f'Sat. {target_komp}', x=ltps, y=kvtp,
                    marker_color=COLOR_KOMP, opacity=0.8))
            fig_btp.update_layout(barmode='group', yaxis_range=[3,6.6],
                xaxis_tickangle=-25, height=420)
            st.plotly_chart(elite_layout(fig_btp,f"Comparison — {sel_tp}"),
                use_container_width=True)

        if kvtp:
            st.markdown("<div class='section-header'>📊 Gap Kompetitif per Item</div>",
                        unsafe_allow_html=True)
            gtdf = pd.DataFrame({'Item':ltps,
                'Gap':[x-k for x,k in zip(xvtp,kvtp)]}).sort_values('Gap')
            gtdf['Warna'] = np.where(gtdf['Gap']<0, COLOR_KOMP, COLOR_XYZ)
            fig_gtp = px.bar(gtdf, x='Gap', y='Item', orientation='h', text='Gap')
            fig_gtp.update_traces(marker_color=gtdf['Warna'],
                texttemplate='%{x:.2f}', textposition='outside')
            st.plotly_chart(elite_layout(fig_gtp,
                f"Gap XYZ vs {target_komp} — {sel_tp}", height=380),
                use_container_width=True)

    st.markdown("---")

    # ── Jenis transaksi ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Jenis Transaksi vs Skor Layanan</div>",
                unsafe_allow_html=True)
    d1m = {
        'TELLER': df[df['D1_TYPE']=='TELLER']['OVR_TELLER_XYZ'].mean(),
        'CS':     df[df['D1_TYPE']=='CS']['OVR_CS_XYZ'].mean(),
        'KEDUANYA': df[df['D1_TYPE']=='BOTH'][['OVR_TELLER_XYZ','OVR_CS_XYZ']].mean(axis=1).mean()
    }
    d1df = pd.DataFrame({'Jenis':list(d1m.keys()),'Skor':list(d1m.values())})
    fig_d1 = px.bar(d1df, x='Jenis', y='Skor', color='Jenis', text='Skor',
        color_discrete_map={'TELLER':COLOR_XYZ,'CS':COLOR_KOMP,'KEDUANYA':'#FBBF24'})
    fig_d1.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig_d1.update_yaxes(range=[4.5,6.4])
    st.plotly_chart(elite_layout(fig_d1,"Skor Berdasarkan Jenis Transaksi"),
        use_container_width=True)

# =====================================================================
# TAB 5 — EMOSI & LOYALITAS
# =====================================================================
with tab5:
    epc = [c for c in ["T_I1A_2","T_I1A_5","T_I1A_8","T_I1A_11","T_I1A_14",
           "T_I1A_17","T_I1A_20","T_I1A_23","T_I1A_26"] if c in df.columns]
    enc = [c for c in ["T_I1A_29","T_I1A_32","T_I1A_35","T_I1A_38",
           "T_I1A_41","T_I1A_44","T_I1A_47"] if c in df.columns]
    epkc = [c for c in ["T_I1A_3","T_I1A_6","T_I1A_9","T_I1A_12","T_I1A_15",
            "T_I1A_18","T_I1A_21","T_I1A_24","T_I1A_27"] if c in df_has_komp.columns]
    enkc = [c for c in ["T_I1A_30","T_I1A_33","T_I1A_36","T_I1A_39",
            "T_I1A_42","T_I1A_45","T_I1A_48"] if c in df_has_komp.columns]

    epl_p = ["Bahagia","Percaya","Dihargai","Diperhatikan",
             "Aman","Fokus","Dimanjakan","Tertarik","Penuh Semangat"]
    epl_n = ["Tidak Puas","Frustasi","Kecewa","Tertekan",
             "Tidak Bahagia","Diabaikan","Tergesa-gesa"]

    epv = [df[c].mean() for c in epc]
    env = [df[c].mean() for c in enc]
    epkv = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan for c in epkc]
    enkv = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan for c in enkc]

    ep_avg = np.nanmean(epv)
    en_avg = np.nanmean(env)
    epk_avg = np.nanmean(epkv)
    enk_avg = np.nanmean(enkv)

    # KPI
    ek = st.columns(4)
    ek[0].markdown(fmt_card("Emosi Positif XYZ",  f"{ep_avg:.2f}", "blue", "Avg 9 dim."),
        unsafe_allow_html=True)
    ek[1].markdown(fmt_card("Emosi Negatif XYZ",  f"{en_avg:.2f}", "amber", "↓ makin baik"),
        unsafe_allow_html=True)
    ek[2].markdown(fmt_card("Emosi Positif Komp", f"{epk_avg:.2f}", "red", "Avg 9 dim."),
        unsafe_allow_html=True)
    ek[3].markdown(fmt_card("Emosi Negatif Komp", f"{enk_avg:.2f}", "amber", "↓ makin baik"),
        unsafe_allow_html=True)

    insight_box(
        f"XYZ unggul di emosi positif ({ep_avg:.2f} vs {epk_avg:.2f}) dan lebih rendah "
        f"di emosi negatif ({en_avg:.2f} vs {enk_avg:.2f}). "
        f"Pengalaman nasabah XYZ secara keseluruhan lebih positif."
    )

    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("<div class='section-header'>😊 Emosi Positif</div>",
                    unsafe_allow_html=True)
        fig_ep = go.Figure()
        fig_ep.add_trace(go.Bar(name='XYZ', x=epl_p[:len(epv)], y=epv,
            marker_color=COLOR_XYZ, text=np.round(epv,2), textposition='outside'))
        fig_ep.add_trace(go.Bar(name=target_komp, x=epl_p[:len(epkv)], y=epkv,
            marker_color=COLOR_KOMP, text=np.round(epkv,2), textposition='outside'))
        fig_ep.update_layout(barmode='group', yaxis_range=[3,6.8],
            xaxis_tickangle=-20, height=360)
        st.plotly_chart(elite_layout(fig_ep), use_container_width=True)

    with ec2:
        st.markdown("<div class='section-header'>😠 Emosi Negatif</div>",
                    unsafe_allow_html=True)
        fig_en2 = go.Figure()
        fig_en2.add_trace(go.Bar(name='XYZ', x=epl_n[:len(env)], y=env,
            marker_color=COLOR_XYZ, text=np.round(env,2), textposition='outside'))
        fig_en2.add_trace(go.Bar(name=target_komp, x=epl_n[:len(enkv)], y=enkv,
            marker_color=COLOR_KOMP, text=np.round(enkv,2), textposition='outside'))
        fig_en2.update_layout(barmode='group', yaxis_range=[1,4],
            xaxis_tickangle=-20, height=360)
        st.plotly_chart(elite_layout(fig_en2,"↓ Makin Rendah Makin Baik"),
            use_container_width=True)

    st.markdown("---")

    # ── Brand Equity ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>💎 Brand Equity — 15 Atribut</div>",
                unsafe_allow_html=True)
    h_xc = [c for c in ["T_H1A_2","T_H1A_5","T_H1A_8","T_H1A_11","T_H1A_14",
        "T_H1A_17","T_H1A_20","T_H1A_23","T_H1A_26","T_H1A_29",
        "T_H1A_32","T_H1A_35","T_H1A_38","T_H1A_41","T_H1A_44"] if c in df.columns]
    h_kc = [c for c in ["T_H1A_3","T_H1A_6","T_H1A_9","T_H1A_12","T_H1A_15",
        "T_H1A_18","T_H1A_21","T_H1A_24","T_H1A_27","T_H1A_30",
        "T_H1A_33","T_H1A_36","T_H1A_39","T_H1A_42","T_H1A_45"]
        if c in df_has_komp.columns]
    hlbl = ["Tetap Gunakan","Kemudahan Transaksi","Digunakan Banyak",
            "Keuntungan Finansial","Produk Lengkap","Promo Gaya Hidup",
            "Kecepatan","Rasa Aman","Kenyamanan","Merasa Dihargai",
            "Bangga","Modern","Bank Turun-Temurun","Cukup Satu Bank","Bergengsi"]
    hxv = [df[c].mean() for c in h_xc]
    hkv = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan for c in h_kc]
    lh  = hlbl[:min(len(hxv),len(hlbl))]

    fig_he = go.Figure()
    fig_he.add_trace(go.Bar(name='Bank XYZ', x=lh, y=hxv[:len(lh)],
        marker_color=COLOR_XYZ, text=np.round(hxv[:len(lh)],2), textposition='outside'))
    fig_he.add_trace(go.Bar(name=target_komp, x=lh, y=hkv[:len(lh)],
        marker_color=COLOR_KOMP, text=np.round(hkv[:len(lh)],2), textposition='outside'))
    fig_he.update_layout(barmode='group', yaxis_range=[3,6.8],
        xaxis_tickangle=-25, height=380)
    st.plotly_chart(elite_layout(fig_he,"Brand Equity XYZ vs Kompetitor"),
        use_container_width=True)

    # ── Korelasi & Scatter ────────────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Korelasi Emosi vs Outcome</div>",
                unsafe_allow_html=True)
    ep_avg_s = df[[c for c in epc if c in df.columns]].mean(axis=1)
    en_avg_s = df[[c for c in enc if c in df.columns]].mean(axis=1)

    cce = pd.DataFrame({'Emosi Pos':ep_avg_s,'Emosi Neg':en_avg_s,
        'NPS':df['G1A'],'Kepuasan':df['E1A'],'Loyalitas':df['F1A']}).corr()
    ec_corr_c, ec_sc1, ec_sc2 = st.columns(3)
    with ec_corr_c:
        fig_ec = px.imshow(cce, text_auto=".2f",
            color_continuous_scale='RdBu', aspect='auto', zmin=-1, zmax=1)
        st.plotly_chart(elite_layout(fig_ec, height=320), use_container_width=True)
    with ec_sc1:
        sc1df = df[['G1A','G1A_CAT']].copy()
        sc1df['Emosi Pos'] = ep_avg_s
        fig_sc1 = px.scatter(sc1df, x='Emosi Pos', y='G1A', color='G1A_CAT',
            color_discrete_map=NPS_COLORS, trendline='ols', opacity=0.5,
            labels={'G1A':'NPS'})
        st.plotly_chart(elite_layout(fig_sc1,"Emosi Pos vs NPS",height=320),
            use_container_width=True)
    with ec_sc2:
        sc2df = df[['F1A','G1A_CAT']].copy()
        sc2df['Emosi Pos'] = ep_avg_s
        fig_sc2 = px.scatter(sc2df, x='Emosi Pos', y='F1A', color='G1A_CAT',
            color_discrete_map=NPS_COLORS, trendline='ols', opacity=0.5,
            labels={'F1A':'Loyalitas'})
        st.plotly_chart(elite_layout(fig_sc2,"Emosi Pos vs Loyalitas",height=320),
            use_container_width=True)

# =====================================================================
# TAB 6 — DIGITALISASI
# =====================================================================
with tab6:
    # KPI Digitalisasi
    dc = [c for c in ["T_J1_1","T_J1_2","T_J1_3","T_J1_4","T_J1_5"] if c in df.columns]
    dlbl = ["Digitalisasi","Digital Signage","Smart Table","Tablet Survey","Akses Cabang"]
    dv = [df[c].mean() for c in dc]

    st.markdown("<div class='section-header'>📱 KPI Digitalisasi</div>",
                unsafe_allow_html=True)
    dk = st.columns(len(dc))
    for i,(cw,lb,vl) in enumerate(zip(dk,dlbl,dv)):
        col = "green" if vl>=5.5 else ("amber" if vl>=4.5 else "red")
        cw.markdown(fmt_card(lb,f"{vl:.2f}",col,"Skala 1–6"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    dg1, dg2 = st.columns(2)

    with dg1:
        fig_dg = px.bar(x=dv, y=dlbl[:len(dv)], orientation='h',
            color=dv, color_continuous_scale='Blues', text=np.round(dv,2))
        fig_dg.update_traces(textposition='outside')
        fig_dg.update_xaxes(range=[3,6.8])
        st.plotly_chart(elite_layout(fig_dg,"Skor Persepsi Digitalisasi"),
            use_container_width=True)

    with dg2:
        if "T_J1_1" in df.columns:
            dp = df.groupby('PROV')["T_J1_1"].mean().reset_index()
            dp.columns = ['Provinsi','Skor']
            dp = dp.sort_values('Skor',ascending=True)
            fig_dp = px.bar(dp, x='Skor', y='Provinsi', orientation='h',
                color='Skor', color_continuous_scale='Blues', text='Skor')
            fig_dp.update_traces(texttemplate='%{x:.2f}', textposition='outside')
            fig_dp.update_xaxes(range=[3,6.8])
            st.plotly_chart(elite_layout(fig_dp,"Digitalisasi per Provinsi"),
                use_container_width=True)

    st.markdown("---")

    # ── Sarana Elektronik ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🖥️ Sarana Elektronik Layanan</div>",
                unsafe_allow_html=True)
    slc  = [f"T_SL2_{i}" for i in range(1,17) if f"T_SL2_{i}" in df.columns]
    sllb = [short_label(c,col_map) for c in slc]
    slv  = [df[c].mean() for c in slc]

    sl1, sl2 = st.columns(2)
    with sl1:
        fig_sl = px.bar(x=slv, y=sllb[:len(slv)], orientation='h',
            color=slv, color_continuous_scale='Teal', text=np.round(slv,2))
        fig_sl.update_traces(textposition='outside')
        fig_sl.update_xaxes(range=[3,6.8])
        st.plotly_chart(elite_layout(fig_sl,"Ketersediaan & Fungsi Sarana"),
            use_container_width=True)

    with sl2:
        if "T_J1_1" in df.columns:
            scd = df[['T_J1_1','E1A','G1A_CAT']].dropna()
            fig_ds = px.scatter(scd, x='T_J1_1', y='E1A', color='G1A_CAT',
                color_discrete_map=NPS_COLORS, trendline='ols', opacity=0.5,
                labels={'T_J1_1':'Persepsi Digitalisasi','E1A':'Kepuasan'})
            st.plotly_chart(elite_layout(fig_ds,"Digitalisasi vs Kepuasan"),
                use_container_width=True)

    # ── E-form ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📋 E-Form & Awareness</div>",
                unsafe_allow_html=True)
    ef1, ef2 = st.columns(2)
    with ef1:
        if 'D2' in df.columns:
            ef = df['D2'].dropna().value_counts().reset_index()
            ef.columns = ['Status','Jumlah']
            fig_ef = px.pie(ef, values='Jumlah', names='Status', hole=0.55,
                color_discrete_sequence=['#8B5CF6','#3B82F6','#F43F5E'])
            st.plotly_chart(elite_layout(fig_ef,"Penggunaan E-Form"),
                use_container_width=True)
    with ef2:
        if 'D4' in df.columns:
            ea = df['D4'].dropna().value_counts().reset_index()
            ea.columns = ['Status','Jumlah']
            fig_ea = px.bar(ea, x='Jumlah', y='Status', orientation='h',
                color_discrete_sequence=['#0EA5E9'], text='Jumlah')
            fig_ea.update_traces(textposition='outside')
            st.plotly_chart(elite_layout(fig_ea,"Awareness E-Form"),
                use_container_width=True)

    # ── Saran Perbaikan ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>📝 Saran Perbaikan</div>",
                unsafe_allow_html=True)
    j2m = {"T_J2_1":"Digitalisasi Layanan","T_J2_2":"Digital Signage",
           "T_J2_3":"Smart Table","T_J2_4":"Tablet Survey","T_J2_5":"Akses Cabang"}
    j2l = st.selectbox("Topik:", list(j2m.values()))
    j2c = [k for k,v in j2m.items() if v==j2l][0]
    if j2c in df.columns:
        sr = df[j2c].dropna().value_counts().reset_index()
        sr.columns = ['Saran','Jumlah']
        sr = sr[sr['Saran'].str.strip().isin(['','None','Tidak Ada',
            'Tidak Ada /  Tidak Tahu']) == False]
        if len(sr) > 0:
            st.dataframe(sr, use_container_width=True, hide_index=True, height=200)

# =====================================================================
# TAB 7 — PROFIL & SEGMENTASI
# =====================================================================
with tab7:
    # Demografi
    st.markdown("<div class='section-header'>👥 Profil Demografis</div>",
                unsafe_allow_html=True)
    d1,d2,d3,d4 = st.columns(4)
    with d1:
        fig_gn = px.pie(df, names='S1', hole=0.55,
            color_discrete_sequence=[COLOR_XYZ,'#60A5FA'])
        st.plotly_chart(elite_layout(fig_gn,"Gender"), use_container_width=True)
    with d2:
        ac = df['S2_2'].value_counts().reset_index()
        ac.columns = ['Usia','N']
        fig_ag = px.bar(ac, x='Usia', y='N', color_discrete_sequence=[COLOR_XYZ], text='N')
        fig_ag.update_traces(textposition='outside')
        fig_ag.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(elite_layout(fig_ag,"Usia"), use_container_width=True)
    with d3:
        ec = df['P3'].value_counts().reset_index()
        ec.columns = ['Pend','N']
        fig_ed = px.bar(ec, x='N', y='Pend', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='N')
        fig_ed.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_ed,"Pendidikan"), use_container_width=True)
    with d4:
        jc = df['P4'].value_counts().reset_index()
        jc.columns = ['Pekerjaan','N']
        fig_jb = px.bar(jc, x='N', y='Pekerjaan', orientation='h',
            color_discrete_sequence=['#8B5CF6'], text='N')
        fig_jb.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_jb,"Pekerjaan"), use_container_width=True)

    # SES
    ss1, ss2 = st.columns(2)
    with ss1:
        sc = df['P5'].value_counts().reset_index()
        sc.columns = ['SES','N']
        fig_ss = px.bar(sc, x='N', y='SES', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='N')
        fig_ss.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_ss,"Tingkat Pengeluaran (SES)"),
            use_container_width=True)
    with ss2:
        ic = df['P6'].value_counts().reset_index()
        ic.columns = ['Penghasilan','N']
        fig_ic = px.bar(ic, x='N', y='Penghasilan', orientation='h',
            color_discrete_sequence=['#10B981'], text='N')
        fig_ic.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_ic,"Distribusi Penghasilan"),
            use_container_width=True)

    # ── Treemap ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🗺️ Peta Geografis</div>",
                unsafe_allow_html=True)
    geo = df.groupby(['PROV','KABKOTA','CABANG']).agg(
        N=('SERIAL','count'), NPS=('G1A','mean'),
        Kepuasan=('E1A','mean'), Loyalitas=('F1A','mean')
    ).reset_index()
    cm_sel = st.selectbox("Warna berdasarkan:",['NPS','Kepuasan','Loyalitas','N'])
    fig_tr = px.treemap(geo,
        path=[px.Constant("Nasional"),'PROV','KABKOTA','CABANG'],
        values='N', color=cm_sel, color_continuous_scale='RdYlGn',
        hover_data={'NPS':':.2f','Kepuasan':':.2f','Loyalitas':':.2f'})
    fig_tr.update_layout(height=500)
    st.plotly_chart(elite_layout(fig_tr,f"Treemap (Warna={cm_sel})"),
        use_container_width=True)

    # ── Segmentasi Interaktif ─────────────────────────────────────────
    st.markdown("<div class='section-header'>🔀 Segmentasi Interaktif</div>",
                unsafe_allow_html=True)
    sg1,sg2,sg3 = st.columns(3)
    with sg1:
        seg_by = st.selectbox("Segmen by:", [
            'S1 → Gender','S2_2 → Usia','S4 → Tenure',
            'S7 → Frekuensi','P3 → Pendidikan',
            'P4 → Pekerjaan','P1 → Status Nikah','P5 → SES'
        ])
    with sg2:
        seg_met = st.selectbox("Metrik:", [
            'G1A → NPS','E1A → Kepuasan','F1A → Loyalitas',
            'OVR_TELLER_XYZ → Teller','OVR_CS_XYZ → CS',
            'OVR_ATM_XYZ → ATM','OVR_SEKURITI_XYZ → Sekuriti',
            'OVR_KC_XYZ → Kantor Cabang'
        ])
    with sg3:
        seg_ct = st.radio("Chart:", ["Bar","Box"], horizontal=True)

    sk = seg_by.split(' → ')[0].strip()
    mk = seg_met.split(' → ')[0].strip()

    if sk in df.columns and mk in df.columns:
        if seg_ct == "Bar":
            sa = df.groupby(sk)[mk].mean().reset_index()
            sa.columns = ['Segmen','Nilai']
            sa = sa.sort_values('Nilai', ascending=True)
            fig_sg = px.bar(sa, x='Nilai', y='Segmen', orientation='h',
                color='Nilai', color_continuous_scale='Blues', text='Nilai')
            fig_sg.update_traces(texttemplate='%{x:.2f}', textposition='outside')
        else:
            fig_sg = px.box(df, x=sk, y=mk, color=sk, points='outliers')
            fig_sg.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(elite_layout(fig_sg,
            f"{seg_met.split('→')[1]} per {seg_by.split('→')[1]}", height=400),
            use_container_width=True)

    # ── Frekuensi vs Outcome ──────────────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Frekuensi Transaksi vs Outcome</div>",
                unsafe_allow_html=True)
    fr = df.groupby('S7').agg(
        NPS=('G1A','mean'), Kepuasan=('E1A','mean'), Loyalitas=('F1A','mean'),
        Count=('SERIAL','count')
    ).reset_index()
    fig_fr = go.Figure()
    for cn,cc,nl in [('NPS',COLOR_XYZ,'NPS'),('Kepuasan','#34D399','Kepuasan'),
                     ('Loyalitas','#FBBF24','Loyalitas')]:
        fig_fr.add_trace(go.Bar(name=nl, x=fr['S7'], y=fr[cn],
            marker_color=cc, text=fr[cn].round(2), textposition='outside'))
    fig_fr.update_layout(barmode='group', xaxis_tickangle=-15, height=360)
    st.plotly_chart(elite_layout(fig_fr,"Frekuensi Transaksi vs Outcome"),
        use_container_width=True)

    # ── Tujuan Rekening ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>🎯 Tujuan Buka Rekening vs Loyalitas</div>",
                unsafe_allow_html=True)
    tj = df['A2'].dropna().str.split(';').explode().str.strip()
    tjdf = tj.to_frame('Tujuan').join(df['F1A'],how='left')
    tjagg = tjdf.groupby('Tujuan').agg(
        Loyalitas=('F1A','mean'), Count=('F1A','count')
    ).reset_index()
    tjagg = tjagg[tjagg['Count']>=10].sort_values('Loyalitas')
    fig_tj = px.bar(tjagg, x='Loyalitas', y='Tujuan', orientation='h',
        color='Count', color_continuous_scale='Blues', text='Loyalitas',
        hover_data=['Count'])
    fig_tj.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    fig_tj.update_xaxes(range=[4.5,6.4])
    st.plotly_chart(elite_layout(fig_tj,"Loyalitas per Tujuan Buka Rekening"),
        use_container_width=True)

# =====================================================================
# TAB 8 — VOICE OF CUSTOMER
# =====================================================================
with tab8:
    # KPI VoC
    nv,ppv,pasvv,dv2 = calc_nps(df['G1A_CAT'])
    vk = st.columns(4)
    vk[0].markdown(fmt_card("NPS Score", f"{nv:.0f}", "blue",
        f"{df['G1A_CAT'].eq('Promoter').sum()} promoter"), unsafe_allow_html=True)
    vk[1].markdown(fmt_card("Promoter %", f"{ppv:.0f}%", "green",
        "Pasti rekomendasikan"), unsafe_allow_html=True)
    vk[2].markdown(fmt_card("Passive %", f"{pasvv:.0f}%", "amber",
        "Netral"), unsafe_allow_html=True)
    vk[3].markdown(fmt_card("Detractor %", f"{dv2:.0f}%", "red",
        "Tidak rekomendasikan"), unsafe_allow_html=True)

    # ── Distribusi NPS ────────────────────────────────────────────────
    vh1, vh2 = st.columns(2)
    with vh1:
        fig_nh = px.histogram(df, x='G1A', nbins=11,
            color='G1A_CAT', color_discrete_map=NPS_COLORS,
            labels={'G1A':'Skor NPS XYZ'})
        fig_nh.update_layout(bargap=0.1)
        st.plotly_chart(elite_layout(fig_nh,"Distribusi NPS XYZ"),
            use_container_width=True)
    with vh2:
        if len(df_has_komp)>0:
            fig_nhk = px.histogram(df_has_komp, x='G1C', nbins=11,
                color='G1C_CAT', color_discrete_map=NPS_COLORS,
                labels={'G1C':f'NPS {target_komp}'})
            fig_nhk.update_layout(bargap=0.1)
            st.plotly_chart(elite_layout(fig_nhk,f"Distribusi NPS {target_komp}"),
                use_container_width=True)

    st.markdown("---")

    # ── Wordcloud & Keyword ───────────────────────────────────────────
    st.markdown("<div class='section-header'>☁️ Analisis Teks</div>",
                unsafe_allow_html=True)
    wcf1, wcf2, wcf3 = st.columns(3)
    with wcf1:
        fcat = st.selectbox("Filter NPS:",["Semua","Promoter","Passive","Detractor"])
    with wcf2:
        wsrc = st.selectbox("Sumber:", [
            "G1B — Alasan NPS XYZ",
            "G1D — Alasan NPS Kompetitor",
            "E1AA — Alasan Kepuasan XYZ",
            "E1BB — Alasan Kepuasan Komp"
        ])
    with wcf3:
        mwl = st.slider("Min. panjang kata:", 3, 7, 4)

    cmap_dict = {
        "G1B — Alasan NPS XYZ":       ("G1B","G1A_CAT", df),
        "G1D — Alasan NPS Kompetitor": ("G1D","G1C_CAT", df_has_komp),
        "E1AA — Alasan Kepuasan XYZ":  ("E1AA","G1A_CAT", df),
        "E1BB — Alasan Kepuasan Komp": ("E1BB","G1C_CAT", df_has_komp),
    }
    tcol, cacat, dfsrc = cmap_dict[wsrc]
    dfwc = dfsrc if fcat=="Semua" \
           else dfsrc[dfsrc[cacat]==fcat] if cacat in dfsrc.columns else dfsrc

    def make_wc(series, cmap_name='Blues', mw=80):
        # Parse komentar dulu
        parsed = clean_comments(series)
        text = " ".join(parsed.astype(str).tolist()).lower()
        words = re.findall(rf'\b[a-zA-Z]{{{mwl},}}\b', text)
        wc_words = [w for w in words if w not in stopwords_id]
        if len(wc_words) < 5: return None, []
        wobj = WordCloud(width=700, height=350, background_color='#0F172A',
            colormap=cmap_name, max_words=mw, stopwords=stopwords_id,
            prefer_horizontal=0.8).generate(" ".join(wc_words))
        return wobj, Counter(wc_words).most_common(10)

    wc_obj, top_kw = make_wc(
        dfwc[tcol] if tcol in dfwc.columns else pd.Series(dtype=str)
    )

    wcc1, wcc2 = st.columns([1.6,1])
    with wcc1:
        if wc_obj:
            fig_wc, ax = plt.subplots(figsize=(7,3.5),
                facecolor='#0F172A')
            ax.imshow(wc_obj, interpolation='bilinear')
            ax.axis('off')
            fig_wc.tight_layout(pad=0)
            st.pyplot(fig_wc)
            plt.close()
        else:
            st.info("Tidak ada teks yang cukup.")

    with wcc2:
        if top_kw:
            kdf = pd.DataFrame(top_kw, columns=['Kata','Frekuensi'])
            fig_kw = px.bar(kdf.sort_values('Frekuensi'), x='Frekuensi', y='Kata',
                orientation='h', color='Frekuensi',
                color_continuous_scale='Blues', text='Frekuensi')
            fig_kw.update_traces(textposition='outside')
            fig_kw.update_layout(height=340, showlegend=False)
            st.plotly_chart(elite_layout(fig_kw,"Top 10 Kata Kunci"),
                use_container_width=True)

    st.markdown("---")

    # ── Top tema Promoter vs Detractor ───────────────────────────────
    st.markdown("<div class='section-header'>💡 Tema Utama: Promoter vs Detractor</div>",
                unsafe_allow_html=True)
    pa1, pa2 = st.columns(2)

    def top_theme(series, n=8):
        parsed = clean_comments(series)
        text = " ".join(parsed.astype(str).tolist()).lower()
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        wc = [w for w in words if w not in stopwords_id]
        return Counter(wc).most_common(n)

    with pa1:
        st.markdown("<span style='color:#34D399;font-weight:700;'>🌟 Promoter</span>",
                    unsafe_allow_html=True)
        pw = top_theme(df[df['G1A_CAT']=='Promoter']['G1B'])
        if pw:
            pdf = pd.DataFrame(pw, columns=['Tema','Count'])
            fig_p = px.bar(pdf, x='Count', y='Tema', orientation='h',
                color_discrete_sequence=['#34D399'], text='Count')
            fig_p.update_traces(textposition='outside')
            fig_p.update_layout(height=320)
            st.plotly_chart(elite_layout(fig_p), use_container_width=True)

    with pa2:
        st.markdown("<span style='color:#F87171;font-weight:700;'>⚠️ Detractor</span>",
                    unsafe_allow_html=True)
        dw = top_theme(df[df['G1A_CAT']=='Detractor']['G1B'])
        if dw:
            ddf2 = pd.DataFrame(dw, columns=['Tema','Count'])
            fig_d2 = px.bar(ddf2, x='Count', y='Tema', orientation='h',
                color_discrete_sequence=[COLOR_KOMP], text='Count')
            fig_d2.update_traces(textposition='outside')
            fig_d2.update_layout(height=320)
            st.plotly_chart(elite_layout(fig_d2), use_container_width=True)

    if top_kw:
        insight_box(f"Kata paling dominan ({fcat}): **'{top_kw[0][0]}'** "
                    f"({top_kw[0][1]}x). Cerminan tema utama persepsi nasabah terhadap XYZ.")

    st.markdown("---")

    # ── Verbatim Explorer (komentar diparsing) ────────────────────────
    st.markdown("<div class='section-header'>🔍 Verbatim Explorer</div>",
                unsafe_allow_html=True)
    ve1,ve2,ve3,ve4 = st.columns(4)
    with ve1:
        vnf = st.multiselect("Kategori NPS:",['Promoter','Passive','Detractor'],
            default=['Promoter','Passive','Detractor'])
    with ve2:
        vpf = st.multiselect("Provinsi:",sorted(df['PROV'].dropna().unique()))
    with ve3:
        vsr = st.text_input("🔎 Cari kata kunci:")
    with ve4:
        vms = st.slider("Min. NPS Score:", 0, 10, 0)

    vdf = df[df['G1A_CAT'].isin(vnf)] if vnf else df
    if vpf: vdf = vdf[vdf['PROV'].isin(vpf)]
    vdf = vdf[vdf['G1A'] >= vms]
    if vsr:
        vdf = vdf[vdf['G1B'].fillna('').str.contains(vsr,case=False) |
                  vdf['E1AA'].fillna('').str.contains(vsr,case=False)]

    # Parse komentar sebelum ditampilkan
    vdf_display = vdf.copy()
    if 'G1B' in vdf_display.columns:
        vdf_display['Alasan NPS (Parsed)'] = clean_comments(vdf_display['G1B'])
    if 'E1AA' in vdf_display.columns:
        vdf_display['Alasan Kepuasan (Parsed)'] = clean_comments(vdf_display['E1AA'])

    st.info(f"Menampilkan {len(vdf_display):,} responden")

    disp_cols = {
        'CABANG':'Cabang','PROV':'Provinsi','S1':'Gender','S2_2':'Usia',
        'S4':'Tenure','G1A':'NPS Score','G1A_CAT':'Kategori',
        'E1A':'Kepuasan','Alasan NPS (Parsed)':'Alasan NPS',
        'Alasan Kepuasan (Parsed)':'Alasan Kepuasan'
    }
    existing = {k:v for k,v in disp_cols.items() if k in vdf_display.columns}
    st.dataframe(
        vdf_display[list(existing.keys())].rename(columns=existing)
            .sort_values('NPS Score'),
        use_container_width=True, height=400, hide_index=True
    )

# =====================================================================
# TAB 9 — AI ANALYST
# =====================================================================
with tab9:
    st.markdown("<div class='section-header'>🤖 AI CX Analyst</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <p>Tanyakan apapun tentang data survei ini. AI akan menjawab berdasarkan 
    data aktual yang sedang ditampilkan (sesuai filter aktif). Contoh pertanyaan:
    "Apa kelemahan utama XYZ dibanding kompetitor?", 
    "Cabang mana yang paling perlu diperhatikan?",
    "Apa yang membuat nasabah jadi Promoter?"</p>
    </div>
    """, unsafe_allow_html=True)

    # Build context
    ctx = build_context(df, df_has_komp)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Quick questions
    st.markdown("**⚡ Pertanyaan Cepat:**")
    qq_cols = st.columns(4)
    quick_questions = [
        "Apa kelemahan utama XYZ vs kompetitor?",
        "Cabang mana yang perlu paling diperhatikan?",
        "Apa faktor utama yang mendorong Promoter?",
        "Apa yang membuat nasabah jadi Detractor?"
    ]
    for i, (col_q, qq) in enumerate(zip(qq_cols, quick_questions)):
        if col_q.button(qq, key=f"qq_{i}"):
            st.session_state.chat_history.append({"role":"user","content":qq})
            with st.spinner("AI sedang menganalisis data..."):
                reply = call_claude_api(st.session_state.chat_history, ctx)
            st.session_state.chat_history.append({"role":"assistant","content":reply})
            st.rerun()

    # Chat display
    if st.session_state.chat_history:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(
                    f"<div class='chat-user'><p>👤 {msg['content']}</p></div>"
                    "<div class='chat-clearfix'></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-ai'><p>🤖 {msg['content']}</p></div>"
                    "<div class='chat-clearfix'></div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Input
    ai_c1, ai_c2 = st.columns([4,1])
    with ai_c1:
        user_input = st.text_input("💬 Tanyakan sesuatu tentang data ini:",
            placeholder="Contoh: Apa insight utama dari data ini?",
            key="ai_input")
    with ai_c2:
        st.markdown("<br>", unsafe_allow_html=True)
        send_btn = st.button("Kirim 🚀", use_container_width=True)

    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.spinner("🤖 AI sedang menganalisis..."):
            reply = call_claude_api(st.session_state.chat_history, ctx)
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Reset Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # ── Context Preview ───────────────────────────────────────────────
    with st.expander("📊 Lihat Data Konteks yang Dikirim ke AI"):
        st.code(ctx, language='text')

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#334155 !important; font-size:11px;'>"
    "🚀 Bank XYZ CX Intelligence Dashboard v3.0 | Dark Mode | AI-Powered | "
    "Powered by Streamlit & Plotly</p>",
    unsafe_allow_html=True
)
