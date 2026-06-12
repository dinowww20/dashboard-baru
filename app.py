import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from collections import Counter

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    st.error("Library 'wordcloud' atau 'matplotlib' belum terinstal.")
    st.stop()

# =====================================================================
# 1. KONFIGURASI
# =====================================================================
st.set_page_config(
    page_title="Executive CX Analytics - Bank XYZ",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background-color: #F8FAFC !important; }
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    box-shadow: 2px 0 15px rgba(0,0,0,0.03);
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #0F172A !important; font-weight: 600;
}
.metric-card {
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #E2E8F0; padding: 18px; border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    text-align: center; transition: transform 0.3s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 20px -5px rgba(37,99,235,0.15);
    border-color: #BFDBFE;
}
.metric-title {
    color: #64748B; font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
}
.metric-value { color: #0F172A; font-size: 30px; font-weight: 900; line-height: 1; }
.metric-value.blue  { color: #2563EB; }
.metric-value.green { color: #10B981; }
.metric-value.red   { color: #E11D48; }
.metric-value.amber { color: #F59E0B; }
.metric-sub { color: #94A3B8; font-size: 11px; margin-top: 5px; }
.section-header {
    font-size: 15px; font-weight: 800; color: #0F172A;
    border-left: 4px solid #2563EB; padding-left: 12px;
    margin: 18px 0 10px 0;
}
.insight-box {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border: 1px solid #BFDBFE; border-radius: 12px;
    padding: 14px 18px; margin: 8px 0;
}
.insight-box p { color: #1E40AF !important; font-size: 13px; margin: 0; }
.stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: #ffffff; border: 1px solid #E2E8F0;
    border-radius: 8px 8px 0 0; padding: 10px 18px;
    color: #64748B !important; font-weight: 600; font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background-color: #2563EB !important; color: #ffffff !important;
    border-color: #2563EB;
}
h1, h2, h3, h4 { color: #0F172A !important; font-weight: 800 !important; }
p { color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

NPS_COLORS  = {'Promoter': '#10B981', 'Passive': '#F59E0B', 'Detractor': '#EF4444'}
COLOR_XYZ   = '#2563EB'
COLOR_KOMP  = '#E11D48'
COLOR_IMPRT = '#94A3B8'

def fmt_card(title, value, color="", sub=""):
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
        title=dict(text=title, font=dict(size=14, color='#0F172A')),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=20, r=20),
        font=dict(color='#334155', size=11),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#FFFFFF", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **({"height": height} if height else {})
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9', zeroline=False, automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', zeroline=False, automargin=True)
    return fig

def short_label(col, col_map, max_len=40):
    name = col_map.get(col, col)
    name = re.sub(r'\s*-\s*(XYZ|kompetitor)\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\(.*?\)', '', name).strip()
    return name[:max_len] + "…" if len(name) > max_len else name

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
# 3. SIDEBAR — FILTER GLOBAL
# =====================================================================
st.sidebar.markdown(
    "<h3 style='color:#2563EB !important; font-weight:800;'>🎛️ Filter Analitik</h3>",
    unsafe_allow_html=True
)

# ── Filter Kompetitor ─────────────────────────────────────────────────
st.sidebar.markdown("**🏦 Benchmark Kompetitor**")
komp_list   = sorted(df_raw['KOMP'].dropna().unique().tolist())
target_komp = st.sidebar.selectbox(
    "Pilih Kompetitor:",
    ["Semua Kompetitor (Rata-rata)"] + komp_list,
    help="Pilih bank kompetitor spesifik untuk perbandingan head-to-head"
)

st.sidebar.markdown("---")

# ── Filter Lokasi (Cascading) ─────────────────────────────────────────
st.sidebar.markdown("**📍 Filter Lokasi**")
prov_opts = ["Semua"] + sorted(df_raw['PROV'].dropna().unique().tolist())
sel_prov  = st.sidebar.multiselect("Provinsi", prov_opts[1:])

kota_pool = df_raw[df_raw['PROV'].isin(sel_prov)] if sel_prov else df_raw
kota_opts = sorted(kota_pool['KABKOTA'].dropna().unique().tolist())
sel_kota  = st.sidebar.multiselect("Kab / Kota", kota_opts)

cab_pool = kota_pool[kota_pool['KABKOTA'].isin(sel_kota)] if sel_kota else kota_pool
cab_opts = sorted(cab_pool['CABANG'].dropna().unique().tolist())
sel_cab  = st.sidebar.multiselect("Kantor Cabang", cab_opts)

st.sidebar.markdown("---")

# ── Filter Profil Responden ───────────────────────────────────────────
with st.sidebar.expander("👤 Profil Responden", expanded=False):
    sel_gender = st.multiselect(
        "Jenis Kelamin",
        sorted(df_raw['S1'].dropna().unique().tolist())
    )
    sel_usia = st.multiselect(
        "Rentang Usia",
        sorted(df_raw['S2_2'].dropna().unique().tolist())
    )
    sel_tenure = st.multiselect(
        "Lama Menjadi Nasabah",
        sorted(df_raw['S4'].dropna().unique().tolist())
    )
    sel_frekuensi = st.multiselect(
        "Frekuensi Transaksi",
        sorted(df_raw['S7'].dropna().unique().tolist())
    )
    sel_panel = st.multiselect(
        "Panel (Teller/CS)",
        sorted(df_raw['PANEL'].dropna().unique().tolist())
    )
    sel_sos = st.multiselect(
        "Status Pernikahan",
        sorted(df_raw['P1'].dropna().unique().tolist())
    )
    sel_pekerjaan = st.multiselect(
        "Pekerjaan",
        sorted(df_raw['P4'].dropna().unique().tolist())
    )
    sel_pendidikan = st.multiselect(
        "Pendidikan Terakhir",
        sorted(df_raw['P3'].dropna().unique().tolist())
    )

with st.sidebar.expander("🏦 Perilaku Perbankan", expanded=False):
    sel_bank_utama = st.multiselect(
        "Bank Utama Simpan Dana",
        sorted(df_raw['A1B'].dropna().unique().tolist())
    )
    sel_bank_transaksi = st.multiselect(
        "Bank Utama Transaksi",
        sorted(df_raw['A1C'].dropna().unique().tolist())
    )
    sel_nps_cat = st.multiselect(
        "Kategori NPS",
        ['Promoter', 'Passive', 'Detractor']
    )

# ── Apply Filter ──────────────────────────────────────────────────────
df = df_raw.copy()
if sel_prov:          df = df[df['PROV'].isin(sel_prov)]
if sel_kota:          df = df[df['KABKOTA'].isin(sel_kota)]
if sel_cab:           df = df[df['CABANG'].isin(sel_cab)]
if sel_gender:        df = df[df['S1'].isin(sel_gender)]
if sel_usia:          df = df[df['S2_2'].isin(sel_usia)]
if sel_tenure:        df = df[df['S4'].isin(sel_tenure)]
if sel_frekuensi:     df = df[df['S7'].isin(sel_frekuensi)]
if sel_panel:         df = df[df['PANEL'].isin(sel_panel)]
if sel_sos:           df = df[df['P1'].isin(sel_sos)]
if sel_pekerjaan:     df = df[df['P4'].isin(sel_pekerjaan)]
if sel_pendidikan:    df = df[df['P3'].isin(sel_pendidikan)]
if sel_bank_utama:    df = df[df['A1B'].isin(sel_bank_utama)]
if sel_bank_transaksi:df = df[df['A1C'].isin(sel_bank_transaksi)]
if sel_nps_cat:       df = df[df['G1A_CAT'].isin(sel_nps_cat)]

# Subset kompetitor
df_komp     = df.copy() if target_komp == "Semua Kompetitor (Rata-rata)" \
              else df[df['KOMP'] == target_komp]
df_has_komp = df_komp[df_komp['KOMP'].notna()]

st.sidebar.markdown("---")
st.sidebar.success(f"📊 Total Responden: **{len(df):,}**")
st.sidebar.info(f"🏦 Resp. dgn Kompetitor: **{len(df_has_komp):,}**")
if len(df) < 30:
    st.sidebar.warning("⚠️ Sampel terlalu kecil, hasil mungkin tidak representatif.")

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<h2 style='text-align:center; font-weight:900; letter-spacing:-1px;
color:#0F172A !important; margin-bottom:20px;'>
🏦 BANK XYZ — EXECUTIVE CX INTELLIGENCE DASHBOARD
</h2>""", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ Data kosong. Sesuaikan filter Anda.")
    st.stop()

# =====================================================================
# HELPER: Hitung NPS score
# =====================================================================
def calc_nps(series_cat):
    total = series_cat.notna().sum()
    if total == 0:
        return 0, 0, 0, 0
    prom = (series_cat == 'Promoter').sum()
    pasv = (series_cat == 'Passive').sum()
    detr = (series_cat == 'Detractor').sum()
    score = (prom - detr) / total * 100
    return round(score, 1), round(prom/total*100, 1), round(pasv/total*100, 1), round(detr/total*100, 1)

# =====================================================================
# DIMENSI MAP — Mapping kolom per touchpoint
# =====================================================================
def get_dim_map(df):
    return {
        "Kantor Cabang (Fasilitas)": {
            "imp":  [f"T_KC1_{i}" for i in range(1, 36) if f"T_KC1_{i}" in df.columns],
            "xyz":  [c for c in df.columns if c.startswith("T_KC2_") and
                     c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116",
                                "T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and
                     int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_KC2_") and
                     c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116",
                                "T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and
                     int(c.split("_")[-1]) % 3 == 0],
        },
        "Sekuriti": {
            "imp":  [f"T_SC1_{i}" for i in range(1, 16) if f"T_SC1_{i}" in df.columns],
            "xyz":  [c for c in df.columns if c.startswith("T_SC2_") and
                     c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_SC2_") and
                     c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Teller": {
            "imp":  [f"T_TL2_{i}" for i in range(1, 20) if f"T_TL2_{i}" in df.columns],
            "xyz":  [c for c in df.columns if c.startswith("T_TL3_") and
                     c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_TL3_") and
                     c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Customer Service": {
            "imp":  [f"T_CS2_{i}" for i in range(1, 24) if f"T_CS2_{i}" in df.columns],
            "xyz":  [c for c in df.columns if c.startswith("T_CS3_") and
                     c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_CS3_") and
                     c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Customer Advisor": {
            "imp":  [f"T_CA1_{i}" for i in range(1, 20) if f"T_CA1_{i}" in df.columns],
            "xyz":  [c for c in df.columns if c.startswith("T_CA2_") and c != "T_CA2_20"],
            "komp": [],
        },
        "ATM": {
            "imp":  [f"T_AT2_{i}" for i in range(1, 19) if f"T_AT2_{i}" in df.columns],
            "xyz":  [c for c in df.columns if c.startswith("T_AT3_") and
                     c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_AT3_") and
                     c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1]) % 3 == 0],
        },
    }

DIMENSI_MAP = get_dim_map(df_raw)

# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🌟 Executive Summary",
    "🏢 Kinerja Layanan",
    "🏆 Brand & Kompetitor",
    "🎯 Touchpoint & IPA",
    "💡 Emosi & Loyalitas",
    "📱 Digitalisasi",
    "👥 Profil & Segmentasi",
    "💬 Voice of Customer"
])

# =====================================================================
# TAB 1 — EXECUTIVE SUMMARY
# =====================================================================
with tab1:
    # ── KPI Cards ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📌 Indikator Utama</div>",
                unsafe_allow_html=True)

    nps_score, pct_prom, pct_pasv, pct_detr = calc_nps(df['G1A_CAT'])
    nps_k, pct_pk, pct_pasv_k, pct_dk       = calc_nps(df_has_komp['G1C_CAT']) \
                                                if len(df_has_komp) > 0 \
                                                else (0, 0, 0, 0)
    sat_xyz  = df['E1A'].mean()
    sat_kom  = df_has_komp['E1B'].mean() if len(df_has_komp) > 0 else np.nan
    loy_xyz  = df['F1A'].mean()
    loy_kom  = df_has_komp['F1B'].mean() if len(df_has_komp) > 0 else np.nan

    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.markdown(fmt_card("NPS Score XYZ",     f"{nps_score:.0f}", "blue",
        f"P:{pct_prom:.0f}% | D:{pct_detr:.0f}%"), unsafe_allow_html=True)
    k2.markdown(fmt_card("NPS Kompetitor",    f"{nps_k:.0f}", "red",
        f"P:{pct_pk:.0f}% | D:{pct_dk:.0f}%"), unsafe_allow_html=True)
    k3.markdown(fmt_card("Gap NPS",
        f"+{nps_score-nps_k:.0f}" if nps_score >= nps_k else f"{nps_score-nps_k:.0f}",
        "green" if nps_score >= nps_k else "red",
        "XYZ − Kompetitor"), unsafe_allow_html=True)
    k4.markdown(fmt_card("Kepuasan XYZ",      f"{sat_xyz:.2f}", "blue", "Skala 1–6"),
        unsafe_allow_html=True)
    k5.markdown(fmt_card("Kepuasan Komp",
        f"{sat_kom:.2f}" if not np.isnan(sat_kom) else "N/A", "red", "Skala 1–6"),
        unsafe_allow_html=True)
    k6.markdown(fmt_card("Loyalitas XYZ",     f"{loy_xyz:.2f}", "green", "Skala 1–6"),
        unsafe_allow_html=True)
    k7.markdown(fmt_card("Total Responden",   f"{len(df):,}", "amber",
        f"{len(df_has_komp):,} dgn komp."), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Auto Insight ─────────────────────────────────────────────────
    gap_nps = nps_score - nps_k
    if gap_nps > 50:
        insight_box(f"Bank XYZ unggul sangat signifikan vs kompetitor dengan gap NPS {gap_nps:.0f} poin. "
                    f"Promoter XYZ ({pct_prom:.0f}%) jauh lebih tinggi dari kompetitor ({pct_pk:.0f}%).")
    elif gap_nps > 20:
        insight_box(f"Bank XYZ unggul {gap_nps:.0f} poin NPS di atas kompetitor. "
                    f"Pertahankan keunggulan terutama di area dengan gap terbesar.")
    else:
        insight_box(f"Gap NPS XYZ vs kompetitor hanya {gap_nps:.0f} poin. "
                    f"Perlu perhatian lebih untuk memperlebar keunggulan kompetitif.")

    # ── Baris 2: Donut NPS + Scorecard ───────────────────────────────
    r2c1, r2c2 = st.columns([1, 2.2])

    with r2c1:
        st.markdown("<div class='section-header'>Komposisi NPS XYZ</div>",
                    unsafe_allow_html=True)
        nps_comp = df['G1A_CAT'].value_counts().reset_index()
        nps_comp.columns = ['Kategori','Jumlah']
        fig_donut = px.pie(nps_comp, values='Jumlah', names='Kategori',
            hole=0.6, color='Kategori', color_discrete_map=NPS_COLORS)
        fig_donut.update_traces(
            textposition='outside', textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_donut.add_annotation(
            text=f"NPS<br><b>{nps_score:.0f}</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color='#0F172A')
        )
        st.plotly_chart(elite_layout(fig_donut), use_container_width=True)

        # NPS Kompetitor mini donut
        if len(df_has_komp) > 0:
            nps_comp_k = df_has_komp['G1C_CAT'].value_counts().reset_index()
            nps_comp_k.columns = ['Kategori','Jumlah']
            fig_donut_k = px.pie(nps_comp_k, values='Jumlah', names='Kategori',
                hole=0.6, color='Kategori', color_discrete_map=NPS_COLORS)
            fig_donut_k.update_traces(
                textposition='outside', textinfo='percent+label',
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig_donut_k.add_annotation(
                text=f"NPS Komp<br><b>{nps_k:.0f}</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color='#0F172A')
            )
            st.plotly_chart(
                elite_layout(fig_donut_k, f"NPS {target_komp}"),
                use_container_width=True
            )

    with r2c2:
        st.markdown("<div class='section-header'>Scorecard Dimensi: XYZ vs Kompetitor</div>",
                    unsafe_allow_html=True)
        scorecard_data = [
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
        rows = []
        for label, col_xyz, col_kom in scorecard_data:
            xyz_val = df[col_xyz].mean() if col_xyz in df.columns else np.nan
            kom_val = df_has_komp[col_kom].mean() \
                if col_kom and col_kom in df_has_komp.columns and len(df_has_komp) > 0 \
                else np.nan
            gap = xyz_val - kom_val if not (np.isnan(xyz_val) or np.isnan(kom_val)) else np.nan
            rows.append({
                "Dimensi":       label,
                "XYZ":           round(xyz_val, 2),
                "Kompetitor":    round(kom_val, 2) if not np.isnan(kom_val) else None,
                "Gap":           round(gap, 2) if not np.isnan(gap) else None
            })
        sc_df = pd.DataFrame(rows)

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(
            name='Bank XYZ', y=sc_df['Dimensi'], x=sc_df['XYZ'],
            orientation='h', marker_color=COLOR_XYZ,
            text=sc_df['XYZ'], texttemplate='%{x:.2f}', textposition='outside'
        ))
        fig_sc.add_trace(go.Bar(
            name='Kompetitor', y=sc_df['Dimensi'],
            x=pd.to_numeric(sc_df['Kompetitor'], errors='coerce'),
            orientation='h', marker_color=COLOR_KOMP, opacity=0.85,
            text=sc_df['Kompetitor'], texttemplate='%{x}', textposition='outside'
        ))
        fig_sc.update_layout(barmode='group', xaxis_range=[4, 6.5], height=420)
        st.plotly_chart(elite_layout(fig_sc), use_container_width=True)

    # ── Baris 3: Top/Bottom + Korelasi ───────────────────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("<div class='section-header'>🏆 Top & Bottom 5 Cabang</div>",
                    unsafe_allow_html=True)
        ovr_xyz_cols = [c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c
                        and c not in ["OVR_KC_OPERASIONAL_XYZ","OVR_KC_PARKIR_XYZ",
                                      "OVR_KC_BANKINGHALL_XYZ","OVR_KC_TOILET_XYZ"]]
        if ovr_xyz_cols:
            cab_score = df.groupby('CABANG')[ovr_xyz_cols].mean().mean(axis=1).reset_index()
            cab_score.columns = ['CABANG','Skor']
            top5    = cab_score.nlargest(5,'Skor').assign(Status='🌟 Top 5')
            bottom5 = cab_score.nsmallest(5,'Skor').assign(Status='⚠️ Bottom 5')
            tb_df   = pd.concat([top5, bottom5])
            fig_tb  = px.bar(tb_df, x='Skor', y='CABANG', color='Status',
                orientation='h', text='Skor',
                color_discrete_map={'🌟 Top 5': COLOR_XYZ, '⚠️ Bottom 5': COLOR_KOMP})
            fig_tb.update_traces(texttemplate='%{x:.2f}', textposition='outside')
            fig_tb.update_xaxes(range=[4.5, 6.3])
            st.plotly_chart(elite_layout(fig_tb), use_container_width=True)

    with r3c2:
        st.markdown("<div class='section-header'>📊 Korelasi Indikator Utama</div>",
                    unsafe_allow_html=True)
        corr_map = {
            'NPS': 'G1A', 'Kepuasan': 'E1A', 'Loyalitas': 'F1A',
            'Teller': 'OVR_TELLER_XYZ', 'CS': 'OVR_CS_XYZ',
            'ATM': 'OVR_ATM_XYZ', 'Sekuriti': 'OVR_SEKURITI_XYZ',
            'KC': 'OVR_KC_XYZ'
        }
        valid_map = {k: v for k, v in corr_map.items() if v in df.columns}
        corr_df  = df[list(valid_map.values())].copy()
        corr_df.columns = list(valid_map.keys())
        fig_corr = px.imshow(corr_df.corr(), text_auto=".2f",
            color_continuous_scale='RdBu', aspect='auto', zmin=-1, zmax=1)
        st.plotly_chart(elite_layout(fig_corr, height=380), use_container_width=True)

    # ── Baris 4: NPS Trend per Provinsi ──────────────────────────────
    st.markdown("<div class='section-header'>🗺️ Distribusi NPS per Provinsi</div>",
                unsafe_allow_html=True)
    prov_nps = df.groupby('PROV').agg(
        NPS_Mean=('G1A','mean'),
        Kepuasan=('E1A','mean'),
        Loyalitas=('F1A','mean'),
        Count=('SERIAL','count')
    ).reset_index().sort_values('NPS_Mean', ascending=True)

    fig_prov = px.bar(prov_nps, x='NPS_Mean', y='PROV', orientation='h',
        color='NPS_Mean', color_continuous_scale='Blues',
        text='NPS_Mean', hover_data=['Kepuasan','Loyalitas','Count'])
    fig_prov.update_traces(texttemplate='%{x:.1f}', textposition='outside')
    fig_prov.update_xaxes(range=[7, 10.5])
    st.plotly_chart(elite_layout(fig_prov, "Rata-rata Skor NPS per Provinsi",
        height=400), use_container_width=True)

# =====================================================================
# TAB 2 — KINERJA LAYANAN CABANG
# =====================================================================
with tab2:
    # ── Heatmap cabang ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔥 Heatmap Kinerja Cabang</div>",
                unsafe_allow_html=True)

    # Filter heatmap
    hm_col1, hm_col2 = st.columns([2,1])
    with hm_col1:
        ovr_xyz_all = [c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c]
        ovr_labels  = {c: c.replace("OVR_","").replace("_XYZ","").replace("_"," ").title()
                       for c in ovr_xyz_all}
        sel_hm_dims = st.multiselect(
            "Pilih dimensi untuk heatmap:",
            options=list(ovr_labels.keys()),
            default=ovr_xyz_all,
            format_func=lambda x: ovr_labels[x]
        )
    with hm_col2:
        min_resp = st.slider("Min. responden per cabang:", 5, 30, 10)

    if sel_hm_dims:
        hm_df = df.groupby('CABANG').filter(lambda x: len(x) >= min_resp)
        hm_df = hm_df.groupby('CABANG')[sel_hm_dims].mean().round(2)
        hm_df.columns = [ovr_labels[c] for c in hm_df.columns]
        # Sort by rata-rata skor
        hm_df = hm_df.loc[hm_df.mean(axis=1).sort_values(ascending=False).index]

        fig_heat = px.imshow(
            hm_df, text_auto=".2f",
            color_continuous_scale='RdYlGn',
            aspect='auto', zmin=4, zmax=6
        )
        fig_heat.update_layout(height=max(400, len(hm_df) * 22))
        st.plotly_chart(elite_layout(fig_heat,
            "Heatmap Skor OVR per Cabang (Merah=Rendah, Hijau=Tinggi)"),
            use_container_width=True)

        # Auto insight heatmap
        worst_cab = hm_df.mean(axis=1).idxmin()
        best_cab  = hm_df.mean(axis=1).idxmax()
        insight_box(
            f"Cabang dengan performa terbaik: **{best_cab}** "
            f"(rata-rata {hm_df.loc[best_cab].mean():.2f}). "
            f"Cabang yang perlu perhatian: **{worst_cab}** "
            f"(rata-rata {hm_df.loc[worst_cab].mean():.2f})."
        )

    st.markdown("---")

    # ── Drill-down item level ─────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Drill-Down Item Level</div>",
                unsafe_allow_html=True)

    dd_c1, dd_c2, dd_c3 = st.columns(3)
    with dd_c1:
        sel_dim = st.selectbox("Dimensi:", list(DIMENSI_MAP.keys()))
    with dd_c2:
        drill_mode = st.radio("Tampilan:", ["Bar Chart", "Scatter IPA"], horizontal=True)
    with dd_c3:
        sel_cabang_drill = st.selectbox(
            "Filter Cabang (opsional):",
            ["Semua"] + sorted(df['CABANG'].dropna().unique().tolist())
        )

    df_drill = df if sel_cabang_drill == "Semua" \
               else df[df['CABANG'] == sel_cabang_drill]
    df_drill_komp = df_has_komp if sel_cabang_drill == "Semua" \
                    else df_has_komp[df_has_komp['CABANG'] == sel_cabang_drill]

    dim_info  = DIMENSI_MAP[sel_dim]
    imp_cols  = [c for c in dim_info["imp"]  if c in df_drill.columns]
    xyz_cols  = [c for c in dim_info["xyz"]  if c in df_drill.columns]
    komp_cols = [c for c in dim_info["komp"] if c in df_drill_komp.columns]
    min_len   = min(len(imp_cols), len(xyz_cols))

    if min_len > 0:
        labels   = [short_label(c, col_map) for c in imp_cols[:min_len]]
        imp_vals = [df_drill[c].mean()  for c in imp_cols[:min_len]]
        xyz_vals = [df_drill[c].mean()  for c in xyz_cols[:min_len]]
        kom_vals = [df_drill_komp[c].mean() if c in df_drill_komp.columns else np.nan
                    for c in komp_cols[:min_len]] if komp_cols else []

        if drill_mode == "Bar Chart":
            dc1, dc2 = st.columns([2.5, 1])
            with dc1:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Bar(
                    name='Importance', x=imp_vals, y=labels,
                    orientation='h', marker_color=COLOR_IMPRT, opacity=0.6
                ))
                fig_dd.add_trace(go.Bar(
                    name='Satisfaction XYZ', x=xyz_vals, y=labels,
                    orientation='h', marker_color=COLOR_XYZ
                ))
                if kom_vals:
                    fig_dd.add_trace(go.Bar(
                        name=f'Satisfaction {target_komp}',
                        x=kom_vals, y=labels,
                        orientation='h', marker_color=COLOR_KOMP, opacity=0.8
                    ))
                fig_dd.update_layout(barmode='group', xaxis_range=[3, 6.5],
                                     height=max(350, min_len * 28))
                st.plotly_chart(elite_layout(fig_dd,
                    f"Importance vs Satisfaction — {sel_dim}"),
                    use_container_width=True)

            with dc2:
                gap_sat_imp = pd.DataFrame({
                    'Item': labels,
                    'Gap (Sat−Imp)': [x - i for x, i in zip(xyz_vals, imp_vals)]
                }).sort_values('Gap (Sat−Imp)')
                gap_sat_imp['Warna'] = np.where(
                    gap_sat_imp['Gap (Sat−Imp)'] < 0, COLOR_KOMP, COLOR_XYZ
                )
                fig_g = px.bar(gap_sat_imp, x='Gap (Sat−Imp)', y='Item',
                    orientation='h', text='Gap (Sat−Imp)')
                fig_g.update_traces(
                    marker_color=gap_sat_imp['Warna'],
                    texttemplate='%{x:.2f}', textposition='outside'
                )
                fig_g.update_layout(height=max(350, min_len * 28))
                st.plotly_chart(elite_layout(fig_g, "Gap: Satisfaction − Importance"),
                    use_container_width=True)

                # Auto insight
                worst_item = gap_sat_imp.iloc[0]
                if worst_item['Gap (Sat−Imp)'] < 0:
                    insight_box(
                        f"Item paling kritis pada {sel_dim}: "
                        f"**{worst_item['Item']}** (gap {worst_item['Gap (Sat−Imp)']:.2f}). "
                        f"Importance tinggi tapi satisfaction masih di bawah harapan."
                    )

        else:  # Scatter IPA
            ipa_df = pd.DataFrame({
                'Item': labels,
                'Importance': imp_vals,
                'Satisfaction': xyz_vals
            })
            m_i, m_s = np.nanmean(imp_vals), np.nanmean(xyz_vals)
            fig_ipa = px.scatter(ipa_df, x='Satisfaction', y='Importance',
                text='Item', color_discrete_sequence=[COLOR_XYZ])
            if kom_vals:
                fig_ipa.add_trace(go.Scatter(
                    x=kom_vals, y=imp_vals, mode='markers', name=target_komp,
                    marker=dict(size=10, symbol='x', color=COLOR_KOMP,
                                line=dict(width=2))
                ))
            fig_ipa.update_traces(
                marker=dict(size=12), textposition='top center',
                selector=dict(mode='markers+text')
            )
            fig_ipa.add_vline(x=m_s, line_dash="dash", line_color="#94A3B8")
            fig_ipa.add_hline(y=m_i, line_dash="dash", line_color="#94A3B8")
            for ann_x, ann_y, text, color in [
                (ipa_df['Satisfaction'].min(), ipa_df['Importance'].max(),
                 "⚠️ PERBAIKI SEGERA", COLOR_KOMP),
                (ipa_df['Satisfaction'].max(), ipa_df['Importance'].max(),
                 "🌟 PERTAHANKAN", "#10B981"),
                (ipa_df['Satisfaction'].min(), ipa_df['Importance'].min(),
                 "💤 PRIORITAS RENDAH", "#94A3B8"),
                (ipa_df['Satisfaction'].max(), ipa_df['Importance'].min(),
                 "✅ BERLEBIHAN", "#F59E0B"),
            ]:
                fig_ipa.add_annotation(x=ann_x, y=ann_y, text=text,
                    showarrow=False, font=dict(color=color, size=10),
                    bgcolor="rgba(255,255,255,0.85)")
            fig_ipa.update_layout(height=500)
            st.plotly_chart(elite_layout(fig_ipa, f"IPA Matrix — {sel_dim}"),
                use_container_width=True)

    st.markdown("---")

    # ── Waktu Tunggu ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>⏱️ Analisis Waktu Tunggu</div>",
                unsafe_allow_html=True)
    wt1, wt2, wt3 = st.columns(3)

    with wt1:
        df_tl = df[df['PANEL'] == 'Teller'].dropna(subset=['TL5','TL6'])
        if len(df_tl) > 0:
            wt_df = pd.DataFrame({
                'Metrik': ['Aktual','Toleransi'],
                'Menit':  [df_tl['TL5'].mean(), df_tl['TL6'].mean()]
            })
            fig_wt = px.bar(wt_df, x='Metrik', y='Menit', color='Metrik',
                text='Menit',
                color_discrete_map={'Aktual': COLOR_KOMP, 'Toleransi': COLOR_XYZ})
            fig_wt.update_traces(texttemplate='%{y:.1f} mnt', textposition='outside')
            fig_wt.update_yaxes(range=[0, df_tl['TL6'].mean() * 1.5])
            st.plotly_chart(elite_layout(fig_wt, "⏳ Teller: Aktual vs Toleransi"),
                use_container_width=True)
            insight_box(
                f"Rata-rata tunggu Teller: **{df_tl['TL5'].mean():.1f} menit** "
                f"(toleransi: {df_tl['TL6'].mean():.1f} menit). "
                f"Gap buffer: {df_tl['TL6'].mean() - df_tl['TL5'].mean():.1f} menit."
            )

    with wt2:
        df_cs_wt = df[df['PANEL'] == 'CS'].dropna(subset=['CS5','CS6'])
        if len(df_cs_wt) > 0:
            wt_df2 = pd.DataFrame({
                'Metrik': ['Aktual','Toleransi'],
                'Menit':  [df_cs_wt['CS5'].mean(), df_cs_wt['CS6'].mean()]
            })
            fig_wt2 = px.bar(wt_df2, x='Metrik', y='Menit', color='Metrik',
                text='Menit',
                color_discrete_map={'Aktual': COLOR_KOMP, 'Toleransi': COLOR_XYZ})
            fig_wt2.update_traces(texttemplate='%{y:.1f} mnt', textposition='outside')
            fig_wt2.update_yaxes(range=[0, df_cs_wt['CS6'].mean() * 1.5])
            st.plotly_chart(elite_layout(fig_wt2, "⏳ CS: Aktual vs Toleransi"),
                use_container_width=True)

    with wt3:
        # Jam sibuk Teller & CS
        st.markdown("**⏰ Jam Paling Sibuk**")
        jam_tl = df['TL1'].dropna().value_counts().reset_index()
        jam_tl.columns = ['Jam','Count']
        jam_cs = df['CS1'].dropna().value_counts().reset_index()
        jam_cs.columns = ['Jam','Count']
        fig_jam = go.Figure()
        fig_jam.add_trace(go.Bar(
            name='Teller', x=jam_tl['Jam'], y=jam_tl['Count'],
            marker_color=COLOR_XYZ
        ))
        fig_jam.add_trace(go.Bar(
            name='CS', x=jam_cs['Jam'], y=jam_cs['Count'],
            marker_color=COLOR_KOMP
        ))
        fig_jam.update_layout(barmode='group', xaxis_tickangle=-30)
        st.plotly_chart(elite_layout(fig_jam, "Jam Paling Sibuk: Teller vs CS"),
            use_container_width=True)

        # Top jam sibuk
        if len(jam_tl) > 0:
            top_jam = jam_tl.iloc[0]['Jam']
            insight_box(f"Jam paling sibuk untuk Teller: **{top_jam}**. "
                        f"Pertimbangkan penambahan staf di jam ini.")

# =====================================================================
# TAB 3 — BRAND & KOMPETITOR
# =====================================================================
with tab3:
    # ── 24 atribut brand ─────────────────────────────────────────────
    brand_imp_cols = [f"T_C1A_{i}" for i in range(1,25) if f"T_C1A_{i}" in df.columns]
    brand_xyz_cols = sorted(
        [c for c in df.columns if c.startswith("T_C1B_") and
         int(c.split("_")[-1]) % 3 == 2],
        key=lambda x: int(x.split("_")[-1])
    )
    brand_kom_cols = sorted(
        [c for c in df.columns if c.startswith("T_C1B_") and
         int(c.split("_")[-1]) % 3 == 0],
        key=lambda x: int(x.split("_")[-1])
    )

    brand_labels = [short_label(c, col_map) for c in brand_imp_cols]
    brand_imp    = [df[c].mean() for c in brand_imp_cols]
    brand_xyz    = [df[c].mean() for c in brand_xyz_cols[:len(brand_imp_cols)]]
    brand_kom    = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                    for c in brand_kom_cols[:len(brand_imp_cols)]]

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("<div class='section-header'>🕸️ Radar Brand: XYZ vs Kompetitor</div>",
                    unsafe_allow_html=True)
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=brand_xyz, theta=brand_labels, fill='toself',
            name='Bank XYZ', line_color=COLOR_XYZ,
            fillcolor=f'rgba(37,99,235,0.15)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=brand_kom, theta=brand_labels, fill='toself',
            name=target_komp, line_color=COLOR_KOMP,
            fillcolor=f'rgba(225,29,72,0.1)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[3,6])),
            legend=dict(orientation="h", y=-0.15), height=500
        )
        st.plotly_chart(elite_layout(fig_radar), use_container_width=True)

    with b2:
        st.markdown("<div class='section-header'>🎯 IPA Matrix — 24 Atribut Brand</div>",
                    unsafe_allow_html=True)
        brand_df = pd.DataFrame({
            'Label': brand_labels,
            'Importance': brand_imp,
            'Satisfaction': brand_xyz
        })
        m_imp = np.nanmean(brand_imp)
        m_sat = np.nanmean(brand_xyz)

        # Beri warna per kuadran
        def quadrant(row):
            if row['Importance'] >= m_imp and row['Satisfaction'] < m_sat:
                return "⚠️ Perbaiki Segera"
            elif row['Importance'] >= m_imp and row['Satisfaction'] >= m_sat:
                return "🌟 Pertahankan"
            elif row['Importance'] < m_imp and row['Satisfaction'] < m_sat:
                return "💤 Prioritas Rendah"
            else:
                return "✅ Berlebihan"

        brand_df['Kuadran'] = brand_df.apply(quadrant, axis=1)
        q_colors = {
            "⚠️ Perbaiki Segera": COLOR_KOMP,
            "🌟 Pertahankan": "#10B981",
            "💤 Prioritas Rendah": "#94A3B8",
            "✅ Berlebihan": "#F59E0B"
        }
        fig_ipa_b = px.scatter(brand_df, x='Satisfaction', y='Importance',
            text='Label', color='Kuadran', color_discrete_map=q_colors,
            hover_data=['Kuadran'])
        fig_ipa_b.update_traces(
            marker=dict(size=11), textposition='top center',
            textfont=dict(size=9)
        )
        fig_ipa_b.add_vline(x=m_sat, line_dash="dash", line_color="#CBD5E1")
        fig_ipa_b.add_hline(y=m_imp, line_dash="dash", line_color="#CBD5E1")
        fig_ipa_b.update_layout(height=500)
        st.plotly_chart(elite_layout(fig_ipa_b), use_container_width=True)

    # ── Gap bar chart ─────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Gap Kompetitif per Atribut Brand</div>",
                unsafe_allow_html=True)
    gap_b = pd.DataFrame({
        'Atribut': brand_labels,
        'XYZ': brand_xyz,
        'Kompetitor': brand_kom,
        'Gap': [x - k for x, k in zip(brand_xyz, brand_kom)]
    }).sort_values('Gap')
    gap_b['Warna'] = np.where(gap_b['Gap'] < 0, COLOR_KOMP, COLOR_XYZ)

    fig_gap_b = px.bar(gap_b, x='Gap', y='Atribut', orientation='h',
        text='Gap', hover_data=['XYZ','Kompetitor'])
    fig_gap_b.update_traces(
        marker_color=gap_b['Warna'],
        texttemplate='%{x:.2f}', textposition='outside'
    )
    st.plotly_chart(
        elite_layout(fig_gap_b, f"Gap Brand XYZ vs {target_komp} (+ = XYZ unggul)",
            height=500),
        use_container_width=True
    )

    # Auto insight brand gap
    top_gap  = gap_b.nlargest(1, 'Gap').iloc[0]
    bot_gap  = gap_b.nsmallest(1, 'Gap').iloc[0]
    insight_box(
        f"Keunggulan terbesar XYZ: **{top_gap['Atribut']}** "
        f"(gap +{top_gap['Gap']:.2f}). "
        f"Atribut yang perlu ditingkatkan: **{bot_gap['Atribut']}** "
        f"(gap {bot_gap['Gap']:.2f})."
    )

    # ── Share of wallet ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏦 Share of Wallet</div>",
                unsafe_allow_html=True)
    sw1, sw2, sw3 = st.columns(3)

    with sw1:
        banks_other = df['A1AX'].dropna().str.split(';').explode().str.strip()
        banks_other = banks_other[banks_other != '']
        bc = banks_other.value_counts().head(8).reset_index()
        bc.columns = ['Bank','Jumlah']
        fig_sw = px.bar(bc, x='Jumlah', y='Bank', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_sw.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_sw, "Bank Lain yang Aktif Digunakan"),
            use_container_width=True)

    with sw2:
        simpan = df['A1B'].value_counts().reset_index()
        simpan.columns = ['Bank','Jumlah']
        fig_s = px.pie(simpan, values='Jumlah', names='Bank', hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(elite_layout(fig_s, "Bank Utama Simpan Dana"),
            use_container_width=True)

    with sw3:
        transaksi = df['A1C'].value_counts().reset_index()
        transaksi.columns = ['Bank','Jumlah']
        fig_t = px.pie(transaksi, values='Jumlah', names='Bank', hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(elite_layout(fig_t, "Bank Utama Bertransaksi"),
            use_container_width=True)

    # Insight share of wallet
    top_rival = bc.iloc[0]['Bank'] if len(bc) > 0 else "N/A"
    xyz_as_main_save  = (df['A1B'] == 'Bank XYZ').mean() * 100
    xyz_as_main_trans = (df['A1C'] == 'Bank XYZ').mean() * 100
    insight_box(
        f"XYZ menjadi rekening utama simpan dana bagi **{xyz_as_main_save:.0f}%** responden "
        f"dan utama transaksi bagi **{xyz_as_main_trans:.0f}%** responden. "
        f"Kompetitor terbesar: **{top_rival}**."
    )

# =====================================================================
# TAB 4 — TOUCHPOINT & IPA
# =====================================================================
with tab4:
    st.markdown("<div class='section-header'>🎯 IPA Matrix per Touchpoint</div>",
                unsafe_allow_html=True)

    tp1, tp2, tp3 = st.columns(3)
    with tp1:
        sel_tp = st.selectbox("Touchpoint:", list(DIMENSI_MAP.keys()))
    with tp2:
        tp_view = st.radio("Tampilan:", ["IPA Scatter","Bar Comparison"], horizontal=True)
    with tp3:
        tp_cab_filter = st.selectbox(
            "Filter Cabang:",
            ["Semua"] + sorted(df['CABANG'].unique().tolist())
        )

    df_tp      = df if tp_cab_filter == "Semua" else df[df['CABANG'] == tp_cab_filter]
    df_tp_komp = df_has_komp if tp_cab_filter == "Semua" \
                 else df_has_komp[df_has_komp['CABANG'] == tp_cab_filter]

    dim_tp     = DIMENSI_MAP[sel_tp]
    imp_tp     = [c for c in dim_tp["imp"]  if c in df_tp.columns]
    xyz_tp     = [c for c in dim_tp["xyz"]  if c in df_tp.columns]
    komp_tp    = [c for c in dim_tp["komp"] if c in df_tp_komp.columns]
    min_tp     = min(len(imp_tp), len(xyz_tp))

    if min_tp > 0:
        labels_tp = [short_label(c, col_map) for c in imp_tp[:min_tp]]
        imp_v     = [df_tp[c].mean()          for c in imp_tp[:min_tp]]
        xyz_v     = [df_tp[c].mean()          for c in xyz_tp[:min_tp]]
        kom_v     = [df_tp_komp[c].mean() if c in df_tp_komp.columns else np.nan
                     for c in komp_tp[:min_tp]] if komp_tp else []

        if tp_view == "IPA Scatter":
            tp_col1, tp_col2 = st.columns([1.6, 1])
            with tp_col1:
                ipa_tp_df = pd.DataFrame({
                    'Item': labels_tp,
                    'Importance': imp_v,
                    'Satisfaction XYZ': xyz_v
                })
                m_i_tp = np.nanmean(imp_v)
                m_s_tp = np.nanmean(xyz_v)
                ipa_tp_df['Kuadran'] = ipa_tp_df.apply(
                    lambda r: "⚠️ Perbaiki" if r['Importance'] >= m_i_tp and r['Satisfaction XYZ'] < m_s_tp
                    else "🌟 Pertahankan" if r['Importance'] >= m_i_tp and r['Satisfaction XYZ'] >= m_s_tp
                    else "💤 Rendah" if r['Importance'] < m_i_tp and r['Satisfaction XYZ'] < m_s_tp
                    else "✅ Lebih", axis=1
                )
                q_col = {"⚠️ Perbaiki": COLOR_KOMP, "🌟 Pertahankan": "#10B981",
                         "💤 Rendah": "#94A3B8", "✅ Lebih": "#F59E0B"}

                fig_ipa_tp = px.scatter(ipa_tp_df, x='Satisfaction XYZ', y='Importance',
                    text='Item', color='Kuadran', color_discrete_map=q_col)
                if kom_v:
                    fig_ipa_tp.add_trace(go.Scatter(
                        x=kom_v, y=imp_v, mode='markers', name=target_komp,
                        marker=dict(size=10, symbol='x', color=COLOR_KOMP,
                                    line=dict(width=2))
                    ))
                fig_ipa_tp.update_traces(
                    marker=dict(size=12), textposition='top center',
                    textfont=dict(size=9), selector=dict(mode='markers+text')
                )
                fig_ipa_tp.add_vline(x=m_s_tp, line_dash="dash", line_color="#CBD5E1")
                fig_ipa_tp.add_hline(y=m_i_tp, line_dash="dash", line_color="#CBD5E1")
                fig_ipa_tp.update_layout(height=480)
                st.plotly_chart(elite_layout(fig_ipa_tp, f"IPA Matrix — {sel_tp}"),
                    use_container_width=True)

            with tp_col2:
                # Tabel prioritas per kuadran
                st.markdown("**📋 Prioritas per Kuadran:**")
                for q_name, q_col_val in q_col.items():
                    items_q = ipa_tp_df[ipa_tp_df['Kuadran'] == q_name]['Item'].tolist()
                    if items_q:
                        st.markdown(
                            f"<span style='color:{q_col_val}; font-weight:700;'>{q_name}</span>",
                            unsafe_allow_html=True
                        )
                        for it in items_q:
                            st.markdown(f"  • {it}")

        else:  # Bar Comparison
            fig_bar_tp = go.Figure()
            fig_bar_tp.add_trace(go.Bar(
                name='Importance', x=labels_tp, y=imp_v,
                marker_color=COLOR_IMPRT, opacity=0.7
            ))
            fig_bar_tp.add_trace(go.Bar(
                name='Satisfaction XYZ', x=labels_tp, y=xyz_v,
                marker_color=COLOR_XYZ
            ))
            if kom_v:
                fig_bar_tp.add_trace(go.Bar(
                    name=f'Satisfaction {target_komp}', x=labels_tp, y=kom_v,
                    marker_color=COLOR_KOMP, opacity=0.8
                ))
            fig_bar_tp.update_layout(barmode='group', yaxis_range=[3, 6.5],
                                      xaxis_tickangle=-30, height=450)
            st.plotly_chart(
                elite_layout(fig_bar_tp, f"Comparison — {sel_tp}"),
                use_container_width=True
            )

        # Gap kompetitif
        if kom_v:
            st.markdown("<div class='section-header'>📊 Gap Kompetitif per Item</div>",
                        unsafe_allow_html=True)
            gap_tp_df = pd.DataFrame({
                'Item': labels_tp,
                'Gap': [x - k for x, k in zip(xyz_v, kom_v)]
            }).sort_values('Gap')
            gap_tp_df['Warna'] = np.where(gap_tp_df['Gap'] < 0, COLOR_KOMP, COLOR_XYZ)
            fig_gap_tp = px.bar(gap_tp_df, x='Gap', y='Item', orientation='h',
                text='Gap')
            fig_gap_tp.update_traces(
                marker_color=gap_tp_df['Warna'],
                texttemplate='%{x:.2f}', textposition='outside'
            )
            st.plotly_chart(
                elite_layout(fig_gap_tp, f"Gap XYZ vs {target_komp} — {sel_tp}",
                    height=400),
                use_container_width=True
            )

    st.markdown("---")

    # ── Jenis transaksi vs skor ───────────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Jenis Transaksi vs Skor Layanan</div>",
                unsafe_allow_html=True)
    d1_map = {
        'TELLER': df[df['D1_TYPE']=='TELLER']['OVR_TELLER_XYZ'].mean(),
        'CS':     df[df['D1_TYPE']=='CS']['OVR_CS_XYZ'].mean(),
        'BOTH':   df[df['D1_TYPE']=='BOTH'][['OVR_TELLER_XYZ','OVR_CS_XYZ']].mean(axis=1).mean()
    }
    d1_df = pd.DataFrame({
        'Jenis': list(d1_map.keys()),
        'Skor':  list(d1_map.values())
    })
    fig_d1 = px.bar(d1_df, x='Jenis', y='Skor', color='Jenis',
        text='Skor',
        color_discrete_map={'TELLER': COLOR_XYZ, 'CS': COLOR_KOMP, 'BOTH': '#F59E0B'})
    fig_d1.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig_d1.update_yaxes(range=[4.5, 6.3])
    st.plotly_chart(elite_layout(fig_d1,
        "Rata-rata Skor Berdasarkan Jenis Transaksi Hari Ini"),
        use_container_width=True)

# =====================================================================
# TAB 5 — EMOSI & LOYALITAS
# =====================================================================
with tab5:
    # Kolom emosi
    emo_pos_xyz_cols = [c for c in ["T_I1A_2","T_I1A_5","T_I1A_8","T_I1A_11",
        "T_I1A_14","T_I1A_17","T_I1A_20","T_I1A_23","T_I1A_26"] if c in df.columns]
    emo_neg_xyz_cols = [c for c in ["T_I1A_29","T_I1A_32","T_I1A_35","T_I1A_38",
        "T_I1A_41","T_I1A_44","T_I1A_47"] if c in df.columns]
    emo_pos_kom_cols = [c for c in ["T_I1A_3","T_I1A_6","T_I1A_9","T_I1A_12",
        "T_I1A_15","T_I1A_18","T_I1A_21","T_I1A_24","T_I1A_27"] if c in df.columns]
    emo_neg_kom_cols = [c for c in ["T_I1A_30","T_I1A_33","T_I1A_36","T_I1A_39",
        "T_I1A_42","T_I1A_45","T_I1A_48"] if c in df.columns]

    emo_labels_pos = ["Bahagia","Percaya","Dihargai","Diperhatikan",
                      "Aman","Fokus","Dimanjakan","Tertarik","Penuh Semangat"]
    emo_labels_neg = ["Tidak Puas","Frustasi","Kecewa","Tertekan",
                      "Tidak Bahagia","Diabaikan","Tergesa-gesa"]

    emo_pos_xyz = [df[c].mean()             for c in emo_pos_xyz_cols]
    emo_neg_xyz = [df[c].mean()             for c in emo_neg_xyz_cols]
    emo_pos_kom = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                   for c in emo_pos_kom_cols]
    emo_neg_kom = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                   for c in emo_neg_kom_cols]

    # Score agregat emosi
    emo_pos_score_xyz = np.nanmean(emo_pos_xyz)
    emo_neg_score_xyz = np.nanmean(emo_neg_xyz)
    emo_pos_score_kom = np.nanmean(emo_pos_kom)
    emo_neg_score_kom = np.nanmean(emo_neg_kom)

    # KPI emosi
    e_k1, e_k2, e_k3, e_k4 = st.columns(4)
    e_k1.markdown(fmt_card("Emosi Positif XYZ",  f"{emo_pos_score_xyz:.2f}", "blue",
        "Avg 9 dimensi positif"), unsafe_allow_html=True)
    e_k2.markdown(fmt_card("Emosi Negatif XYZ",  f"{emo_neg_score_xyz:.2f}", "amber",
        "Makin rendah makin baik"), unsafe_allow_html=True)
    e_k3.markdown(fmt_card("Emosi Positif Komp", f"{emo_pos_score_kom:.2f}", "red",
        "Avg 9 dimensi positif"), unsafe_allow_html=True)
    e_k4.markdown(fmt_card("Emosi Negatif Komp", f"{emo_neg_score_kom:.2f}", "amber",
        "Makin rendah makin baik"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    insight_box(
        f"XYZ unggul di emosi positif ({emo_pos_score_xyz:.2f} vs {emo_pos_score_kom:.2f}) "
        f"dan lebih rendah di emosi negatif ({emo_neg_score_xyz:.2f} vs {emo_neg_score_kom:.2f}), "
        f"menunjukkan pengalaman nasabah XYZ lebih positif secara keseluruhan."
    )

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("<div class='section-header'>😊 Emosi Positif</div>",
                    unsafe_allow_html=True)
        fig_ep = go.Figure()
        fig_ep.add_trace(go.Bar(
            name='XYZ', x=emo_labels_pos[:len(emo_pos_xyz)], y=emo_pos_xyz,
            marker_color=COLOR_XYZ, text=np.round(emo_pos_xyz, 2), textposition='outside'
        ))
        fig_ep.add_trace(go.Bar(
            name=target_komp, x=emo_labels_pos[:len(emo_pos_kom)], y=emo_pos_kom,
            marker_color=COLOR_KOMP, text=np.round(emo_pos_kom, 2), textposition='outside'
        ))
        fig_ep.update_layout(barmode='group', yaxis_range=[3, 6.8], height=380)
        st.plotly_chart(elite_layout(fig_ep), use_container_width=True)

    with e2:
        st.markdown("<div class='section-header'>😠 Emosi Negatif</div>",
                    unsafe_allow_html=True)
        fig_en = go.Figure()
        fig_en.add_trace(go.Bar(
            name='XYZ', x=emo_labels_neg[:len(emo_neg_xyz)], y=emo_neg_xyz,
            marker_color=COLOR_XYZ, text=np.round(emo_neg_xyz, 2), textposition='outside'
        ))
        fig_en.add_trace(go.Bar(
            name=target_komp, x=emo_labels_neg[:len(emo_neg_kom)], y=emo_neg_kom,
            marker_color=COLOR_KOMP, text=np.round(emo_neg_kom, 2), textposition='outside'
        ))
        fig_en.update_layout(barmode='group', yaxis_range=[1, 4], height=380)
        st.plotly_chart(elite_layout(fig_en,
            "Makin Rendah = Makin Baik"), use_container_width=True)

    st.markdown("---")

    # ── Brand Equity T_H1A_ ───────────────────────────────────────────
    st.markdown("<div class='section-header'>💎 Brand Equity — 15 Atribut Loyalitas</div>",
                unsafe_allow_html=True)

    h1a_xyz_cols = [c for c in ["T_H1A_2","T_H1A_5","T_H1A_8","T_H1A_11","T_H1A_14",
        "T_H1A_17","T_H1A_20","T_H1A_23","T_H1A_26","T_H1A_29",
        "T_H1A_32","T_H1A_35","T_H1A_38","T_H1A_41","T_H1A_44"]
        if c in df.columns]
    h1a_kom_cols = [c for c in ["T_H1A_3","T_H1A_6","T_H1A_9","T_H1A_12","T_H1A_15",
        "T_H1A_18","T_H1A_21","T_H1A_24","T_H1A_27","T_H1A_30",
        "T_H1A_33","T_H1A_36","T_H1A_39","T_H1A_42","T_H1A_45"]
        if c in df_has_komp.columns]
    h1a_labels = [
        "Tetap Gunakan","Kemudahan Transaksi","Digunakan Banyak Orang",
        "Keuntungan Finansial","Produk Lengkap","Promo Gaya Hidup",
        "Kecepatan Transaksi","Rasa Aman","Kenyamanan Fasilitas",
        "Merasa Dihargai","Bangga","Up to Date/Modern",
        "Bank Turun-Temurun","Cukup Satu Bank","Bergengsi"
    ]

    h_xyz = [df[c].mean() for c in h1a_xyz_cols]
    h_kom = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
             for c in h1a_kom_cols]

    labels_h = h1a_labels[:min(len(h_xyz), len(h1a_labels))]
    fig_h1a = go.Figure()
    fig_h1a.add_trace(go.Bar(
        name='Bank XYZ', x=labels_h, y=h_xyz[:len(labels_h)],
        marker_color=COLOR_XYZ, text=np.round(h_xyz[:len(labels_h)], 2),
        textposition='outside'
    ))
    fig_h1a.add_trace(go.Bar(
        name=target_komp, x=labels_h, y=h_kom[:len(labels_h)],
        marker_color=COLOR_KOMP, text=np.round(h_kom[:len(labels_h)], 2),
        textposition='outside'
    ))
    fig_h1a.update_layout(barmode='group', yaxis_range=[3, 6.8],
                           xaxis_tickangle=-30, height=400)
    st.plotly_chart(elite_layout(fig_h1a, "Brand Equity XYZ vs Kompetitor"),
        use_container_width=True)

    # ── Korelasi emosi → outcome ──────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Korelasi: Emosi vs Outcome</div>",
                unsafe_allow_html=True)
    emo_pos_avg = df[[c for c in emo_pos_xyz_cols if c in df.columns]].mean(axis=1)
    emo_neg_avg = df[[c for c in emo_neg_xyz_cols if c in df.columns]].mean(axis=1)
    corr_emo = pd.DataFrame({
        'Emosi Positif': emo_pos_avg,
        'Emosi Negatif': emo_neg_avg,
        'NPS':           df['G1A'],
        'Kepuasan':      df['E1A'],
        'Loyalitas':     df['F1A'],
    }).corr()
    fig_ec = px.imshow(corr_emo, text_auto=".2f",
        color_continuous_scale='RdBu', aspect='auto', zmin=-1, zmax=1)
    st.plotly_chart(elite_layout(fig_ec,
        "Korelasi Emosi vs NPS, Kepuasan, Loyalitas"),
        use_container_width=True)

    # Scatter emosi positif vs NPS
    ec1, ec2 = st.columns(2)
    with ec1:
        scatter_df = df[['G1A','E1A','F1A','G1A_CAT']].copy()
        scatter_df['Emosi Positif'] = emo_pos_avg
        fig_sc_e = px.scatter(scatter_df, x='Emosi Positif', y='G1A',
            color='G1A_CAT', color_discrete_map=NPS_COLORS,
            trendline='ols', opacity=0.6,
            labels={'G1A': 'Skor NPS'})
        st.plotly_chart(elite_layout(fig_sc_e, "Emosi Positif vs NPS"),
            use_container_width=True)
    with ec2:
        fig_sc_e2 = px.scatter(scatter_df, x='Emosi Positif', y='F1A',
            color='G1A_CAT', color_discrete_map=NPS_COLORS,
            trendline='ols', opacity=0.6,
            labels={'F1A': 'Loyalitas'})
        st.plotly_chart(elite_layout(fig_sc_e2, "Emosi Positif vs Loyalitas"),
            use_container_width=True)

# =====================================================================
# TAB 6 — DIGITALISASI
# =====================================================================
with tab6:
    st.markdown("<div class='section-header'>📱 Persepsi Digitalisasi Cabang</div>",
                unsafe_allow_html=True)

    dig_cols   = [c for c in ["T_J1_1","T_J1_2","T_J1_3","T_J1_4","T_J1_5"]
                  if c in df.columns]
    dig_labels = [
        "Digitalisasi Layanan Cabang",
        "Digital Signage",
        "Smart Table",
        "Tablet Survey",
        "Akses Cabang"
    ]
    dig_vals = [df[c].mean() for c in dig_cols]

    # KPI cards digitalisasi
    dg_cols = st.columns(len(dig_cols))
    for i, (col_w, label, val) in enumerate(zip(dg_cols, dig_labels, dig_vals)):
        color = "green" if val >= 5.5 else ("amber" if val >= 4.5 else "red")
        col_w.markdown(fmt_card(label, f"{val:.2f}", color, "Skala 1–6"),
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    dg1, dg2 = st.columns(2)
    with dg1:
        fig_dig = px.bar(
            x=dig_vals, y=dig_labels[:len(dig_vals)], orientation='h',
            color=dig_vals, color_continuous_scale='Blues',
            text=np.round(dig_vals, 2)
        )
        fig_dig.update_traces(textposition='outside')
        fig_dig.update_xaxes(range=[3, 6.8])
        st.plotly_chart(elite_layout(fig_dig, "Skor Persepsi Digitalisasi"),
            use_container_width=True)

    with dg2:
        # Digitalisasi per provinsi
        dig_prov_col = "T_J1_1" if "T_J1_1" in df.columns else None
        if dig_prov_col:
            dig_prov = df.groupby('PROV')[dig_prov_col].mean().reset_index()
            dig_prov.columns = ['Provinsi','Skor']
            dig_prov = dig_prov.sort_values('Skor', ascending=True)
            fig_dp = px.bar(dig_prov, x='Skor', y='Provinsi', orientation='h',
                color='Skor', color_continuous_scale='Blues', text='Skor')
            fig_dp.update_traces(texttemplate='%{x:.2f}', textposition='outside')
            fig_dp.update_xaxes(range=[3, 6.8])
            st.plotly_chart(elite_layout(fig_dp, "Digitalisasi per Provinsi"),
                use_container_width=True)

    st.markdown("---")

    # ── Sarana elektronik ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🖥️ Ketersediaan & Fungsi Sarana Elektronik</div>",
                unsafe_allow_html=True)

    sl_cols   = [f"T_SL2_{i}" for i in range(1,17) if f"T_SL2_{i}" in df.columns]
    sl_labels = [short_label(c, col_map) for c in sl_cols]
    sl_vals   = [df[c].mean() for c in sl_cols]

    sl1, sl2 = st.columns(2)
    with sl1:
        fig_sl = px.bar(x=sl_vals, y=sl_labels[:len(sl_vals)], orientation='h',
            color=sl_vals, color_continuous_scale='Teal', text=np.round(sl_vals, 2))
        fig_sl.update_traces(textposition='outside')
        fig_sl.update_xaxes(range=[3, 6.8])
        st.plotly_chart(elite_layout(fig_sl, "Skor Sarana Elektronik Layanan"),
            use_container_width=True)

    with sl2:
        # Scatter: Digitalisasi vs Kepuasan
        if "T_J1_1" in df.columns:
            sc_dig = df[['T_J1_1','E1A','G1A','G1A_CAT']].dropna()
            fig_dig_sat = px.scatter(sc_dig, x='T_J1_1', y='E1A',
                color='G1A_CAT', color_discrete_map=NPS_COLORS,
                trendline='ols', opacity=0.6,
                labels={'T_J1_1': 'Persepsi Digitalisasi', 'E1A': 'Kepuasan Overall'})
            st.plotly_chart(elite_layout(fig_dig_sat,
                "Korelasi: Persepsi Digitalisasi vs Kepuasan"),
                use_container_width=True)

    # ── E-form adoption ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>📋 Adopsi E-Form Pembukaan Rekening</div>",
                unsafe_allow_html=True)
    ef1, ef2 = st.columns(2)

    with ef1:
        if 'D2' in df.columns:
            eform = df['D2'].dropna().value_counts().reset_index()
            eform.columns = ['Status','Jumlah']
            fig_ef = px.pie(eform, values='Jumlah', names='Status', hole=0.55,
                color_discrete_sequence=['#8B5CF6','#D946EF','#F43F5E'])
            st.plotly_chart(elite_layout(fig_ef, "Status Penggunaan E-Form"),
                use_container_width=True)

    with ef2:
        if 'D4' in df.columns:
            eform_aware = df['D4'].dropna().value_counts().reset_index()
            eform_aware.columns = ['Status','Jumlah']
            fig_ea = px.bar(eform_aware, x='Jumlah', y='Status', orientation='h',
                color_discrete_sequence=['#0EA5E9'], text='Jumlah')
            fig_ea.update_traces(textposition='outside')
            st.plotly_chart(elite_layout(fig_ea, "Awareness E-Form"),
                use_container_width=True)

    # Saran perbaikan digitalisasi
    st.markdown("<div class='section-header'>📝 Saran Perbaikan Digitalisasi</div>",
                unsafe_allow_html=True)
    j2_map = {
        "T_J2_1": "Digitalisasi Layanan",
        "T_J2_2": "Digital Signage",
        "T_J2_3": "Smart Table",
        "T_J2_4": "Tablet Survey",
        "T_J2_5": "Akses Cabang"
    }
    sel_j2_label = st.selectbox("Topik saran:", list(j2_map.values()))
    sel_j2_col   = [k for k,v in j2_map.items() if v == sel_j2_label][0]
    if sel_j2_col in df.columns:
        saran = df[sel_j2_col].dropna().value_counts().reset_index()
        saran.columns = ['Saran','Jumlah']
        saran = saran[saran['Saran'].str.strip() != '']
        if len(saran) > 0:
            st.dataframe(saran, use_container_width=True, hide_index=True, height=200)

# =====================================================================
# TAB 7 — PROFIL & SEGMENTASI
# =====================================================================
with tab7:
    st.markdown("<div class='section-header'>👥 Profil Demografis</div>",
                unsafe_allow_html=True)

    d1c, d2c, d3c, d4c = st.columns(4)
    with d1c:
        fig_g = px.pie(df, names='S1', hole=0.55,
            color_discrete_sequence=[COLOR_XYZ, '#60A5FA'])
        st.plotly_chart(elite_layout(fig_g, "Gender"), use_container_width=True)
    with d2c:
        age_c = df['S2_2'].value_counts().reset_index()
        age_c.columns = ['Usia','Jumlah']
        fig_a = px.bar(age_c, x='Usia', y='Jumlah',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_a.update_traces(textposition='outside')
        fig_a.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(elite_layout(fig_a, "Kelompok Usia"), use_container_width=True)
    with d3c:
        edu_c = df['P3'].value_counts().reset_index()
        edu_c.columns = ['Pendidikan','Jumlah']
        fig_e = px.bar(edu_c, x='Jumlah', y='Pendidikan', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_e.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_e, "Pendidikan"), use_container_width=True)
    with d4c:
        job_c = df['P4'].value_counts().reset_index()
        job_c.columns = ['Pekerjaan','Jumlah']
        fig_j = px.bar(job_c, x='Jumlah', y='Pekerjaan', orientation='h',
            color_discrete_sequence=['#8B5CF6'], text='Jumlah')
        fig_j.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_j, "Pekerjaan"), use_container_width=True)

    # ── SES & Penghasilan ─────────────────────────────────────────────
    ses1, ses2 = st.columns(2)
    with ses1:
        ses_c = df['P5'].value_counts().reset_index()
        ses_c.columns = ['SES','Jumlah']
        fig_ses = px.bar(ses_c, x='Jumlah', y='SES', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_ses.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_ses, "Tingkat Pengeluaran (SES)"),
            use_container_width=True)
    with ses2:
        inc_c = df['P6'].value_counts().reset_index()
        inc_c.columns = ['Penghasilan','Jumlah']
        fig_inc = px.bar(inc_c, x='Jumlah', y='Penghasilan', orientation='h',
            color_discrete_sequence=['#10B981'], text='Jumlah')
        fig_inc.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_inc, "Distribusi Penghasilan Rumah Tangga"),
            use_container_width=True)

    # ── Treemap geografis ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🗺️ Peta Geografis</div>",
                unsafe_allow_html=True)
    geo_df = df.groupby(['PROV','KABKOTA','CABANG']).agg(
        Jumlah=('SERIAL','count'),
        NPS=('G1A','mean'),
        Kepuasan=('E1A','mean'),
        Loyalitas=('F1A','mean')
    ).reset_index()

    color_metric = st.selectbox(
        "Warna berdasarkan:",
        ['NPS','Kepuasan','Loyalitas','Jumlah']
    )
    fig_tree = px.treemap(geo_df,
        path=[px.Constant("Nasional"),'PROV','KABKOTA','CABANG'],
        values='Jumlah', color=color_metric,
        color_continuous_scale='RdYlGn',
        hover_data={'NPS':':.2f','Kepuasan':':.2f','Loyalitas':':.2f'})
    fig_tree.update_layout(height=500)
    st.plotly_chart(elite_layout(fig_tree,
        f"Treemap Geografis (Warna = {color_metric})"),
        use_container_width=True)

    st.markdown("---")

    # ── Cross-tab segmentasi interaktif ──────────────────────────────
    st.markdown("<div class='section-header'>🔀 Segmentasi Interaktif</div>",
                unsafe_allow_html=True)

    seg_c1, seg_c2, seg_c3 = st.columns(3)
    with seg_c1:
        seg_by = st.selectbox("Segmentasi by:", [
            'S1 → Gender', 'S2_2 → Usia', 'S4 → Tenure',
            'S7 → Frekuensi Transaksi', 'P3 → Pendidikan',
            'P4 → Pekerjaan', 'P1 → Status Nikah',
            'P5 → SES Pengeluaran'
        ])
    with seg_c2:
        seg_metric = st.selectbox("Metrik:", [
            'G1A → NPS', 'E1A → Kepuasan', 'F1A → Loyalitas',
            'OVR_TELLER_XYZ → Teller', 'OVR_CS_XYZ → CS',
            'OVR_ATM_XYZ → ATM', 'OVR_SEKURITI_XYZ → Sekuriti',
            'OVR_KC_XYZ → Kantor Cabang'
        ])
    with seg_c3:
        seg_chart = st.radio("Tipe chart:", ["Bar","Box"], horizontal=True)

    seg_key  = seg_by.split(' → ')[0].strip()
    metr_key = seg_metric.split(' → ')[0].strip()

    if seg_key in df.columns and metr_key in df.columns:
        if seg_chart == "Bar":
            seg_agg = df.groupby(seg_key)[metr_key].mean().reset_index()
            seg_agg.columns = ['Segmen','Nilai']
            seg_agg = seg_agg.sort_values('Nilai', ascending=True)
            fig_seg = px.bar(seg_agg, x='Nilai', y='Segmen', orientation='h',
                color='Nilai', color_continuous_scale='Blues', text='Nilai')
            fig_seg.update_traces(texttemplate='%{x:.2f}', textposition='outside')
        else:
            fig_seg = px.box(df, x=seg_key, y=metr_key, color=seg_key,
                points='outliers')
            fig_seg.update_layout(xaxis_tickangle=-30)

        st.plotly_chart(
            elite_layout(fig_seg, f"{seg_metric.split('→')[1]} per {seg_by.split('→')[1]}",
                height=420),
            use_container_width=True
        )

    st.markdown("---")

    # ── Frekuensi transaksi vs outcome ────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Frekuensi Transaksi vs Outcome</div>",
                unsafe_allow_html=True)
    freq_df = df.groupby('S7').agg(
        NPS=('G1A','mean'), Kepuasan=('E1A','mean'), Loyalitas=('F1A','mean'),
        Count=('SERIAL','count')
    ).reset_index()
    fig_freq = go.Figure()
    for col_freq, col_color, label_f in [
        ('NPS', COLOR_XYZ, 'NPS'),
        ('Kepuasan', '#10B981', 'Kepuasan'),
        ('Loyalitas', '#F59E0B', 'Loyalitas')
    ]:
        fig_freq.add_trace(go.Bar(
            name=label_f, x=freq_df['S7'], y=freq_df[col_freq],
            marker_color=col_color, text=freq_df[col_freq].round(2),
            textposition='outside'
        ))
    fig_freq.update_layout(barmode='group', xaxis_tickangle=-20, height=380)
    st.plotly_chart(elite_layout(fig_freq,
        "Frekuensi Transaksi vs NPS, Kepuasan, Loyalitas"),
        use_container_width=True)

    # ── Tujuan buka rekening vs loyalitas ────────────────────────────
    st.markdown("<div class='section-header'>🎯 Tujuan Buka Rekening vs Loyalitas</div>",
                unsafe_allow_html=True)
    tujuan_series = df['A2'].dropna().str.split(';').explode().str.strip()
    tujuan_df     = tujuan_series.to_frame('Tujuan').join(df['F1A'], how='left')
    tujuan_agg    = tujuan_df.groupby('Tujuan').agg(
        Loyalitas=('F1A','mean'), Count=('F1A','count')
    ).reset_index().sort_values('Loyalitas')
    tujuan_agg    = tujuan_agg[tujuan_agg['Count'] >= 10]
    fig_tujuan    = px.bar(tujuan_agg, x='Loyalitas', y='Tujuan',
        orientation='h', color='Count', color_continuous_scale='Blues',
        text='Loyalitas', hover_data=['Count'])
    fig_tujuan.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    fig_tujuan.update_xaxes(range=[4.5, 6.3])
    st.plotly_chart(elite_layout(fig_tujuan,
        "Rata-rata Loyalitas per Tujuan Membuka Rekening"),
        use_container_width=True)

# =====================================================================
# TAB 8 — VOICE OF CUSTOMER
# =====================================================================
with tab8:
    # ── KPI VoC ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>💬 Voice of Customer Overview</div>",
                unsafe_allow_html=True)

    nps_v, pp_v, pasv_v, det_v = calc_nps(df['G1A_CAT'])
    v_k1, v_k2, v_k3, v_k4 = st.columns(4)
    v_k1.markdown(fmt_card("NPS Score XYZ", f"{nps_v:.0f}", "blue",
        f"{len(df[df['G1A_CAT']=='Promoter'])} Promoter"), unsafe_allow_html=True)
    v_k2.markdown(fmt_card("Promoter %", f"{pp_v:.0f}%", "green",
        "Pasti merekomendasikan"), unsafe_allow_html=True)
    v_k3.markdown(fmt_card("Passive %", f"{pasv_v:.0f}%", "amber",
        "Netral"), unsafe_allow_html=True)
    v_k4.markdown(fmt_card("Detractor %", f"{det_v:.0f}%", "red",
        "Tidak merekomendasikan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribusi NPS ────────────────────────────────────────────────
    voc1, voc2 = st.columns(2)
    with voc1:
        fig_nh = px.histogram(df, x='G1A', nbins=11,
            color='G1A_CAT', color_discrete_map=NPS_COLORS,
            labels={'G1A': 'Skor NPS XYZ'})
        fig_nh.update_layout(bargap=0.1)
        st.plotly_chart(elite_layout(fig_nh, "Distribusi Skor NPS XYZ"),
            use_container_width=True)
    with voc2:
        if len(df_has_komp) > 0:
            fig_nhk = px.histogram(df_has_komp, x='G1C', nbins=11,
                color='G1C_CAT', color_discrete_map=NPS_COLORS,
                labels={'G1C': f'Skor NPS {target_komp}'})
            fig_nhk.update_layout(bargap=0.1)
            st.plotly_chart(elite_layout(fig_nhk,
                f"Distribusi Skor NPS {target_komp}"),
                use_container_width=True)

    st.markdown("---")

    # ── Filter & Wordcloud ────────────────────────────────────────────
    st.markdown("<div class='section-header'>☁️ Analisis Teks & Wordcloud</div>",
                unsafe_allow_html=True)

    wc_f1, wc_f2, wc_f3 = st.columns(3)
    with wc_f1:
        filter_cat = st.selectbox(
            "Filter Kategori NPS:",
            ["Semua", "Promoter", "Passive", "Detractor"]
        )
    with wc_f2:
        wc_source = st.selectbox(
            "Sumber teks:",
            ["G1B (Alasan NPS XYZ)", "G1D (Alasan NPS Kompetitor)",
             "E1AA (Alasan Kepuasan XYZ)", "E1BB (Alasan Kepuasan Komp)"]
        )
    with wc_f3:
        min_word_len = st.slider("Min. panjang kata:", 3, 7, 4)

    col_wc_map = {
        "G1B (Alasan NPS XYZ)":       ("G1B",  "G1A_CAT"),
        "G1D (Alasan NPS Kompetitor)": ("G1D",  "G1C_CAT"),
        "E1AA (Alasan Kepuasan XYZ)":  ("E1AA", "G1A_CAT"),
        "E1BB (Alasan Kepuasan Komp)": ("E1BB", "G1C_CAT"),
    }
    text_col, cat_col = col_wc_map[wc_source]
    df_wc_src = df_has_komp if "Kompetitor" in wc_source else df
    df_wc = df_wc_src if filter_cat == "Semua" \
            else df_wc_src[df_wc_src[cat_col] == filter_cat] \
            if cat_col in df_wc_src.columns else df_wc_src

    stopwords_id = {
        'yang','untuk','dengan','pada','dari','sebagai','tidak','karena',
        'sangat','lebih','sudah','saya','bank','bisa','dan','di','ke','ini',
        'itu','ada','juga','net','subnet','positive','negative','comments',
        'dalam','oleh','akan','telah','dapat','kami','anda','nya','atau',
        'jadi','baru','lagi','saat','pernah','masih','serta','yaitu','namun',
        'jika','agar','bagi','atas','antara','setiap','para','mereka','kita',
        'xyz','bank','nasabah','layanan','cabang','rekening'
    }

    def make_wordcloud(series, cmap='Blues', max_w=80):
        text = " ".join(series.dropna().astype(str).tolist()).lower()
        words = re.findall(rf'\b[a-zA-Z]{{{min_word_len},}}\b', text)
        words_c = [w for w in words if w not in stopwords_id]
        if len(words_c) < 5:
            return None, None
        wc = WordCloud(width=700, height=350, background_color='#F8FAFC',
            colormap=cmap, max_words=max_w, stopwords=stopwords_id).generate(
            " ".join(words_c))
        top_words = Counter(words_c).most_common(10)
        return wc, top_words

    wc_im, top_w = make_wordcloud(
        df_wc[text_col] if text_col in df_wc.columns else pd.Series(dtype=str)
    )

    wc_col1, wc_col2 = st.columns([1.6, 1])
    with wc_col1:
        if wc_im:
            fig_wc, ax = plt.subplots(figsize=(7, 3.5))
            ax.imshow(wc_im, interpolation='bilinear')
            ax.axis('off')
            fig_wc.patch.set_facecolor('#F8FAFC')
            fig_wc.tight_layout(pad=0)
            st.pyplot(fig_wc)
            plt.close()
        else:
            st.info("Tidak ada teks yang cukup untuk wordcloud.")

    with wc_col2:
        if top_w:
            st.markdown("**🔑 Top 10 Kata Kunci:**")
            kw_df = pd.DataFrame(top_w, columns=['Kata','Frekuensi'])
            fig_kw = px.bar(kw_df.sort_values('Frekuensi'),
                x='Frekuensi', y='Kata', orientation='h',
                color='Frekuensi', color_continuous_scale='Blues',
                text='Frekuensi')
            fig_kw.update_traces(textposition='outside')
            fig_kw.update_layout(height=350, showlegend=False)
            st.plotly_chart(elite_layout(fig_kw), use_container_width=True)

    st.markdown("---")

    # ── Top 3 Promoter / Detractor ────────────────────────────────────
    st.markdown("<div class='section-header'>💡 Top Alasan Promoter & Detractor</div>",
                unsafe_allow_html=True)
    pa1, pa2 = st.columns(2)

    def top_themes(df_src, text_col_t, cat, n=8):
        text = " ".join(
            df_src[df_src['G1A_CAT'] == cat][text_col_t].dropna().astype(str).tolist()
        ).lower()
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        words_c = [w for w in words if w not in stopwords_id]
        return Counter(words_c).most_common(n)

    with pa1:
        st.markdown("**🌟 Promoter — Top Tema**")
        prom_words = top_themes(df, 'G1B', 'Promoter')
        if prom_words:
            prom_df = pd.DataFrame(prom_words, columns=['Tema','Count'])
            fig_p = px.bar(prom_df, x='Count', y='Tema', orientation='h',
                color_discrete_sequence=['#10B981'], text='Count')
            fig_p.update_traces(textposition='outside')
            fig_p.update_layout(height=320)
            st.plotly_chart(elite_layout(fig_p), use_container_width=True)

    with pa2:
        st.markdown("**⚠️ Detractor — Top Tema**")
        det_words = top_themes(df, 'G1B', 'Detractor')
        if det_words:
            det_df = pd.DataFrame(det_words, columns=['Tema','Count'])
            fig_d = px.bar(det_df, x='Count', y='Tema', orientation='h',
                color_discrete_sequence=[COLOR_KOMP], text='Count')
            fig_d.update_traces(textposition='outside')
            fig_d.update_layout(height=320)
            st.plotly_chart(elite_layout(fig_d), use_container_width=True)

    # Insight dari tema
    if top_w:
        top_kata = top_w[0][0] if top_w else "N/A"
        insight_box(
            f"Kata yang paling sering muncul pada {filter_cat}: **'{top_kata}'**. "
            f"Ini mencerminkan tema utama yang mendorong persepsi nasabah terhadap Bank XYZ."
        )

    st.markdown("---")

    # ── Verbatim Explorer ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Verbatim Explorer</div>",
                unsafe_allow_html=True)

    ve1, ve2, ve3, ve4 = st.columns(4)
    with ve1:
        v_nps_filter = st.multiselect(
            "Kategori NPS:", ['Promoter','Passive','Detractor'],
            default=['Promoter','Passive','Detractor']
        )
    with ve2:
        v_prov_filter = st.multiselect(
            "Provinsi:", sorted(df['PROV'].dropna().unique().tolist())
        )
    with ve3:
        v_search = st.text_input("🔎 Cari kata kunci:")
    with ve4:
        v_min_nps = st.slider("Min. NPS Score:", 0, 10, 0)

    verb_df = df[df['G1A_CAT'].isin(v_nps_filter)] if v_nps_filter else df
    if v_prov_filter:
        verb_df = verb_df[verb_df['PROV'].isin(v_prov_filter)]
    verb_df = verb_df[verb_df['G1A'] >= v_min_nps]
    if v_search:
        verb_df = verb_df[
            verb_df['G1B'].fillna('').str.contains(v_search, case=False) |
            verb_df['E1AA'].fillna('').str.contains(v_search, case=False)
        ]

    st.info(f"Menampilkan {len(verb_df):,} responden")

    display_cols = {
        'CABANG': 'Cabang', 'PROV': 'Provinsi',
        'S1': 'Gender', 'S2_2': 'Usia', 'S4': 'Tenure',
        'G1A': 'Skor NPS', 'G1A_CAT': 'Kategori NPS',
        'E1A': 'Kepuasan', 'G1B': 'Alasan NPS', 'E1AA': 'Alasan Kepuasan'
    }
    existing_cols = {k: v for k, v in display_cols.items() if k in verb_df.columns}
    st.dataframe(
        verb_df[list(existing_cols.keys())]
            .rename(columns=existing_cols)
            .sort_values('Skor NPS'),
        use_container_width=True, height=380, hide_index=True
    )

st.markdown("---")
st.caption("🚀 Bank XYZ — Executive CX Intelligence Dashboard v2.0 | Powered by Streamlit & Plotly")
