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
    border: 1px solid #E2E8F0; padding: 20px; border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 20px -5px rgba(37,99,235,0.15);
    border-color: #BFDBFE;
}
.metric-title {
    color: #64748B; font-size: 12px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px;
}
.metric-value { color: #0F172A; font-size: 34px; font-weight: 900; line-height: 1; }
.metric-value.blue  { color: #2563EB; }
.metric-value.green { color: #10B981; }
.metric-value.red   { color: #E11D48; }
.metric-value.amber { color: #F59E0B; }
.metric-sub { color: #94A3B8; font-size: 12px; margin-top: 6px; }
.section-header {
    font-size: 16px; font-weight: 800; color: #0F172A;
    border-left: 4px solid #2563EB; padding-left: 12px;
    margin: 20px 0 12px 0;
}
.stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: #ffffff; border: 1px solid #E2E8F0;
    border-radius: 8px 8px 0 0; padding: 10px 20px;
    color: #64748B !important; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #2563EB !important; color: #ffffff !important;
    border-color: #2563EB;
}
h1, h2, h3, h4 { color: #0F172A !important; font-weight: 800 !important; }
p { color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

# Konstanta warna
NPS_COLORS   = {'Promoter': '#10B981', 'Passive': '#F59E0B', 'Detractor': '#EF4444'}
COLOR_XYZ    = '#2563EB'
COLOR_KOMP   = '#E11D48'
COLOR_IMPRT  = '#94A3B8'

def fmt_card(title, value, color="", sub=""):
    return f"""
    <div class='metric-card'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value {color}'>{value}</div>
        {'<div class="metric-sub">' + sub + '</div>' if sub else ''}
    </div>"""

def elite_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color='#0F172A')),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=55, b=20, l=160, r=20),
        font=dict(color='#334155', size=12),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#FFFFFF", font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(
        showgrid=True, gridcolor='#F1F5F9', zeroline=False,
        automargin=True, tickfont=dict(color='#475569')
    )
    fig.update_yaxes(
        showgrid=True, gridcolor='#F1F5F9', zeroline=False,
        automargin=True, tickfont=dict(color='#475569')
    )
    return fig

# =====================================================================
# 2. LOAD DATA
# =====================================================================
@st.cache_data
def load_data():
    df       = pd.read_csv('data/df_clean.csv', low_memory=False)
    col_map  = pd.read_csv('data/col_mapping.csv').set_index('kode')['nama_panjang'].to_dict()
    summary  = pd.read_csv('data/summary_scores.csv')
    return df, col_map, summary

try:
    df_raw, col_map, summary_df = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# =====================================================================
# 3. SIDEBAR — FILTER GLOBAL
# =====================================================================
st.sidebar.markdown(
    "<h3 style='color:#2563EB !important; font-weight:800;'>🎛️ Filter Analitik</h3>",
    unsafe_allow_html=True
)

# Filter kompetitor benchmark
komp_list   = sorted(df_raw['KOMP'].dropna().unique().tolist())
target_komp = st.sidebar.selectbox(
    "🎯 Benchmark Kompetitor",
    ["Semua Kompetitor (Rata-rata)"] + komp_list
)

st.sidebar.markdown("---")
st.sidebar.caption("📌 Filter Segmen Nasabah")

# Cascading filter: Provinsi → Kab/Kota → Cabang
prov_opts = sorted(df_raw['PROV'].dropna().unique().tolist())
sel_prov  = st.sidebar.multiselect("📍 Provinsi", prov_opts)

kota_pool = df_raw[df_raw['PROV'].isin(sel_prov)] if sel_prov else df_raw
kota_opts = sorted(kota_pool['KABKOTA'].dropna().unique().tolist())
sel_kota  = st.sidebar.multiselect("🏙️ Kab / Kota", kota_opts)

cab_pool  = kota_pool[kota_pool['KABKOTA'].isin(sel_kota)] if sel_kota else kota_pool
cab_opts  = sorted(cab_pool['CABANG'].dropna().unique().tolist())
sel_cab   = st.sidebar.multiselect("🏢 Kantor Cabang", cab_opts)

st.sidebar.markdown("---")
st.sidebar.caption("👤 Profil Responden")

sel_gender = st.sidebar.multiselect(
    "🚻 Jenis Kelamin",
    sorted(df_raw['S1'].dropna().unique().tolist())
)
sel_usia = st.sidebar.multiselect(
    "🎂 Rentang Usia",
    sorted(df_raw['S2_2'].dropna().unique().tolist())
)
sel_tenure = st.sidebar.multiselect(
    "⏳ Lama Menjadi Nasabah",
    sorted(df_raw['S4'].dropna().unique().tolist())
)
sel_frekuensi = st.sidebar.multiselect(
    "🔄 Frekuensi Transaksi",
    sorted(df_raw['S7'].dropna().unique().tolist())
)
sel_panel = st.sidebar.multiselect(
    "🪪 Panel",
    sorted(df_raw['PANEL'].dropna().unique().tolist())
)

# ── Apply filter ──────────────────────────────────────────────────────
df = df_raw.copy()
if sel_prov:      df = df[df['PROV'].isin(sel_prov)]
if sel_kota:      df = df[df['KABKOTA'].isin(sel_kota)]
if sel_cab:       df = df[df['CABANG'].isin(sel_cab)]
if sel_gender:    df = df[df['S1'].isin(sel_gender)]
if sel_usia:      df = df[df['S2_2'].isin(sel_usia)]
if sel_tenure:    df = df[df['S4'].isin(sel_tenure)]
if sel_frekuensi: df = df[df['S7'].isin(sel_frekuensi)]
if sel_panel:     df = df[df['PANEL'].isin(sel_panel)]

# Subset kompetitor
df_komp = df.copy() if target_komp == "Semua Kompetitor (Rata-rata)" \
          else df[df['KOMP'] == target_komp]
df_has_komp = df_komp[df_komp['KOMP'].notna()]

st.sidebar.markdown("---")
st.sidebar.success(f"📊 Total Responden: **{len(df):,}**")
st.sidebar.info(f"🏦 Responden dgn Kompetitor: **{len(df_has_komp):,}**")

# ── Header utama ──────────────────────────────────────────────────────
st.markdown("""
<h2 style='text-align:center; font-weight:900; letter-spacing:-1px;
color:#0F172A !important; margin-bottom:20px;'>
🏦 BANK XYZ — EXECUTIVE CX INTELLIGENCE DASHBOARD
</h2>""", unsafe_allow_html=True)

if df.empty:
    st.warning("⚠️ Data kosong. Sesuaikan filter Anda.")
    st.stop()

# =====================================================================
# 4. TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌟 Executive Summary",
    "🏢 Kinerja Layanan Cabang",
    "🏆 Brand & Kompetitor",
    "🎯 Touchpoint & IPA",
    "💡 Emosi, Loyalitas & Digital",
    "👥 Profil & Segmentasi",
    "💬 Voice of Customer"
])

# =====================================================================
# TAB 1 — EXECUTIVE SUMMARY
# =====================================================================
with tab1:
    # ── KPI Cards ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📌 Indikator Utama</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    nps_xyz  = df['G1A'].mean()
    nps_kom  = df_has_komp['G1C'].mean() if len(df_has_komp) > 0 else np.nan
    sat_xyz  = df['E1A'].mean()
    sat_kom  = df_has_komp['E1B'].mean() if len(df_has_komp) > 0 else np.nan
    loy_xyz  = df['F1A'].mean()
    loy_kom  = df_has_komp['F1B'].mean() if len(df_has_komp) > 0 else np.nan

    # Hitung NPS score sesungguhnya (% Promoter - % Detractor)
    total_nps   = df['G1A_CAT'].notna().sum()
    pct_prom    = (df['G1A_CAT'] == 'Promoter').sum() / total_nps * 100
    pct_detr    = (df['G1A_CAT'] == 'Detractor').sum() / total_nps * 100
    nps_score   = pct_prom - pct_detr

    total_kom   = df_has_komp['G1C_CAT'].notna().sum()
    pct_prom_k  = (df_has_komp['G1C_CAT'] == 'Promoter').sum() / total_kom * 100 if total_kom > 0 else 0
    pct_detr_k  = (df_has_komp['G1C_CAT'] == 'Detractor').sum() / total_kom * 100 if total_kom > 0 else 0
    nps_score_k = pct_prom_k - pct_detr_k

    k1.markdown(fmt_card("NPS Score XYZ", f"{nps_score:.1f}", "blue",
        f"Promoter {pct_prom:.0f}% | Detractor {pct_detr:.0f}%"), unsafe_allow_html=True)
    k2.markdown(fmt_card("NPS Score Kompetitor", f"{nps_score_k:.1f}", "red",
        f"Promoter {pct_prom_k:.0f}% | Detractor {pct_detr_k:.0f}%"), unsafe_allow_html=True)
    k3.markdown(fmt_card("Kepuasan XYZ", f"{sat_xyz:.2f}", "blue", "Skala 1–6"),
        unsafe_allow_html=True)
    k4.markdown(fmt_card("Kepuasan Kompetitor",
        f"{sat_kom:.2f}" if not np.isnan(sat_kom) else "N/A", "red", "Skala 1–6"),
        unsafe_allow_html=True)
    k5.markdown(fmt_card("Loyalitas XYZ", f"{loy_xyz:.2f}", "green", "Skala 1–6"),
        unsafe_allow_html=True)
    k6.markdown(fmt_card("Loyalitas Kompetitor",
        f"{loy_kom:.2f}" if not np.isnan(loy_kom) else "N/A", "red", "Skala 1–6"),
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Baris 2: NPS Donut + Scorecard Dimensi ───────────────────────
    r2c1, r2c2 = st.columns([1, 2])

    with r2c1:
        st.markdown("<div class='section-header'>Komposisi NPS XYZ</div>", unsafe_allow_html=True)
        nps_comp = df['G1A_CAT'].value_counts().reset_index()
        nps_comp.columns = ['Kategori', 'Jumlah']
        fig_donut = px.pie(
            nps_comp, values='Jumlah', names='Kategori', hole=0.58,
            color='Kategori', color_discrete_map=NPS_COLORS
        )
        fig_donut.update_traces(
            textposition='outside', textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        st.plotly_chart(elite_layout(fig_donut), use_container_width=True)

    with r2c2:
        st.markdown("<div class='section-header'>Scorecard Dimensi Layanan: XYZ vs Kompetitor</div>",
            unsafe_allow_html=True)
        # Buat tabel scorecard dari OVR_ columns
        scorecard_data = [
            ("Kantor Cabang",    "OVR_KC_XYZ",          "OVR_KC_KOM"),
            ("Operasional",      "OVR_KC_OPERASIONAL_XYZ","OVR_KC_OPERASIONAL_KOM"),
            ("Parkir",           "OVR_KC_PARKIR_XYZ",   "OVR_KC_PARKIR_KOM"),
            ("Banking Hall",     "OVR_KC_BANKINGHALL_XYZ","OVR_KC_BANKINGHALL_KOM"),
            ("Toilet",           "OVR_KC_TOILET_XYZ",   "OVR_KC_TOILET_KOM"),
            ("Sekuriti",         "OVR_SEKURITI_XYZ",    "OVR_SEKURITI_KOM"),
            ("Teller",           "OVR_TELLER_XYZ",      "OVR_TELLER_KOM"),
            ("Customer Service", "OVR_CS_XYZ",          "OVR_CS_KOM"),
            ("Customer Advisor", "OVR_CA_XYZ",          None),
            ("Sarana Elektronik","OVR_SARANA_XYZ",      None),
            ("ATM",              "OVR_ATM_XYZ",         "OVR_ATM_KOM"),
        ]
        rows = []
        for label, col_xyz, col_kom in scorecard_data:
            xyz_val = df[col_xyz].mean() if col_xyz in df.columns else np.nan
            kom_val = df_has_komp[col_kom].mean() \
                      if col_kom and col_kom in df_has_komp.columns and len(df_has_komp) > 0 \
                      else np.nan
            gap = xyz_val - kom_val if not np.isnan(xyz_val) and not np.isnan(kom_val) else np.nan
            rows.append({
                "Dimensi": label,
                "XYZ": round(xyz_val, 2),
                "Kompetitor": round(kom_val, 2) if not np.isnan(kom_val) else "—",
                "Gap (XYZ−Komp)": round(gap, 2) if not np.isnan(gap) else "—"
            })
        sc_df = pd.DataFrame(rows)

        fig_sc = go.Figure(data=[
            go.Bar(name='Bank XYZ', x=sc_df['Dimensi'], y=sc_df['XYZ'],
                   marker_color=COLOR_XYZ, text=sc_df['XYZ'],
                   texttemplate='%{text:.2f}', textposition='outside'),
            go.Bar(name='Kompetitor',
                   x=sc_df['Dimensi'],
                   y=pd.to_numeric(sc_df['Kompetitor'], errors='coerce'),
                   marker_color=COLOR_KOMP,
                   text=sc_df['Kompetitor'],
                   texttemplate='%{text}', textposition='outside'),
        ])
        fig_sc.update_layout(barmode='group', yaxis_range=[4, 6.3])
        st.plotly_chart(elite_layout(fig_sc, "Skor Rata-rata per Dimensi (Skala 1–6)"),
            use_container_width=True)

    # ── Baris 3: Top/Bottom Cabang + Korelasi ────────────────────────
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown("<div class='section-header'>Top & Bottom 5 Cabang (by Kepuasan Overall)</div>",
            unsafe_allow_html=True)
        # Gabungkan semua OVR_ XYZ per cabang
        ovr_xyz_cols = [c for c in df.columns if c.startswith("OVR_") and c.endswith("_XYZ")]
        cab_score = df.groupby('CABANG')[ovr_xyz_cols].mean().mean(axis=1).reset_index()
        cab_score.columns = ['CABANG', 'Skor']
        top5    = cab_score.nlargest(5, 'Skor')
        bottom5 = cab_score.nsmallest(5, 'Skor')
        tb_df   = pd.concat([
            top5.assign(Status='🌟 Top 5'),
            bottom5.assign(Status='⚠️ Bottom 5')
        ])
        fig_tb = px.bar(
            tb_df, x='Skor', y='CABANG', color='Status', orientation='h',
            color_discrete_map={'🌟 Top 5': COLOR_XYZ, '⚠️ Bottom 5': COLOR_KOMP},
            text='Skor'
        )
        fig_tb.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_tb.update_xaxes(range=[4.5, 6.3])
        st.plotly_chart(elite_layout(fig_tb), use_container_width=True)

    with r3c2:
        st.markdown("<div class='section-header'>Matriks Korelasi Indikator Utama</div>",
            unsafe_allow_html=True)
        corr_cols = {
            'NPS': 'G1A', 'Kepuasan': 'E1A', 'Loyalitas': 'F1A',
            'Teller': 'OVR_TELLER_XYZ', 'CS': 'OVR_CS_XYZ',
            'ATM': 'OVR_ATM_XYZ', 'Sekuriti': 'OVR_SEKURITI_XYZ'
        }
        corr_df = df[[v for v in corr_cols.values() if v in df.columns]].copy()
        corr_df.columns = [k for k, v in corr_cols.items() if v in df.columns]
        corr_matrix = corr_df.corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale='Blues', aspect='auto'
        )
        st.plotly_chart(elite_layout(fig_corr, "Korelasi Antar Indikator Utama"),
            use_container_width=True)

# =====================================================================
# TAB 2 — KINERJA LAYANAN CABANG
# =====================================================================
with tab2:
    st.markdown("<div class='section-header'>🏢 Heatmap Kinerja Cabang per Dimensi</div>",
        unsafe_allow_html=True)

    ovr_xyz_cols = [c for c in df.columns if c.startswith("OVR_") and c.endswith("_XYZ")]
    heatmap_df   = df.groupby('CABANG')[ovr_xyz_cols].mean().round(2)
    heatmap_df.columns = [c.replace("OVR_","").replace("_XYZ","") for c in heatmap_df.columns]

    fig_heat = px.imshow(
        heatmap_df, text_auto=".2f",
        color_continuous_scale='Blues',
        aspect='auto', zmin=4, zmax=6
    )
    fig_heat.update_layout(height=700)
    st.plotly_chart(elite_layout(fig_heat, "Heatmap Skor OVR per Cabang (Skala 1–6)"),
        use_container_width=True)

    st.markdown("---")

    # ── Drill-down item level ─────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Drill-Down Item Level per Dimensi</div>",
        unsafe_allow_html=True)

    # Mapping dimensi → kolom importance & satisfaction XYZ & kompetitor
    DIMENSI_MAP = {
        "Kantor Cabang (Fasilitas)": {
            "imp":  [f"T_KC1_{i}" for i in range(1, 36)],
            "xyz":  [c for c in df.columns if c.startswith("T_KC2_") and
                     not c.endswith(("_107","_110","_113","_116","_108","_111","_114","_117")) and
                     int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_KC2_") and
                     not c.endswith(("_107","_110","_113","_116","_108","_111","_114","_117")) and
                     int(c.split("_")[-1]) % 3 == 0],
        },
        "Sekuriti": {
            "imp":  [f"T_SC1_{i}" for i in range(1, 16)],
            "xyz":  [c for c in df.columns if c.startswith("T_SC2_") and
                     c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_SC2_") and
                     c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Teller": {
            "imp":  [f"T_TL2_{i}" for i in range(1, 20)],
            "xyz":  [c for c in df.columns if c.startswith("T_TL3_") and
                     c != "T_TL3_59" and c != "T_TL3_60" and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_TL3_") and
                     c != "T_TL3_59" and c != "T_TL3_60" and int(c.split("_")[-1]) % 3 == 0],
        },
        "Customer Service": {
            "imp":  [f"T_CS2_{i}" for i in range(1, 24)],
            "xyz":  [c for c in df.columns if c.startswith("T_CS3_") and
                     c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_CS3_") and
                     c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1]) % 3 == 0],
        },
        "Customer Advisor": {
            "imp":  [f"T_CA1_{i}" for i in range(1, 20)],
            "xyz":  [c for c in df.columns if c.startswith("T_CA2_") and c != "T_CA2_20"],
            "komp": [],
        },
        "ATM": {
            "imp":  [f"T_AT2_{i}" for i in range(1, 19)],
            "xyz":  [c for c in df.columns if c.startswith("T_AT3_") and
                     c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1]) % 3 == 2],
            "komp": [c for c in df.columns if c.startswith("T_AT3_") and
                     c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1]) % 3 == 0],
        },
    }

    sel_dim = st.selectbox("Pilih Dimensi:", list(DIMENSI_MAP.keys()))
    dim_info = DIMENSI_MAP[sel_dim]

    imp_cols  = [c for c in dim_info["imp"]  if c in df.columns]
    xyz_cols  = [c for c in dim_info["xyz"]  if c in df.columns]
    komp_cols = [c for c in dim_info["komp"] if c in df_has_komp.columns]

    # Buat label dari col_map, potong supaya tidak terlalu panjang
    def short_label(col, max_len=45):
        name = col_map.get(col, col)
        # Hapus suffix " - XYZ", " - kompetitor"
        name = re.sub(r'\s*-\s*(XYZ|kompetitor)\s*$', '', name, flags=re.IGNORECASE)
        return name[:max_len] + "..." if len(name) > max_len else name

    min_len = min(len(imp_cols), len(xyz_cols))
    if min_len > 0:
        labels   = [short_label(c) for c in imp_cols[:min_len]]
        imp_vals = [df[c].mean() for c in imp_cols[:min_len]]
        xyz_vals = [df[c].mean() for c in xyz_cols[:min_len]]
        kom_vals = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                    for c in komp_cols[:min_len]] if komp_cols else []

        dd1, dd2 = st.columns([2, 1])
        with dd1:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Bar(
                name='Importance', x=imp_vals, y=labels,
                orientation='h', marker_color=COLOR_IMPRT,
                opacity=0.7
            ))
            fig_dd.add_trace(go.Bar(
                name='Satisfaction XYZ', x=xyz_vals, y=labels,
                orientation='h', marker_color=COLOR_XYZ
            ))
            if kom_vals:
                fig_dd.add_trace(go.Bar(
                    name='Satisfaction Kompetitor', x=kom_vals, y=labels,
                    orientation='h', marker_color=COLOR_KOMP, opacity=0.8
                ))
            fig_dd.update_layout(barmode='group', xaxis_range=[3, 6.3], height=500)
            st.plotly_chart(
                elite_layout(fig_dd, f"Importance vs Satisfaction — {sel_dim}"),
                use_container_width=True
            )

        with dd2:
            # Gap = satisfaction XYZ - importance (negatif = area perbaikan)
            gap_data = pd.DataFrame({
                'Item': labels,
                'Gap': [x - i for x, i in zip(xyz_vals, imp_vals)]
            }).sort_values('Gap')
            gap_data['Warna'] = np.where(gap_data['Gap'] < 0, COLOR_KOMP, COLOR_XYZ)
            fig_gap = px.bar(
                gap_data, x='Gap', y='Item', orientation='h',
                text='Gap', title="Gap (Satisfaction − Importance)"
            )
            fig_gap.update_traces(
                marker_color=gap_data['Warna'],
                texttemplate='%{text:.2f}', textposition='outside'
            )
            fig_gap.update_layout(height=500)
            st.plotly_chart(elite_layout(fig_gap), use_container_width=True)

    st.markdown("---")

    # ── Waktu tunggu ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>⏱️ Waktu Tunggu: Aktual vs Toleransi</div>",
        unsafe_allow_html=True)
    wt1, wt2 = st.columns(2)

    with wt1:
        df_tl = df[df['PANEL'] == 'Teller'].copy()
        if len(df_tl) > 0:
            wt_data = pd.DataFrame({
                'Metrik': ['Aktual (menit)', 'Toleransi (menit)'],
                'Teller': [df_tl['TL5'].mean(), df_tl['TL6'].mean()],
            })
            fig_wt = px.bar(
                wt_data, x='Metrik', y='Teller',
                color='Metrik', text='Teller',
                color_discrete_sequence=[COLOR_KOMP, COLOR_XYZ]
            )
            fig_wt.update_traces(texttemplate='%{y:.1f} mnt', textposition='outside')
            fig_wt.update_yaxes(range=[0, max(df_tl['TL6'].mean() * 1.4, 20)])
            st.plotly_chart(elite_layout(fig_wt, "⏳ Teller: Waktu Tunggu"),
                use_container_width=True)

    with wt2:
        df_cs = df[df['PANEL'] == 'CS'].copy()
        if len(df_cs) > 0:
            wt_data2 = pd.DataFrame({
                'Metrik': ['Aktual (menit)', 'Toleransi (menit)'],
                'CS': [df_cs['CS5'].mean(), df_cs['CS6'].mean()],
            })
            fig_wt2 = px.bar(
                wt_data2, x='Metrik', y='CS',
                color='Metrik', text='CS',
                color_discrete_sequence=[COLOR_KOMP, COLOR_XYZ]
            )
            fig_wt2.update_traces(texttemplate='%{y:.1f} mnt', textposition='outside')
            fig_wt2.update_yaxes(range=[0, max(df_cs['CS6'].mean() * 1.4, 20)])
            st.plotly_chart(elite_layout(fig_wt2, "⏳ CS: Waktu Tunggu"),
                use_container_width=True)

# =====================================================================
# TAB 3 — BRAND & KOMPETITOR
# =====================================================================
with tab3:
    # ── 24 atribut brand ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏆 Pemetaan 24 Atribut Brand</div>",
        unsafe_allow_html=True)

    # Kolom importance (T_C1A_1 s/d T_C1A_24)
    brand_imp_cols  = [f"T_C1A_{i}" for i in range(1, 25) if f"T_C1A_{i}" in df.columns]
    # Kolom satisfaction XYZ (nomor genap di T_C1B_)
    brand_xyz_cols  = [c for c in df.columns if c.startswith("T_C1B_") and
                       int(c.split("_")[-1]) % 3 == 2]
    # Kolom satisfaction kompetitor (nomor ganjil, bukan 2 & 3)
    brand_kom_cols  = [c for c in df.columns if c.startswith("T_C1B_") and
                       int(c.split("_")[-1]) % 3 == 0]

    brand_labels = [short_label(c) for c in brand_imp_cols]
    brand_imp    = [df[c].mean() for c in brand_imp_cols]
    brand_xyz    = [df[c].mean() for c in brand_xyz_cols[:len(brand_imp_cols)]]
    brand_kom    = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                    for c in brand_kom_cols[:len(brand_imp_cols)]]

    b1, b2 = st.columns(2)
    with b1:
        # Radar chart XYZ vs Kompetitor
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=brand_xyz, theta=brand_labels, fill='toself',
            name='Bank XYZ', line_color=COLOR_XYZ
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=brand_kom, theta=brand_labels, fill='toself',
            name=target_komp, line_color=COLOR_KOMP, opacity=0.7
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[3, 6], gridcolor='#E2E8F0')
            ),
            legend=dict(orientation="h", y=-0.15)
        )
        st.plotly_chart(
            elite_layout(fig_radar, f"Radar Brand: XYZ vs {target_komp}"),
            use_container_width=True
        )

    with b2:
        # IPA Matrix: importance (y) vs satisfaction XYZ (x)
        brand_df = pd.DataFrame({
            'Label':      brand_labels,
            'Importance': brand_imp,
            'Satisfaction XYZ': brand_xyz,
        })
        mean_imp = np.nanmean(brand_imp)
        mean_sat = np.nanmean(brand_xyz)

        fig_ipa = px.scatter(
            brand_df, x='Satisfaction XYZ', y='Importance',
            text='Label', color_discrete_sequence=[COLOR_XYZ]
        )
        fig_ipa.update_traces(
            marker=dict(size=12, line=dict(width=1.5, color='white')),
            textposition='top center'
        )
        fig_ipa.add_vline(x=mean_sat, line_dash="dash", line_color="#94A3B8")
        fig_ipa.add_hline(y=mean_imp, line_dash="dash", line_color="#94A3B8")
        fig_ipa.add_annotation(x=brand_df['Satisfaction XYZ'].min(), y=brand_df['Importance'].max(),
            text="⚠️ PERBAIKI SEGERA", showarrow=False,
            font=dict(color=COLOR_KOMP, size=10), bgcolor="rgba(255,255,255,0.8)")
        fig_ipa.add_annotation(x=brand_df['Satisfaction XYZ'].max(), y=brand_df['Importance'].max(),
            text="🌟 PERTAHANKAN", showarrow=False,
            font=dict(color="#10B981", size=10), bgcolor="rgba(255,255,255,0.8)")
        fig_ipa.add_annotation(x=brand_df['Satisfaction XYZ'].min(), y=brand_df['Importance'].min(),
            text="💤 PRIORITAS RENDAH", showarrow=False,
            font=dict(color="#94A3B8", size=10), bgcolor="rgba(255,255,255,0.8)")
        fig_ipa.add_annotation(x=brand_df['Satisfaction XYZ'].max(), y=brand_df['Importance'].min(),
            text="✅ BERLEBIHAN", showarrow=False,
            font=dict(color="#F59E0B", size=10), bgcolor="rgba(255,255,255,0.8)")
        st.plotly_chart(
            elite_layout(fig_ipa, "IPA Matrix — Atribut Brand XYZ"),
            use_container_width=True
        )

    # ── Gap bar chart brand ───────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Gap Kompetitif per Atribut Brand</div>",
        unsafe_allow_html=True)
    gap_brand = pd.DataFrame({
        'Atribut': brand_labels,
        'Gap': [x - k for x, k in zip(brand_xyz, brand_kom)]
    }).sort_values('Gap')
    gap_brand['Warna'] = np.where(gap_brand['Gap'] < 0, COLOR_KOMP, COLOR_XYZ)

    fig_gap_brand = px.bar(
        gap_brand, x='Gap', y='Atribut', orientation='h',
        text='Gap'
    )
    fig_gap_brand.update_traces(
        marker_color=gap_brand['Warna'],
        texttemplate='%{text:.2f}', textposition='outside'
    )
    st.plotly_chart(
        elite_layout(fig_gap_brand, f"Gap Brand XYZ vs {target_komp} (positif = XYZ unggul)"),
        use_container_width=True
    )

    # ── Share of wallet ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏦 Share of Wallet — Bank yang Digunakan Bersamaan</div>",
        unsafe_allow_html=True)
    sw1, sw2, sw3 = st.columns(3)

    with sw1:
        # Bank aktif selain XYZ
        banks_other = df['A1AX'].dropna().str.split(';').explode().str.strip()
        banks_count = banks_other[banks_other != ''].value_counts().head(8).reset_index()
        banks_count.columns = ['Bank', 'Jumlah']
        fig_sw = px.bar(banks_count, x='Jumlah', y='Bank', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_sw.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_sw, "Bank Lain yang Aktif Digunakan"),
            use_container_width=True)

    with sw2:
        # Bank utama simpan dana
        simpan = df['A1B'].value_counts().reset_index()
        simpan.columns = ['Bank', 'Jumlah']
        fig_simpan = px.pie(simpan, values='Jumlah', names='Bank', hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(elite_layout(fig_simpan, "Bank Utama Simpan Dana"),
            use_container_width=True)

    with sw3:
        # Bank utama transaksi
        transaksi = df['A1C'].value_counts().reset_index()
        transaksi.columns = ['Bank', 'Jumlah']
        fig_trans = px.pie(transaksi, values='Jumlah', names='Bank', hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(elite_layout(fig_trans, "Bank Utama Bertransaksi"),
            use_container_width=True)

# =====================================================================
# TAB 4 — TOUCHPOINT & IPA
# =====================================================================
with tab4:
    st.markdown("<div class='section-header'>🎯 IPA Matrix per Touchpoint Layanan</div>",
        unsafe_allow_html=True)

    sel_tp = st.selectbox(
        "Pilih Touchpoint:",
        ["Sekuriti", "Teller", "Customer Service", "Customer Advisor", "ATM",
         "Kantor Cabang (Fasilitas)"]
    )

    dim_info = DIMENSI_MAP[sel_tp] if sel_tp in DIMENSI_MAP else None
    if dim_info:
        imp_cols_tp  = [c for c in dim_info["imp"]  if c in df.columns]
        xyz_cols_tp  = [c for c in dim_info["xyz"]  if c in df.columns]
        komp_cols_tp = [c for c in dim_info["komp"] if c in df_has_komp.columns]

        min_tp = min(len(imp_cols_tp), len(xyz_cols_tp))
        if min_tp > 0:
            labels_tp = [short_label(c) for c in imp_cols_tp[:min_tp]]
            imp_tp    = [df[c].mean() for c in imp_cols_tp[:min_tp]]
            xyz_tp    = [df[c].mean() for c in xyz_cols_tp[:min_tp]]
            kom_tp    = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                         for c in komp_cols_tp[:min_tp]] if komp_cols_tp else []

            tp1, tp2 = st.columns([1.5, 1])

            with tp1:
                ipa_df = pd.DataFrame({
                    'Item': labels_tp,
                    'Importance': imp_tp,
                    'Satisfaction': xyz_tp,
                })
                mean_i = np.nanmean(imp_tp)
                mean_s = np.nanmean(xyz_tp)

                fig_ipa_tp = px.scatter(
                    ipa_df, x='Satisfaction', y='Importance',
                    text='Item', color_discrete_sequence=[COLOR_XYZ]
                )
                if kom_tp:
                    fig_ipa_tp.add_trace(go.Scatter(
                        x=kom_tp, y=imp_tp, mode='markers',
                        name=target_komp,
                        marker=dict(size=10, symbol='x', color=COLOR_KOMP,
                                    line=dict(width=2))
                    ))
                fig_ipa_tp.update_traces(
                    marker=dict(size=12), textposition='top center',
                    selector=dict(mode='markers+text')
                )
                fig_ipa_tp.add_vline(x=mean_s, line_dash="dash", line_color="#94A3B8")
                fig_ipa_tp.add_hline(y=mean_i, line_dash="dash", line_color="#94A3B8")
                fig_ipa_tp.add_annotation(x=ipa_df['Satisfaction'].min(), y=ipa_df['Importance'].max(),
                    text="⚠️ PERBAIKI SEGERA", showarrow=False,
                    font=dict(color=COLOR_KOMP, size=10))
                fig_ipa_tp.add_annotation(x=ipa_df['Satisfaction'].max(), y=ipa_df['Importance'].max(),
                    text="🌟 PERTAHANKAN", showarrow=False,
                    font=dict(color="#10B981", size=10))
                st.plotly_chart(
                    elite_layout(fig_ipa_tp, f"IPA Matrix — {sel_tp}"),
                    use_container_width=True
                )

            with tp2:
                # Gap XYZ vs kompetitor per item
                if kom_tp:
                    gap_tp = pd.DataFrame({
                        'Item': labels_tp,
                        'Gap XYZ−Komp': [x - k for x, k in zip(xyz_tp, kom_tp)]
                    }).sort_values('Gap XYZ−Komp')
                    gap_tp['Warna'] = np.where(gap_tp['Gap XYZ−Komp'] < 0, COLOR_KOMP, COLOR_XYZ)
                    fig_gap_tp = px.bar(
                        gap_tp, x='Gap XYZ−Komp', y='Item',
                        orientation='h', text='Gap XYZ−Komp'
                    )
                    fig_gap_tp.update_traces(
                        marker_color=gap_tp['Warna'],
                        texttemplate='%{text:.2f}', textposition='outside'
                    )
                    st.plotly_chart(
                        elite_layout(fig_gap_tp, f"Gap Kompetitif — {sel_tp}"),
                        use_container_width=True
                    )
                else:
                    st.info("Data kompetitor tidak tersedia untuk touchpoint ini.")

    st.markdown("---")

    # ── Jenis transaksi × skor layanan ───────────────────────────────
    st.markdown("<div class='section-header'>🔄 Jenis Transaksi vs Skor Layanan</div>",
        unsafe_allow_html=True)

    d1_xyz = pd.DataFrame({
        'Panel':   ['Teller', 'CS', 'Keduanya'],
        'Skor OVR': [
            df[df['D1_TYPE'] == 'TELLER']['OVR_TELLER_XYZ'].mean(),
            df[df['D1_TYPE'] == 'CS']['OVR_CS_XYZ'].mean(),
            df[df['D1_TYPE'] == 'BOTH'][['OVR_TELLER_XYZ','OVR_CS_XYZ']].mean(axis=1).mean()
        ]
    })
    fig_d1 = px.bar(
        d1_xyz, x='Panel', y='Skor OVR',
        color='Panel', text='Skor OVR',
        color_discrete_sequence=[COLOR_XYZ, COLOR_KOMP, '#F59E0B']
    )
    fig_d1.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    fig_d1.update_yaxes(range=[4.5, 6.3])
    st.plotly_chart(
        elite_layout(fig_d1, "Rata-rata Skor Layanan Berdasarkan Jenis Transaksi"),
        use_container_width=True
    )

# =====================================================================
# TAB 5 — EMOSI, LOYALITAS & DIGITALISASI
# =====================================================================
with tab5:
    e1, e2 = st.columns(2)

    with e1:
        st.markdown("<div class='section-header'>😊 Emotional Experience — XYZ vs Kompetitor</div>",
            unsafe_allow_html=True)

        # Kolom emosi positif XYZ (T_I1A_2, _5, _8, _11, _14, _17, _20, _23, _26)
        emo_pos_xyz_cols = ["T_I1A_2","T_I1A_5","T_I1A_8","T_I1A_11",
                            "T_I1A_14","T_I1A_17","T_I1A_20","T_I1A_23","T_I1A_26"]
        emo_neg_xyz_cols = ["T_I1A_29","T_I1A_32","T_I1A_35","T_I1A_38",
                            "T_I1A_41","T_I1A_44","T_I1A_47"]
        emo_pos_kom_cols = ["T_I1A_3","T_I1A_6","T_I1A_9","T_I1A_12",
                            "T_I1A_15","T_I1A_18","T_I1A_21","T_I1A_24","T_I1A_27"]
        emo_neg_kom_cols = ["T_I1A_30","T_I1A_33","T_I1A_36","T_I1A_39",
                            "T_I1A_42","T_I1A_45","T_I1A_48"]

        emo_labels_pos = ["Bahagia","Percaya","Dihargai","Diperhatikan",
                          "Aman","Fokus","Dimanjakan","Tertarik","Penuh Semangat"]
        emo_labels_neg = ["Tidak Puas","Frustasi","Kecewa","Tertekan",
                          "Tidak Bahagia","Diabaikan","Tergesa-gesa"]

        emo_pos_xyz = [df[c].mean() for c in emo_pos_xyz_cols if c in df.columns]
        emo_pos_kom = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                       for c in emo_pos_kom_cols]
        emo_neg_xyz = [df[c].mean() for c in emo_neg_xyz_cols if c in df.columns]
        emo_neg_kom = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
                       for c in emo_neg_kom_cols]

        fig_emo_pos = go.Figure()
        fig_emo_pos.add_trace(go.Bar(
            name='XYZ', x=emo_labels_pos, y=emo_pos_xyz,
            marker_color=COLOR_XYZ, text=np.round(emo_pos_xyz, 2),
            textposition='outside'
        ))
        fig_emo_pos.add_trace(go.Bar(
            name=target_komp, x=emo_labels_pos, y=emo_pos_kom,
            marker_color=COLOR_KOMP, text=np.round(emo_pos_kom, 2),
            textposition='outside'
        ))
        fig_emo_pos.update_layout(barmode='group', yaxis_range=[3, 6.5])
        st.plotly_chart(
            elite_layout(fig_emo_pos, "😊 Emosi Positif (Makin Tinggi Makin Baik)"),
            use_container_width=True
        )

    with e2:
        st.markdown("<div class='section-header'>😠 Emosi Negatif — XYZ vs Kompetitor</div>",
            unsafe_allow_html=True)
        fig_emo_neg = go.Figure()
        fig_emo_neg.add_trace(go.Bar(
            name='XYZ', x=emo_labels_neg, y=emo_neg_xyz,
            marker_color=COLOR_XYZ, text=np.round(emo_neg_xyz, 2),
            textposition='outside'
        ))
        fig_emo_neg.add_trace(go.Bar(
            name=target_komp, x=emo_labels_neg, y=emo_neg_kom,
            marker_color=COLOR_KOMP, text=np.round(emo_neg_kom, 2),
            textposition='outside'
        ))
        fig_emo_neg.update_layout(barmode='group', yaxis_range=[1, 4])
        st.plotly_chart(
            elite_layout(fig_emo_neg, "😠 Emosi Negatif (Makin Rendah Makin Baik)"),
            use_container_width=True
        )

    st.markdown("---")

    # ── Brand equity T_H1A_ ───────────────────────────────────────────
    st.markdown("<div class='section-header'>💎 Brand Equity & Loyalitas — 15 Atribut</div>",
        unsafe_allow_html=True)

    h1a_xyz_cols = ["T_H1A_2","T_H1A_5","T_H1A_8","T_H1A_11","T_H1A_14",
                    "T_H1A_17","T_H1A_20","T_H1A_23","T_H1A_26","T_H1A_29",
                    "T_H1A_32","T_H1A_35","T_H1A_38","T_H1A_41","T_H1A_44"]
    h1a_kom_cols = ["T_H1A_3","T_H1A_6","T_H1A_9","T_H1A_12","T_H1A_15",
                    "T_H1A_18","T_H1A_21","T_H1A_24","T_H1A_27","T_H1A_30",
                    "T_H1A_33","T_H1A_36","T_H1A_39","T_H1A_42","T_H1A_45"]
    h1a_labels = [
        "Tetap Gunakan","Kemudahan Transaksi","Digunakan Banyak Orang",
        "Keuntungan Finansial","Produk Lengkap","Promo Gaya Hidup",
        "Kecepatan Transaksi","Rasa Aman","Kenyamanan Fasilitas",
        "Merasa Dihargai","Bangga","Up to Date/Modern",
        "Bank Turun-Temurun","Cukup Satu Bank","Bergengsi"
    ]

    h_xyz = [df[c].mean() for c in h1a_xyz_cols if c in df.columns]
    h_kom = [df_has_komp[c].mean() if c in df_has_komp.columns else np.nan
             for c in h1a_kom_cols]

    fig_h1a = go.Figure()
    fig_h1a.add_trace(go.Bar(
        name='Bank XYZ', x=h1a_labels, y=h_xyz,
        marker_color=COLOR_XYZ, text=np.round(h_xyz, 2), textposition='outside'
    ))
    fig_h1a.add_trace(go.Bar(
        name=target_komp, x=h1a_labels, y=h_kom,
        marker_color=COLOR_KOMP, text=np.round(h_kom, 2), textposition='outside'
    ))
    fig_h1a.update_layout(barmode='group', yaxis_range=[3, 6.5], height=420)
    st.plotly_chart(
        elite_layout(fig_h1a, "Brand Equity XYZ vs Kompetitor (Skala 1–6)"),
        use_container_width=True
    )

    st.markdown("---")

    # ── Digitalisasi T_J1_ ────────────────────────────────────────────
    st.markdown("<div class='section-header'>📱 Persepsi Digitalisasi Cabang</div>",
        unsafe_allow_html=True)

    dig_cols   = ["T_J1_1","T_J1_2","T_J1_3","T_J1_4","T_J1_5"]
    dig_labels = [
        "Digitalisasi Layanan Cabang",
        "Digital Signage",
        "Smart Table",
        "Tablet Survey",
        "Akses Cabang"
    ]
    dig_vals = [df[c].mean() for c in dig_cols if c in df.columns]

    dg1, dg2 = st.columns(2)
    with dg1:
        fig_dig = px.bar(
            x=dig_vals, y=dig_labels, orientation='h',
            color=dig_vals, color_continuous_scale='Blues',
            text=np.round(dig_vals, 2)
        )
        fig_dig.update_traces(textposition='outside')
        fig_dig.update_xaxes(range=[3, 6.5])
        st.plotly_chart(
            elite_layout(fig_dig, "Skor Persepsi Digitalisasi (Skala 1–6)"),
            use_container_width=True
        )

    with dg2:
        # Sarana elektronik T_SL2_
        sl_cols   = [f"T_SL2_{i}" for i in range(1, 17) if f"T_SL2_{i}" in df.columns]
        sl_labels = [short_label(c) for c in sl_cols]
        sl_vals   = [df[c].mean() for c in sl_cols]

        fig_sl = px.bar(
            x=sl_vals, y=sl_labels, orientation='h',
            color=sl_vals, color_continuous_scale='Teal',
            text=np.round(sl_vals, 2)
        )
        fig_sl.update_traces(textposition='outside')
        fig_sl.update_xaxes(range=[3, 6.5])
        st.plotly_chart(
            elite_layout(fig_sl, "Ketersediaan & Fungsi Sarana Elektronik"),
            use_container_width=True
        )

    # ── Korelasi emosi vs outcome ─────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Korelasi Emosi Positif vs Outcome</div>",
        unsafe_allow_html=True)

    emo_score_xyz = df[[c for c in emo_pos_xyz_cols if c in df.columns]].mean(axis=1)
    corr_emo = pd.DataFrame({
        'Emosi Positif': emo_score_xyz,
        'NPS':           df['G1A'],
        'Kepuasan':      df['E1A'],
        'Loyalitas':     df['F1A'],
    }).corr()

    fig_emo_corr = px.imshow(
        corr_emo, text_auto=".2f",
        color_continuous_scale='Blues', aspect='auto'
    )
    st.plotly_chart(
        elite_layout(fig_emo_corr, "Korelasi: Emosi Positif vs NPS, Kepuasan, Loyalitas"),
        use_container_width=True
    )

# =====================================================================
# TAB 6 — PROFIL & SEGMENTASI
# =====================================================================
with tab6:
    st.markdown("<div class='section-header'>👥 Profil Demografis Responden</div>",
        unsafe_allow_html=True)

    d1c, d2c, d3c = st.columns(3)
    with d1c:
        fig_g = px.pie(df, names='S1', hole=0.55,
            color_discrete_sequence=[COLOR_XYZ, '#60A5FA'])
        st.plotly_chart(elite_layout(fig_g, "Gender"), use_container_width=True)

    with d2c:
        age_count = df['S2_2'].value_counts().reset_index()
        age_count.columns = ['Usia', 'Jumlah']
        fig_age = px.bar(age_count, x='Usia', y='Jumlah',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_age.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_age, "Kelompok Usia"), use_container_width=True)

    with d3c:
        ten_count = df['S4'].value_counts().reset_index()
        ten_count.columns = ['Tenure', 'Jumlah']
        fig_ten = px.bar(ten_count, x='Jumlah', y='Tenure', orientation='h',
            color_discrete_sequence=['#10B981'], text='Jumlah')
        fig_ten.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_ten, "Lama Menjadi Nasabah"), use_container_width=True)

    d4c, d5c = st.columns(2)
    with d4c:
        edu_count = df['P3'].value_counts().reset_index()
        edu_count.columns = ['Pendidikan', 'Jumlah']
        fig_edu = px.bar(edu_count, x='Jumlah', y='Pendidikan', orientation='h',
            color_discrete_sequence=[COLOR_XYZ], text='Jumlah')
        fig_edu.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_edu, "Pendidikan Terakhir"), use_container_width=True)

    with d5c:
        job_count = df['P4'].value_counts().reset_index()
        job_count.columns = ['Pekerjaan', 'Jumlah']
        fig_job = px.bar(job_count, x='Jumlah', y='Pekerjaan', orientation='h',
            color_discrete_sequence=['#8B5CF6'], text='Jumlah')
        fig_job.update_traces(textposition='outside')
        st.plotly_chart(elite_layout(fig_job, "Pekerjaan"), use_container_width=True)

    # ── Treemap geografis ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>🗺️ Peta Geografis Responden</div>",
        unsafe_allow_html=True)
    geo_df = df.groupby(['PROV','KABKOTA','CABANG']).agg(
        Jumlah=('SERIAL','count'),
        NPS_Mean=('G1A','mean'),
        Kepuasan_Mean=('E1A','mean')
    ).reset_index()

    fig_tree = px.treemap(
        geo_df,
        path=[px.Constant("Nasional"), 'PROV', 'KABKOTA', 'CABANG'],
        values='Jumlah',
        color='NPS_Mean',
        color_continuous_scale='Blues',
        hover_data={'Kepuasan_Mean': ':.2f', 'NPS_Mean': ':.2f'}
    )
    fig_tree.update_layout(height=500)
    st.plotly_chart(elite_layout(fig_tree, "Treemap: Distribusi Responden (Warna = NPS)"),
        use_container_width=True)

    st.markdown("---")

    # ── Cross-tab segmentasi ──────────────────────────────────────────
    st.markdown("<div class='section-header'>🔀 Analisis Segmentasi</div>",
        unsafe_allow_html=True)

    seg_col  = st.selectbox("Segmentasi berdasarkan:", [
        'S1 (Gender)', 'S2_2 (Usia)', 'S4 (Tenure)',
        'S7 (Frekuensi Transaksi)', 'P3 (Pendidikan)',
        'P4 (Pekerjaan)', 'P1 (Status Nikah)'
    ])
    seg_key  = seg_col.split(' ')[0]
    seg_metr = st.selectbox("Metrik yang dibandingkan:", [
        'G1A (NPS)', 'E1A (Kepuasan)', 'F1A (Loyalitas)',
        'OVR_TELLER_XYZ', 'OVR_CS_XYZ', 'OVR_ATM_XYZ'
    ])
    metr_key = seg_metr.split(' ')[0]

    if seg_key in df.columns and metr_key in df.columns:
        seg_df = df.groupby(seg_key)[metr_key].mean().reset_index()
        seg_df.columns = ['Segmen', 'Nilai']
        seg_df = seg_df.sort_values('Nilai', ascending=True)
        fig_seg = px.bar(
            seg_df, x='Nilai', y='Segmen', orientation='h',
            color='Nilai', color_continuous_scale='Blues',
            text='Nilai'
        )
        fig_seg.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(
            elite_layout(fig_seg, f"Rata-rata {seg_metr} per {seg_col}"),
            use_container_width=True
        )

    # ── Frekuensi transaksi vs NPS ────────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Frekuensi Transaksi vs NPS & Kepuasan</div>",
        unsafe_allow_html=True)
    freq_df = df.groupby('S7').agg(
        NPS_Mean=('G1A','mean'),
        Kepuasan_Mean=('E1A','mean'),
        Loyalitas_Mean=('F1A','mean'),
        Count=('SERIAL','count')
    ).reset_index().sort_values('NPS_Mean', ascending=False)

    fig_freq = px.bar(
        freq_df.melt(id_vars='S7', value_vars=['NPS_Mean','Kepuasan_Mean','Loyalitas_Mean']),
        x='S7', y='value', color='variable', barmode='group',
        color_discrete_map={
            'NPS_Mean': COLOR_XYZ,
            'Kepuasan_Mean': '#10B981',
            'Loyalitas_Mean': '#F59E0B'
        },
        text='value'
    )
    fig_freq.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    st.plotly_chart(
        elite_layout(fig_freq, "Frekuensi Transaksi vs Outcome Nasabah"),
        use_container_width=True
    )

    # ── Tujuan buka rekening vs loyalitas ────────────────────────────
    st.markdown("<div class='section-header'>🎯 Tujuan Buka Rekening vs Loyalitas</div>",
        unsafe_allow_html=True)
    tujuan = df['A2'].dropna().str.split(';').explode().str.strip()
    tujuan_df = tujuan.to_frame('Tujuan').join(df['F1A'], how='left')
    tujuan_agg = tujuan_df.groupby('Tujuan')['F1A'].mean().reset_index()
    tujuan_agg.columns = ['Tujuan', 'Loyalitas']
    tujuan_agg = tujuan_agg.sort_values('Loyalitas', ascending=True)

    fig_tujuan = px.bar(
        tujuan_agg, x='Loyalitas', y='Tujuan', orientation='h',
        color_discrete_sequence=[COLOR_XYZ], text='Loyalitas'
    )
    fig_tujuan.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    fig_tujuan.update_xaxes(range=[4.5, 6.3])
    st.plotly_chart(
        elite_layout(fig_tujuan, "Rata-rata Loyalitas Berdasarkan Tujuan Buka Rekening"),
        use_container_width=True
    )

# =====================================================================
# TAB 7 — VOICE OF CUSTOMER
# =====================================================================
with tab7:
    st.markdown("<div class='section-header'>💬 Analisis Voice of Customer</div>",
        unsafe_allow_html=True)

    # ── NPS distribution ──────────────────────────────────────────────
    v1, v2 = st.columns(2)
    with v1:
        fig_nps_hist = px.histogram(
            df, x='G1A', nbins=10,
            color='G1A_CAT', color_discrete_map=NPS_COLORS,
            labels={'G1A': 'Skor NPS', 'count': 'Jumlah'}
        )
        st.plotly_chart(
            elite_layout(fig_nps_hist, "Distribusi Skor NPS XYZ"),
            use_container_width=True
        )
    with v2:
        nps_kom_valid = df_has_komp['G1C'].dropna()
        if len(nps_kom_valid) > 0:
            fig_nps_k = px.histogram(
                df_has_komp, x='G1C', nbins=10,
                color='G1C_CAT', color_discrete_map=NPS_COLORS,
                labels={'G1C': 'Skor NPS Kompetitor'}
            )
            st.plotly_chart(
                elite_layout(fig_nps_k, f"Distribusi Skor NPS {target_komp}"),
                use_container_width=True
            )

    st.markdown("---")

    # ── Wordcloud ─────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>☁️ Wordcloud Alasan NPS</div>",
        unsafe_allow_html=True)

    filter_nps_cat = st.radio(
        "Filter Kategori NPS:",
        ["Semua", "Promoter", "Passive", "Detractor"],
        horizontal=True
    )

    wc1, wc2 = st.columns(2)

    def build_wordcloud(text_series, colormap='Blues', max_words=100):
        text = " ".join(text_series.dropna().astype(str).tolist()).lower()
        stopwords_id = {
            'yang','untuk','dengan','pada','dari','sebagai','tidak','karena',
            'sangat','lebih','sudah','saya','bank','bisa','dan','di','ke',
            'ini','itu','ada','juga','net','subnet','positive','negative',
            'comments','dalam','oleh','akan','telah','dapat','kami','anda',
            'nya','atau','jadi','baru','lagi','saat','pernah','masih'
        }
        if len(text.strip()) < 10:
            return None
        wc = WordCloud(
            width=800, height=400,
            background_color='#F8FAFC',
            colormap=colormap,
            max_words=max_words,
            stopwords=stopwords_id
        ).generate(text)
        return wc

    # Filter data berdasarkan kategori NPS
    df_wc_xyz = df if filter_nps_cat == "Semua" \
                else df[df['G1A_CAT'] == filter_nps_cat]
    df_wc_kom = df_has_komp if filter_nps_cat == "Semua" \
                else df_has_komp[df_has_komp['G1C_CAT'] == filter_nps_cat]

    with wc1:
        st.markdown(f"**☁️ Alasan NPS — Bank XYZ ({filter_nps_cat})**")
        wc_xyz = build_wordcloud(df_wc_xyz['G1B'], colormap='Blues')
        if wc_xyz:
            fig_wc, ax = plt.subplots(figsize=(8, 4))
            ax.imshow(wc_xyz, interpolation='bilinear')
            ax.axis('off')
            fig_wc.patch.set_facecolor('#F8FAFC')
            st.pyplot(fig_wc)
            plt.close()
        else:
            st.info("Tidak ada teks yang cukup untuk wordcloud.")

    with wc2:
        st.markdown(f"**☁️ Alasan NPS — {target_komp} ({filter_nps_cat})**")
        wc_kom = build_wordcloud(df_wc_kom['G1D'], colormap='Reds')
        if wc_kom:
            fig_wc2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.imshow(wc_kom, interpolation='bilinear')
            ax2.axis('off')
            fig_wc2.patch.set_facecolor('#F8FAFC')
            st.pyplot(fig_wc2)
            plt.close()
        else:
            st.info("Tidak ada teks yang cukup untuk wordcloud.")

    st.markdown("---")

    # ── Top keyword ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔑 Top Keyword per Kategori NPS</div>",
        unsafe_allow_html=True)

    stopwords_id = {
        'yang','untuk','dengan','pada','dari','sebagai','tidak','karena',
        'sangat','lebih','sudah','saya','bank','bisa','dan','di','ke',
        'ini','itu','ada','juga','net','subnet','positive','negative',
        'comments','dalam','oleh','akan','telah','dapat','kami','anda'
    }

    kw1, kw2, kw3 = st.columns(3)
    for col_kw, cat, color in zip(
        [kw1, kw2, kw3],
        ['Promoter','Passive','Detractor'],
        [COLOR_XYZ, '#F59E0B', COLOR_KOMP]
    ):
        text_cat = " ".join(
            df[df['G1A_CAT'] == cat]['G1B'].dropna().astype(str).tolist()
        ).lower()
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_cat)
        words_clean = [w for w in words if w not in stopwords_id]
        if words_clean:
            top_words = Counter(words_clean).most_common(8)
            kw_df = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
            fig_kw = px.bar(
                kw_df.sort_values('Frekuensi'),
                x='Frekuensi', y='Kata',
                orientation='h',
                color_discrete_sequence=[color],
                text='Frekuensi'
            )
            fig_kw.update_traces(textposition='outside')
            col_kw.plotly_chart(
                elite_layout(fig_kw, f"Top Kata — {cat}"),
                use_container_width=True
            )

    st.markdown("---")

    # ── Saran perbaikan T_J2_ ─────────────────────────────────────────
    st.markdown("<div class='section-header'>📝 Saran Perbaikan Digitalisasi</div>",
        unsafe_allow_html=True)

    j2_cols = {
        "T_J2_1": "Digitalisasi Layanan",
        "T_J2_2": "Digital Signage",
        "T_J2_3": "Smart Table",
        "T_J2_4": "Tablet Survey",
        "T_J2_5": "Akses Cabang"
    }
    sel_j2 = st.selectbox("Pilih Topik Saran:", list(j2_cols.values()))
    sel_j2_col = [k for k, v in j2_cols.items() if v == sel_j2][0]

    if sel_j2_col in df.columns:
        saran_df = df[sel_j2_col].dropna().value_counts().reset_index()
        saran_df.columns = ['Saran', 'Jumlah']
        saran_df = saran_df[saran_df['Saran'].str.strip() != '']
        st.dataframe(saran_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Verbatim table ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Log Verbatim Nasabah</div>",
        unsafe_allow_html=True)

    v_filter = st.radio(
        "Tampilkan:", ["Semua", "Promoter", "Passive", "Detractor"],
        horizontal=True
    )
    search_text = st.text_input("🔎 Cari kata kunci dalam komentar:")

    verb_df = df if v_filter == "Semua" else df[df['G1A_CAT'] == v_filter]
    if search_text:
        verb_df = verb_df[
            verb_df['G1B'].fillna('').str.contains(search_text, case=False)
        ]

    st.dataframe(
        verb_df[['CABANG','PROV','S1','S2_2','S4','G1A','G1A_CAT','G1B']]
        .rename(columns={
            'CABANG':'Cabang','PROV':'Provinsi','S1':'Gender',
            'S2_2':'Usia','S4':'Tenure','G1A':'Skor NPS',
            'G1A_CAT':'Kategori','G1B':'Komentar'
        })
        .sort_values('Skor NPS'),
        use_container_width=True, height=350, hide_index=True
    )

st.markdown("---")
st.caption("🚀 Bank XYZ — Executive CX Intelligence Dashboard | Powered by Streamlit & Plotly")
