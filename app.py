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
    st.error("Library 'wordcloud' atau 'matplotlib' belum terinstal. Cek requirements.txt!")
    st.stop()

# =====================================================================
# 1. KONFIGURASI & TEMA SAAS ULTRA-PREMIUM
# =====================================================================
st.set_page_config(page_title="Executive CX Analytics - Bank XYZ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC !important; }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 2px 0 15px rgba(0,0,0,0.03);
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0F172A !important; font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #E2E8F0; padding: 24px; border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        text-align: center; transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.15), 0 4px 6px -2px rgba(37, 99, 235, 0.05);
        border-color: #BFDBFE;
    }
    .metric-title { color: #64748B; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
    .metric-value { color: #0F172A; font-size: 38px; font-weight: 900; line-height: 1; }
    .metric-value.highlight { color: #2563EB; }
    .metric-value.danger { color: #E11D48; }

    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff; border: 1px solid #E2E8F0; border-bottom: none;
        border-radius: 8px 8px 0 0; padding: 10px 24px; color: #64748B !important;
        font-weight: 600; transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important; color: #ffffff !important;
        border-color: #2563EB; box-shadow: 0 -4px 6px -1px rgba(37, 99, 235, 0.1);
    }
    h1, h2, h3, h4 { color: #0F172A !important; font-weight: 800 !important; }
    p { color: #334155 !important; }
    </style>
""", unsafe_allow_html=True)

nps_colors = {'Promoter': '#10B981', 'Passive': '#F59E0B', 'Detractor': '#EF4444'}
color_primary, color_secondary = '#1E3A8A', '#3B82F6'

# FIX CROPPING: Penambahan margin kiri (l=180) dan automargin pada axis
def elite_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#0F172A', weight='bold')),
        template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=20, l=180, r=20), font=dict(color='#334155', size=12, weight='bold'),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#FFFFFF", font_size=13, bordercolor="#FFFFFF"),
        legend=dict(font=dict(color="#0F172A"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        coloraxis_colorbar=dict(title=dict(font=dict(color="#0F172A")), tickfont=dict(color="#0F172A"))
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9', zeroline=False, automargin=True, title_font=dict(color='#64748B', weight='bold'), tickfont=dict(color='#475569', weight='bold'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F1F5F9', zeroline=False, automargin=True, title_font=dict(color='#64748B', weight='bold'), tickfont=dict(color='#475569', weight='bold'))
    return fig

# =====================================================================
# 2. LOAD & PRE-PROCESS DATA
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_excel('Dataset_Dashboard_Perfect.xlsx', engine='openpyxl')
    df['NPS_Score'] = df['G1A'].astype(str).str.extract(r'(\d+)').astype(float)
    def categorize_nps(score):
        if pd.isna(score): return 'Unknown'
        return 'Promoter' if score >= 9 else ('Passive' if score >= 7 else 'Detractor')
    df['NPS_Category'] = df['NPS_Score'].apply(categorize_nps)
    df['G1B'] = df['G1B'].fillna('')
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# =====================================================================
# 3. SUPER SIDEBAR (CASCADING & BENCHMARK FILTERS)
# =====================================================================
st.sidebar.markdown("<h3 style='color: #2563EB !important; font-weight: 800;'>🎛️ Filter Analitik</h3>", unsafe_allow_html=True)

# Filter Kompetitor Utama (Untuk Komparasi IPA & Radar)
komp_list = sorted(df_raw['KOMP'].replace(' ', np.nan).dropna().unique().tolist())
target_komp = st.sidebar.selectbox("🎯 Target Benchmarking (Kompetitor)", ["Semua Kompetitor (Rata-rata)"] + komp_list)

st.sidebar.markdown("---")
st.sidebar.caption("Saring Segmen Nasabah:")
prov_options = sorted(df_raw['PROV'].dropna().unique().tolist())
selected_prov = st.sidebar.multiselect("📍 Provinsi", prov_options, default=[])

cabang_options = sorted(df_raw[df_raw['PROV'].isin(selected_prov)]['CABANG'].dropna().unique().tolist()) if selected_prov else sorted(df_raw['CABANG'].dropna().unique().tolist())
selected_cabang = st.sidebar.multiselect("🏢 Kantor Cabang", cabang_options, default=[])

selected_gender = st.sidebar.multiselect("🚻 Jenis Kelamin", sorted(df_raw['S1'].dropna().unique().tolist()), default=[])
selected_age = st.sidebar.multiselect("🎂 Rentang Usia", sorted(df_raw['S2_2'].dropna().unique().tolist()), default=[])
selected_tenure = st.sidebar.multiselect("⏳ Lama Hubungan", sorted(df_raw['S4'].dropna().unique().tolist()), default=[])

# LOGIKA FILTERING DATAFRAME
df = df_raw.copy()
if selected_prov: df = df[df['PROV'].isin(selected_prov)]
if selected_cabang: df = df[df['CABANG'].isin(selected_cabang)]
if selected_gender: df = df[df['S1'].isin(selected_gender)]
if selected_age: df = df[df['S2_2'].isin(selected_age)]
if selected_tenure: df = df[df['S4'].isin(selected_tenure)]

# Dataframe Khusus Kompetitor (Merespons Filter Sidebar)
if target_komp == "Semua Kompetitor (Rata-rata)":
    df_komp = df.copy()
else:
    df_komp = df[df['KOMP'] == target_komp]

st.sidebar.markdown("---")
st.sidebar.success(f"📊 Volume Nasabah: {len(df):,}")

st.markdown("<h2 style='text-align: center; margin-bottom: 25px; font-weight: 900; letter-spacing: -1px; color: #0F172A !important;'>🏦 BANK XYZ - EXECUTIVE CX INTELLIGENCE</h2>", unsafe_allow_html=True)

if df.empty:
    st.warning("Data kosong. Sesuaikan kembali filter Anda.")
    st.stop()

# =====================================================================
# 4. TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌟 Eksekutif", "👥 Demografi", "🏆 Brand Benchmarking", 
    "🎯 Touchpoint & IPA", "🏢 Fasilitas & Digital", "💬 NLP Verbatim"
])

# ---------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-title'>Net Promoter Score (0-10)</div><div class='metric-value highlight'>{df['NPS_Score'].mean():.1f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-title'>Loyalitas Inti (1-6)</div><div class='metric-value'>{df['Outcome_Loyalitas_Inti_XYZ'].mean():.2f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-title'>Emosi Positif</div><div class='metric-value'>{df['Outcome_Emosi_Positif_XYZ'].mean():.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-title'>Emosi Negatif (Makin Kecil Makin Baik)</div><div class='metric-value danger'>{df['Outcome_Emosi_Negatif_XYZ'].mean():.2f}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns([1.2, 2])
    with r2c1:
        nps_count = df[df['NPS_Category'] != 'Unknown']['NPS_Category'].value_counts().reset_index()
        fig_nps = px.pie(nps_count, values='count', names='NPS_Category', hole=0.55, color='NPS_Category', color_discrete_map=nps_colors)
        fig_nps = elite_layout(fig_nps, "Komposisi Sentimen NPS")
        fig_nps.update_traces(textposition='outside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
        st.plotly_chart(fig_nps, use_container_width=True)

    with r2c2:
        top_cabang = df.groupby('CABANG')['Outcome_Loyalitas_Inti_XYZ'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_cabang = px.bar(top_cabang, x='Outcome_Loyalitas_Inti_XYZ', y='CABANG', orientation='h', color_discrete_sequence=['#3B82F6'])
        fig_cabang = elite_layout(fig_cabang, "Top 10 Cabang dg Loyalitas Tertinggi")
        fig_cabang.update_traces(texttemplate='%{x:.2f}', textposition='inside', marker_cornerradius=5)
        fig_cabang.update_xaxes(range=[4, 6.5])
        st.plotly_chart(fig_cabang, use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig_scatter = px.scatter(df, x="Outcome_Emosi_Positif_XYZ", y="Outcome_Loyalitas_Inti_XYZ", color="NPS_Category", color_discrete_map=nps_colors, hover_data=['CABANG'], trendline="ols")
        fig_scatter = elite_layout(fig_scatter, "Korelasi: Emosi Positif vs Loyalitas Aktual")
        fig_scatter.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='White')))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with r3c2:
        macro_corr = df[['Outcome_Loyalitas_Inti_XYZ', 'Outcome_Emosi_Positif_XYZ', 'Outcome_Persepsi_Digitalisasi', 'Overall_CS_XYZ', 'Overall_Teller_XYZ']].corr()
        fig_mcorr = px.imshow(macro_corr, text_auto=".2f", color_continuous_scale='Blues', aspect='auto')
        fig_mcorr = elite_layout(fig_mcorr, "Matriks Korelasi Indikator Utama")
        st.plotly_chart(fig_mcorr, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: DEMOGRAFI & PROFIL
# ---------------------------------------------------------------------
with tab2:
    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        fig_gender = px.pie(df, names='S1', hole=0.55, color_discrete_sequence=['#1E40AF', '#60A5FA'])
        st.plotly_chart(elite_layout(fig_gender, "Proporsi Gender (S1)"), use_container_width=True)
    with r1c2:
        fig_age = px.bar(df['S2_2'].value_counts().reset_index(), x='S2_2', y='count', color_discrete_sequence=['#3B82F6'])
        st.plotly_chart(elite_layout(fig_age, "Distribusi Kelompok Usia"), use_container_width=True)
    with r1c3:
        fig_tenure = px.bar(df['S4'].value_counts().reset_index(), x='count', y='S4', orientation='h', color_discrete_sequence=['#10B981'])
        st.plotly_chart(elite_layout(fig_tenure, "Lama Menjadi Nasabah"), use_container_width=True)

    r2c1, r2c2 = st.columns([2, 1])
    with r2c1:
        fig_tree = px.treemap(df, path=[px.Constant("Nasabah Nasional"), 'PROV', 'CABANG'], color='NPS_Score', color_continuous_scale='Blues')
        st.plotly_chart(elite_layout(fig_tree, "Peta Geografis (Warna = Intensitas NPS)"), use_container_width=True)
    with r2c2:
        komp_counts = df['KOMP'].replace(' ', np.nan).dropna().value_counts().head(5).reset_index()
        fig_komp = px.pie(komp_counts, names='KOMP', values='count', hole=0.55, color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(elite_layout(fig_komp, "Top Kompetitor Utama"), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: BRAND BENCHMARKING (DENGAN KOMPETITOR TARGET)
# ---------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)
    brand_dim = ['Reputasi', 'Jaringan_Fisik', 'Digital_Channel', 'Finansial_Produk', 'Kemudahan_Solusi', 'Koneksi_Emosional']
    with c1:
        fig_radar_brand = go.Figure()
        fig_radar_brand.add_trace(go.Scatterpolar(r=[df[f'Kinerja_{d}_XYZ'].mean() for d in brand_dim], theta=brand_dim, fill='toself', name='Bank XYZ', line_color='#2563EB'))
        fig_radar_brand.add_trace(go.Scatterpolar(r=[df_komp[f'Kinerja_{d}_Komp'].mean() for d in brand_dim], theta=brand_dim, fill='toself', name=target_komp, line_color='#EF4444'))
        fig_radar_brand.update_layout(polar=dict(radialaxis=dict(visible=True, range=[4, 6], gridcolor='#E2E8F0'), bgcolor='#F8FAFC'))
        st.plotly_chart(elite_layout(fig_radar_brand, f"Pemetaan Citra Merek (vs {target_komp})"), use_container_width=True)
        
    with c2:
        atm_dim = ['Keamanan', 'Aksesibilitas_Ketersediaan', 'Keandalan_Mesin', 'Fitur_Fungsionalitas', 'Kenyamanan_Fisik']
        fig_radar_atm = go.Figure()
        fig_radar_atm.add_trace(go.Scatterpolar(r=[df[f'Kinerja_ATM_{d}_XYZ'].mean() for d in atm_dim], theta=atm_dim, fill='toself', name='ATM XYZ', line_color='#2563EB'))
        fig_radar_atm.add_trace(go.Scatterpolar(r=[df_komp[f'Kinerja_ATM_{d}_Komp'].mean() for d in atm_dim], theta=atm_dim, fill='toself', name=target_komp, line_color='#EF4444'))
        fig_radar_atm.update_layout(polar=dict(radialaxis=dict(visible=True, range=[4, 6], gridcolor='#E2E8F0'), bgcolor='#F8FAFC'))
        st.plotly_chart(elite_layout(fig_radar_atm, f"Perbandingan Daya Saing ATM (vs {target_komp})"), use_container_width=True)

    df_gap_brand = pd.DataFrame({'Dimensi': brand_dim, 'Harapan': [df[f'Harapan_{d}'].mean() for d in brand_dim], 'Kinerja XYZ': [df[f'Kinerja_{d}_XYZ'].mean() for d in brand_dim]}).melt(id_vars='Dimensi', var_name='Metrik', value_name='Skor')
    fig_gap_brand = px.bar(df_gap_brand, x='Dimensi', y='Skor', color='Metrik', barmode='group', color_discrete_map={'Harapan':'#94A3B8', 'Kinerja XYZ':'#3B82F6'})
    fig_gap_brand.update_traces(texttemplate='%{y:.2f}', textposition='outside', marker_cornerradius=3)
    fig_gap_brand.update_yaxes(range=[4.5, 6])
    st.plotly_chart(elite_layout(fig_gap_brand, "Analisis Kesenjangan: Harapan vs Kinerja Brand XYZ"), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: TOUCHPOINT & IPA MATRIKS (KONSOLIDASI & INTERAKTIF)
# ---------------------------------------------------------------------
with tab4:
    # Konsolidasi 3 Grafik Menjadi 1 Grafik Grup yang Lebih Insightful
    tp_labels = ['Satpam', 'Teller', 'Customer Service', 'Mesin ATM']
    xyz_scores = [df['Overall_Satpam_XYZ'].mean(), df['Overall_Teller_XYZ'].mean(), df['Overall_CS_XYZ'].mean(), df['Overall_ATM_XYZ'].mean()]
    komp_scores = [df_komp['Overall_Satpam_Komp'].mean(), df_komp['Overall_Teller_Komp'].mean(), df_komp['Overall_CS_Komp'].mean(), df_komp['Overall_ATM_Komp'].mean()]

    fig_tp = go.Figure(data=[
        go.Bar(name='Bank XYZ', x=tp_labels, y=xyz_scores, marker_color='#2563EB', text=np.round(xyz_scores, 2), textposition='auto'),
        go.Bar(name=target_komp, x=tp_labels, y=komp_scores, marker_color='#E11D48', text=np.round(komp_scores, 2), textposition='auto')
    ])
    fig_tp.update_layout(barmode='group', yaxis_range=[4, 6])
    st.plotly_chart(elite_layout(fig_tp, "Komparasi Overall Layanan: XYZ vs Kompetitor"), use_container_width=True)

    st.markdown("---")
    st.markdown("<h3 style='color:#0F172A; font-weight:800;'>🎯 Matriks IPA Kompetitif (Importance-Performance)</h3>", unsafe_allow_html=True)
    
    segmen_ipa = st.selectbox("Pilih Touchpoint untuk Dibedah Interaktif:", ["Customer Service", "Teller", "Satpam", "Fasilitas Cabang"])
    
    ipa_data = []
    if segmen_ipa == "Customer Service":
        dims = ['Antrian_Ketersediaan', 'Kecepatan_Kehandalan', 'Kompetensi_Solusi', 'Edukasi_Digital', 'Sikap_Keramahan', 'Cross_Selling', 'Penampilan_Atribut']
        for d in dims: ipa_data.append({'Indikator': d.replace('_', ' '), 'Harapan': df[f'Harapan_CS_{d}'].mean(), 'Kinerja_XYZ': df[f'Kinerja_CS_{d}_XYZ'].mean(), 'Kinerja_Komp': df_komp[f'Kinerja_CS_{d}_Komp'].mean()})
    elif segmen_ipa == "Teller":
        dims = ['Antrian_Ketersediaan', 'Kecepatan_Akurasi', 'Sikap_Keramahan', 'Edukasi_Digital', 'Penampilan_Atribut']
        for d in dims: ipa_data.append({'Indikator': d.replace('_', ' '), 'Harapan': df[f'Harapan_Teller_{d}'].mean(), 'Kinerja_XYZ': df[f'Kinerja_Teller_{d}_XYZ'].mean(), 'Kinerja_Komp': df_komp[f'Kinerja_Teller_{d}_Komp'].mean()})
    elif segmen_ipa == "Satpam":
        pairs = [("Penampilan", "Harapan_Penampilan_Satpam", "Kinerja_Penampilan_Satpam_XYZ", "Kinerja_Penampilan_Satpam_Komp"), ("Sikap Keramahan", "Harapan_Sikap_Keramahan_Satpam", "Kinerja_Sikap_Keramahan_Satpam_XYZ", "Kinerja_Sikap_Keramahan_Satpam_Komp"), ("Kompetensi Tugas", "Harapan_Kompetensi_Tugas_Satpam", "Kinerja_Kompetensi_Tugas_Satpam_XYZ", "Kinerja_Kompetensi_Tugas_Satpam_Komp"), ("Kesiapan Kehadiran", "Harapan_Kesiapan_Kehadiran_Satpam", "Kinerja_Kesiapan_Kehadiran_Satpam_XYZ", "Kinerja_Kesiapan_Kehadiran_Satpam_Komp")]
        for l, h, k_x, k_c in pairs: ipa_data.append({'Indikator': l, 'Harapan': df[h].mean(), 'Kinerja_XYZ': df[k_x].mean(), 'Kinerja_Komp': df_komp[k_c].mean()})
    elif segmen_ipa == "Fasilitas Cabang":
        pairs = [("Akses Eksterior", "Harapan_Akses_Eksterior", "Kinerja_Akses_Eksterior_XYZ", "Kinerja_Akses_Eksterior_Komp"), ("Parkir", "Harapan_Parkir", "Kinerja_Parkir_XYZ", "Kinerja_Parkir_Komp"), ("Banking Hall", "Harapan_BankingHall_Inti", "Kinerja_BankingHall_Inti_XYZ", "Kinerja_BankingHall_Inti_Komp"), ("Fasilitas Ekstra", "Harapan_Fasilitas_Ekstra", "Kinerja_Fasilitas_Ekstra_XYZ", "Kinerja_Fasilitas_Ekstra_Komp"), ("Toilet", "Harapan_Toilet", "Kinerja_Toilet_XYZ", "Kinerja_Toilet_Komp")]
        for l, h, k_x, k_c in pairs: ipa_data.append({'Indikator': l, 'Harapan': df[h].mean(), 'Kinerja_XYZ': df[k_x].mean(), 'Kinerja_Komp': df_komp[k_c].mean()})

    df_ipa = pd.DataFrame(ipa_data)
    
    r3c1, r3c2 = st.columns([1.5, 1])
    with r3c1:
        # Penambahan Titik Kompetitor (Silang Merah) di Matriks IPA
        fig_ipa = go.Figure()
        fig_ipa.add_trace(go.Scatter(x=df_ipa['Kinerja_XYZ'], y=df_ipa['Harapan'], mode='markers+text', text=df_ipa['Indikator'], textposition='top center', name='Bank XYZ', marker=dict(size=14, color='#3B82F6', line=dict(width=2, color='#0F172A'))))
        fig_ipa.add_trace(go.Scatter(x=df_ipa['Kinerja_Komp'], y=df_ipa['Harapan'], mode='markers', name=target_komp, marker=dict(size=12, symbol='x', color='#E11D48', line=dict(width=2))))
        
        # Garis Rata-rata (Berdasarkan XYZ)
        m_k, m_h = df_ipa['Kinerja_XYZ'].mean(), df_ipa['Harapan'].mean()
        fig_ipa.add_vline(x=m_k, line_dash="dash", line_color="#94A3B8")
        fig_ipa.add_hline(y=m_h, line_dash="dash", line_color="#94A3B8")
        
        # Label Kuadran
        fig_ipa.add_annotation(x=min(df_ipa['Kinerja_XYZ']), y=max(df_ipa['Harapan']), text="⚠️ PERBAIKI SEGERA", showarrow=False, font=dict(color="#E11D48", size=11, weight='bold'), bgcolor="rgba(255,255,255,0.7)")
        fig_ipa.add_annotation(x=max(df_ipa['Kinerja_XYZ']), y=max(df_ipa['Harapan']), text="🌟 PERTAHANKAN", showarrow=False, font=dict(color="#10B981", size=11, weight='bold'), bgcolor="rgba(255,255,255,0.7)")
        
        st.plotly_chart(elite_layout(fig_ipa, f"Matriks Kuadran Strategis: {segmen_ipa}"), use_container_width=True)
        
    with r3c2:
        # Gap sekarang dihitung antara Kinerja XYZ vs Kompetitor (Win/Loss)
        df_ipa['Gap_Kompetitif'] = df_ipa['Kinerja_XYZ'] - df_ipa['Kinerja_Komp']
        df_ipa = df_ipa.sort_values(by='Gap_Kompetitif', ascending=True)
        df_ipa['Warna'] = np.where(df_ipa['Gap_Kompetitif'] < 0, '#E11D48', '#10B981')
        
        fig_gap = px.bar(df_ipa, x='Gap_Kompetitif', y='Indikator', orientation='h', text_auto='.2f')
        fig_gap.update_traces(marker_color=df_ipa['Warna'], textposition='outside', marker_cornerradius=3)
        st.plotly_chart(elite_layout(fig_gap, f"Kesenjangan (Gap) terhadap {target_komp}"), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 5: FASILITAS & DIGITALISASI
# ---------------------------------------------------------------------
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        kc_dim = ['Akses_Eksterior', 'Parkir', 'BankingHall_Inti', 'Fasilitas_Ekstra', 'Kemudahan_Eform', 'Toilet']
        df_kc = pd.DataFrame({'Dimensi': kc_dim, 'Bank XYZ': [df[f'Kinerja_{d}_XYZ'].mean() for d in kc_dim], 'Kompetitor': [df_komp[f'Kinerja_{d}_Komp'].mean() for d in kc_dim]}).melt(id_vars='Dimensi', var_name='Bank', value_name='Skor')
        fig_kc = px.bar(df_kc, x='Skor', y='Dimensi', color='Bank', orientation='h', barmode='group', color_discrete_map={'Bank XYZ':'#2563EB', 'Kompetitor':'#F43F5E'})
        fig_kc.update_xaxes(range=[4.5, 6])
        st.plotly_chart(elite_layout(fig_kc, f"Komparasi Kinerja Fasilitas Fisik (vs {target_komp})"), use_container_width=True)
        
    with c2:
        sl_dim = ['Smart_Tab', 'Signage_Table_Santer', 'Pinpad_Frontliner', 'Mesin_CRM_TCR']
        df_sl = pd.DataFrame({'Perangkat Digital': sl_dim, 'Skor': [df[f'Kinerja_SL_{d}_XYZ'].mean() for d in sl_dim]}).sort_values(by='Skor')
        fig_sl = px.bar(df_sl, x='Skor', y='Perangkat Digital', orientation='h', color_discrete_sequence=['#0EA5E9'], text_auto='.2f')
        fig_sl.update_xaxes(range=[4.5, 6])
        fig_sl.update_traces(marker_cornerradius=5)
        st.plotly_chart(elite_layout(fig_sl, "Utilitas Perangkat Smart Layanan XYZ"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if 'D2' in df.columns:
            eform = df['D2'].dropna().value_counts().reset_index()
            fig_eform = px.pie(eform, names='D2', values='count', hole=0.55, color_discrete_sequence=['#8B5CF6', '#D946EF', '#F43F5E'])
            st.plotly_chart(elite_layout(fig_eform, "Adopsi Pembukaan Rekening via E-Form"), use_container_width=True)
    with c4:
        fig_dig_emo = px.scatter(df, x="Outcome_Persepsi_Digitalisasi", y="Outcome_Emosi_Positif_XYZ", color="S2_2", trendline="ols")
        st.plotly_chart(elite_layout(fig_dig_emo, "Dampak Digitalisasi Terhadap Emosi Nasabah"), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 6: VOICE OF CUSTOMER (NLP & VERBATIM)
# ---------------------------------------------------------------------
with tab6:
    st.markdown("<h4 style='color:#0F172A;'>💬 Analisis Sentimen & Teks Alami (NLP)</h4>", unsafe_allow_html=True)
    
    fig_hist_nps = px.histogram(df, x='NPS_Score', nbins=10, color='NPS_Category', color_discrete_map=nps_colors)
    st.plotly_chart(elite_layout(fig_hist_nps, "Volume Penilaian Berdasarkan Skor Absolut (0-10)"), use_container_width=True)
    st.markdown("---")
    
    all_text = " ".join(df['G1B'].dropna().astype(str).tolist()).lower()
    if len(all_text.strip()) > 10:
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("<p style='font-weight:700; color:#0F172A;'>☁️ Pemetaan Topik: Seluruh Nasabah</p>", unsafe_allow_html=True)
            wordcloud_all = WordCloud(width=800, height=450, background_color='#F8FAFC', colormap='Blues', max_words=100).generate(all_text)
            fig_wc_all, ax_all = plt.subplots(figsize=(8, 4.5))
            ax_all.imshow(wordcloud_all, interpolation='bilinear')
            ax_all.axis('off'); fig_wc_all.patch.set_facecolor('#F8FAFC')
            st.pyplot(fig_wc_all)
            
        with r2c2:
            st.markdown("<p style='font-weight:700; color:#E11D48;'>🛑 Titik Kritis: Kata Kunci Nasabah Kecewa (Detractor)</p>", unsafe_allow_html=True)
            detractor_text = " ".join(df[df['NPS_Category'] == 'Detractor']['G1B'].dropna().astype(str).tolist()).lower()
            if len(detractor_text.strip()) > 5:
                wordcloud_det = WordCloud(width=800, height=450, background_color='#F8FAFC', colormap='Reds', max_words=80).generate(detractor_text)
                fig_wc_det, ax_det = plt.subplots(figsize=(8, 4.5))
                ax_det.imshow(wordcloud_det, interpolation='bilinear')
                ax_det.axis('off'); fig_wc_det.patch.set_facecolor('#F8FAFC')
                st.pyplot(fig_wc_det)
            else:
                st.info("💡 Insight: Sentimen negatif sangat rendah pada irisan data ini.")
                
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        stopwords_id = ['yang','untuk','dengan','pada','dari','sebagai','tidak','karena','sangat','lebih','sudah','saya','bank','bisa','dan','di','ke']
        words_clean = [w for w in words if w not in stopwords_id]
        if words_clean:
            word_counts = Counter(words_clean).most_common(10)
            df_words = pd.DataFrame(word_counts, columns=['Terminologi', 'Frekuensi']).sort_values(by='Frekuensi', ascending=True)
            fig_bar_words = px.bar(df_words, x='Frekuensi', y='Terminologi', orientation='h', text='Frekuensi', color_discrete_sequence=['#3B82F6'])
            fig_bar_words = elite_layout(fig_bar_words, "10 Terminologi Paling Sering Diutarakan")
            fig_bar_words.update_traces(textposition='outside', marker_cornerradius=5)
            st.plotly_chart(fig_bar_words, use_container_width=True)

    st.markdown("<h4 style='color:#0F172A; margin-top:20px;'>🔍 Log Penelusuran Verbatim</h4>", unsafe_allow_html=True)
    filter_voC = st.radio("Saring Konteks Percakapan:", ["Fokus Detractor & Passive (Area Perbaikan)", "Lihat Seluruh Komentar"], horizontal=True)
    vocab_df = df if filter_voC == "Lihat Seluruh Komentar" else df[df['NPS_Category'].isin(['Detractor', 'Passive'])]
    st.dataframe(vocab_df[['CABANG', 'S1', 'S2_2', 'NPS_Score', 'NPS_Category', 'G1B']].sort_values(by='NPS_Score'), use_container_width=True, height=250, hide_index=True)

st.markdown("---")
st.caption("🚀 Developed with Streamlit & Plotly | Ultra-Premium Analytics Platform")
