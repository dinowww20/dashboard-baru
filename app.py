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
# 1. KONFIGURASI & TEMA ELITE
# =====================================================================
st.set_page_config(page_title="Executive CX Analytics - Bank XYZ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #F3F4F6; }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #E5E7EB;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-title { color: #000000; font-size: 14px; font-weight: 800; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #1D4ED8; font-size: 36px; font-weight: 900; }
    .metric-delta { color: #10B981; font-size: 14px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #ffffff; border-bottom: 2px solid #9CA3AF; padding: 5px 15px 0px 15px; border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] { color: #000000 !important; border-bottom: 3px solid #1D4ED8 !important; font-weight: 900; }
    h1, h2, h3, h4, p, span, div, label { color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

nps_colors = {'Promoter': '#10B981', 'Passive': '#F59E0B', 'Detractor': '#EF4444'}
color_primary = '#1E3A8A'   
color_secondary = '#3B82F6' 

def elite_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#000000', weight='bold')),
        template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=20, l=20, r=20),
        font=dict(color='#000000', size=13, weight='bold'),
        hoverlabel=dict(bgcolor="#1E3A8A", font_color="#FFFFFF", font_size=13),
        legend=dict(font=dict(color="#000000"))
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB', zeroline=False, title_font=dict(color='#000000', weight='bold'), tickfont=dict(color='#000000', weight='bold'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#E5E7EB', zeroline=False, title_font=dict(color='#000000', weight='bold'), tickfont=dict(color='#000000', weight='bold'))
    return fig

# =====================================================================
# 2. LOAD & PRE-PROCESS DATA
# =====================================================================
@st.cache_data
def load_data():
    df = pd.read_excel('Dataset_Dashboard_Perfect.xlsx', engine='openpyxl')
    
    # Cleansing NPS Score (Ekstrak angka dari teks "10 PASTI MEREKOMENDASIKAN")
    df['NPS_Score'] = df['G1A'].astype(str).str.extract(r'(\d+)').astype(float)
    
    def categorize_nps(score):
        if pd.isna(score): return 'Unknown'
        if score >= 9: return 'Promoter'
        elif score >= 7: return 'Passive'
        else: return 'Detractor'
        
    df['NPS_Category'] = df['NPS_Score'].apply(categorize_nps)
    
    # Isi NaN pada teks dengan string kosong
    df['G1B'] = df['G1B'].fillna('')
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# =====================================================================
# 3. SIDEBAR (GLOBAL SLICER)
# =====================================================================
st.sidebar.markdown("### 🔍 Parameter Analisis")

selected_prov = st.sidebar.selectbox("Pilih Provinsi", ["Semua Provinsi"] + sorted(df_raw['PROV'].dropna().unique().tolist()))
selected_cabang = st.sidebar.selectbox("Pilih Cabang", ["Semua Cabang"] + sorted(df_raw['CABANG'].dropna().unique().tolist()))
selected_gender = st.sidebar.selectbox("Pilih Gender", ["Semua Gender"] + sorted(df_raw['S1'].dropna().unique().tolist()))
selected_age = st.sidebar.selectbox("Kelompok Usia", ["Semua Usia"] + sorted(df_raw['S2_2'].dropna().unique().tolist()))

# Filter Logic
df = df_raw.copy()
if selected_prov != "Semua Provinsi": df = df[df['PROV'] == selected_prov]
if selected_cabang != "Semua Cabang": df = df[df['CABANG'] == selected_cabang]
if selected_gender != "Semua Gender": df = df[df['S1'] == selected_gender]
if selected_age != "Semua Usia": df = df[df['S2_2'] == selected_age]

st.sidebar.markdown("---")
st.sidebar.success(f"📊 Menampilkan: {len(df):,} Nasabah")

st.markdown("<h2 style='text-align: center; margin-bottom: 30px; font-weight: 900;'>🏦 Bank XYZ - Executive CX Dashboard</h2>", unsafe_allow_html=True)

if df.empty:
    st.warning("Data kosong dengan kombinasi filter ini.")
    st.stop()

# =====================================================================
# 4. TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌟 Executive Summary", "👥 Profil & Demografi", "🏆 Brand Benchmarking", 
    "🎯 Frontliner & IPA Matrix", "🏢 Fasilitas & Digital", "💬 Voice of Customer"
])

# ---------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------------------
with tab1:
    # --- VISUAL 1-4: METRIC CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    nps_avg = df['NPS_Score'].mean()
    loyalty_avg = df['Outcome_Loyalitas_Inti_XYZ'].mean()
    emo_pos = df['Outcome_Emosi_Positif_XYZ'].mean()
    emo_neg = df['Outcome_Emosi_Negatif_XYZ'].mean()
    
    c1.markdown(f"<div class='metric-card'><div class='metric-title'>Net Promoter Score (0-10)</div><div class='metric-value'>{nps_avg:.1f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-title'>Loyalitas Inti (1-6)</div><div class='metric-value'>{loyalty_avg:.2f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-title'>Emosi Positif</div><div class='metric-value'>{emo_pos:.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-title'>Emosi Negatif (Makin kecil makin baik)</div><div class='metric-value' style='color:#EF4444;'>{emo_neg:.2f}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns([1.5, 2])
    
    # --- VISUAL 5: DONUT NPS ---
    with r2c1:
        nps_count = df[df['NPS_Category'] != 'Unknown']['NPS_Category'].value_counts().reset_index()
        fig_nps = px.pie(nps_count, values='count', names='NPS_Category', hole=0.6, color='NPS_Category', color_discrete_map=nps_colors)
        fig_nps = elite_layout(fig_nps, "Distribusi NPS Nasabah")
        fig_nps.update_traces(textposition='outside', textinfo='percent+label', textfont=dict(color='#000000', size=14, weight='bold'))
        fig_nps.update_layout(showlegend=False)
        st.plotly_chart(fig_nps, use_container_width=True)

    # --- VISUAL 6: BAR CABANG TERBAIK ---
    with r2c2:
        top_cabang = df.groupby('CABANG')[['Outcome_Loyalitas_Inti_XYZ']].mean().sort_values(by='Outcome_Loyalitas_Inti_XYZ', ascending=False).head(10).reset_index()
        fig_cabang = px.bar(top_cabang, x='Outcome_Loyalitas_Inti_XYZ', y='CABANG', orientation='h', color_discrete_sequence=[color_secondary])
        fig_cabang = elite_layout(fig_cabang, "Top 10 Cabang dg Loyalitas Tertinggi")
        fig_cabang.update_traces(texttemplate='%{x:.2f}', textposition='outside', textfont_color='#000000')
        fig_cabang.update_xaxes(range=[4, 6.5])
        st.plotly_chart(fig_cabang, use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    # --- VISUAL 7: SCATTER EMOSI VS LOYALITAS ---
    with r3c1:
        fig_scatter = px.scatter(df, x="Outcome_Emosi_Positif_XYZ", y="Outcome_Loyalitas_Inti_XYZ", color="NPS_Category", color_discrete_map=nps_colors, hover_data=['CABANG'], trendline="ols")
        fig_scatter = elite_layout(fig_scatter, "Korelasi: Emosi Positif vs Loyalitas")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    # --- VISUAL 8: KORELASI MAKRO ---
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
    
    # --- VISUAL 9: GENDER ---
    with r1c1:
        fig_gender = px.pie(df, names='S1', title="Gender (S1)", hole=0.4, color_discrete_sequence=[color_primary, '#93C5FD'])
        fig_gender = elite_layout(fig_gender, "Proporsi Gender")
        st.plotly_chart(fig_gender, use_container_width=True)
        
    # --- VISUAL 10: UMUR ---
    with r1c2:
        age_count = df['S2_2'].value_counts().reset_index()
        fig_age = px.bar(age_count, x='S2_2', y='count', color_discrete_sequence=[color_secondary])
        fig_age = elite_layout(fig_age, "Distribusi Kelompok Usia (S2_2)")
        st.plotly_chart(fig_age, use_container_width=True)
        
    # --- VISUAL 11: LAMA JADI NASABAH ---
    with r1c3:
        tenure = df['S4'].value_counts().reset_index()
        fig_tenure = px.bar(tenure, x='count', y='S4', orientation='h', color_discrete_sequence=['#10B981'])
        fig_tenure = elite_layout(fig_tenure, "Lama Menjadi Nasabah (S4)")
        st.plotly_chart(fig_tenure, use_container_width=True)

    r2c1, r2c2 = st.columns([2, 1])
    # --- VISUAL 12: TREEMAP GEOGRAFIS ---
    with r2c1:
        fig_tree = px.treemap(df, path=[px.Constant("Nasabah XYZ"), 'PROV', 'CABANG'], color='NPS_Score', color_continuous_scale='RdYlGn')
        fig_tree = elite_layout(fig_tree, "Peta Hierarki Wilayah (Warna = NPS)")
        fig_tree.update_traces(textfont=dict(color='#000000', size=14, weight='bold'))
        st.plotly_chart(fig_tree, use_container_width=True)
        
    # --- VISUAL 13: TOP KOMPETITOR ---
    with r2c2:
        komp = df['KOMP'].replace(' ', np.nan).dropna().value_counts().head(5).reset_index()
        fig_komp = px.pie(komp, names='KOMP', values='count', hole=0.5, color_discrete_sequence=px.colors.qualitative.Bold)
        fig_komp = elite_layout(fig_komp, "Top Kompetitor Utama")
        st.plotly_chart(fig_komp, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: BRAND BENCHMARKING
# ---------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)
    # --- VISUAL 14: RADAR BRAND ---
    with c1:
        brand_dim = ['Reputasi', 'Jaringan_Fisik', 'Digital_Channel', 'Finansial_Produk', 'Kemudahan_Solusi', 'Koneksi_Emosional']
        xyz_brand = [df[f'Kinerja_{d}_XYZ'].mean() for d in brand_dim]
        komp_brand = [df[f'Kinerja_{d}_Komp'].mean() for d in brand_dim]
        
        fig_radar_brand = go.Figure()
        fig_radar_brand.add_trace(go.Scatterpolar(r=xyz_brand, theta=brand_dim, fill='toself', name='Bank XYZ', line_color=color_primary))
        fig_radar_brand.add_trace(go.Scatterpolar(r=komp_brand, theta=brand_dim, fill='toself', name='Kompetitor', line_color='#EF4444'))
        fig_radar_brand = elite_layout(fig_radar_brand, "Peta Kekuatan Citra Merek (Brand)")
        fig_radar_brand.update_layout(polar=dict(radialaxis=dict(visible=True, range=[4, 6])))
        st.plotly_chart(fig_radar_brand, use_container_width=True)
        
    # --- VISUAL 15: RADAR ATM ---
    with c2:
        atm_dim = ['Keamanan', 'Aksesibilitas_Ketersediaan', 'Keandalan_Mesin', 'Fitur_Fungsionalitas', 'Kenyamanan_Fisik']
        xyz_atm = [df[f'Kinerja_ATM_{d}_XYZ'].mean() for d in atm_dim]
        komp_atm = [df[f'Kinerja_ATM_{d}_Komp'].mean() for d in atm_dim]
        
        fig_radar_atm = go.Figure()
        fig_radar_atm.add_trace(go.Scatterpolar(r=xyz_atm, theta=atm_dim, fill='toself', name='ATM XYZ', line_color=color_primary))
        fig_radar_atm.add_trace(go.Scatterpolar(r=komp_atm, theta=atm_dim, fill='toself', name='ATM Kompetitor', line_color='#EF4444'))
        fig_radar_atm = elite_layout(fig_radar_atm, "Perbandingan Fitur Mesin ATM")
        fig_radar_atm.update_layout(polar=dict(radialaxis=dict(visible=True, range=[4, 6])))
        st.plotly_chart(fig_radar_atm, use_container_width=True)

    # --- VISUAL 16: GAP HARAPAN VS KINERJA BRAND ---
    harapan_brand = [df[f'Harapan_{d}'].mean() for d in brand_dim]
    df_gap_brand = pd.DataFrame({'Dimensi': brand_dim, 'Harapan': harapan_brand, 'Kinerja XYZ': xyz_brand}).melt(id_vars='Dimensi', var_name='Metrik', value_name='Skor')
    fig_gap_brand = px.bar(df_gap_brand, x='Dimensi', y='Skor', color='Metrik', barmode='group', color_discrete_map={'Harapan':'#9CA3AF', 'Kinerja XYZ':color_secondary})
    fig_gap_brand = elite_layout(fig_gap_brand, "Analisis Kesenjangan: Harapan vs Kinerja Brand XYZ")
    fig_gap_brand.update_traces(texttemplate='%{y:.2f}', textposition='outside', textfont_color='#000000')
    fig_gap_brand.update_yaxes(range=[4.5, 6])
    st.plotly_chart(fig_gap_brand, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: FRONTLINER & IPA MATRIX
# ---------------------------------------------------------------------
with tab4:
    # --- VISUAL 17, 18, 19: OVERALL SCORES ---
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(elite_layout(px.bar(x=['XYZ', 'Komp'], y=[df['Overall_Satpam_XYZ'].mean(), df['Overall_Satpam_Komp'].mean()], color=['XYZ','Komp'], title="Overall Satpam", color_discrete_sequence=[color_primary, '#EF4444']), "Overall Satpam").update_layout(showlegend=False), use_container_width=True)
    c2.plotly_chart(elite_layout(px.bar(x=['XYZ', 'Komp'], y=[df['Overall_Teller_XYZ'].mean(), df['Overall_Teller_Komp'].mean()], color=['XYZ','Komp'], title="Overall Teller", color_discrete_sequence=[color_primary, '#EF4444']), "Overall Teller").update_layout(showlegend=False), use_container_width=True)
    c3.plotly_chart(elite_layout(px.bar(x=['XYZ', 'Komp'], y=[df['Overall_CS_XYZ'].mean(), df['Overall_CS_Komp'].mean()], color=['XYZ','Komp'], title="Overall CS", color_discrete_sequence=[color_primary, '#EF4444']), "Overall CS").update_layout(showlegend=False), use_container_width=True)

    st.markdown("---")
    
    r2c1, r2c2 = st.columns([1.5, 1])
    # --- VISUAL 20: IMPORTANCE-PERFORMANCE ANALYSIS (IPA) MATRIKS CS ---
    with r2c1:
        cs_dim = ['Antrian_Ketersediaan', 'Kecepatan_Kehandalan', 'Kompetensi_Solusi', 'Edukasi_Digital', 'Sikap_Keramahan', 'Cross_Selling', 'Penampilan_Atribut']
        cs_harapan = [df[f'Harapan_CS_{d}'].mean() for d in cs_dim]
        cs_kinerja = [df[f'Kinerja_CS_{d}_XYZ'].mean() for d in cs_dim]
        df_ipa = pd.DataFrame({'Dimensi': cs_dim, 'Harapan (Importance)': cs_harapan, 'Kinerja (Performance)': cs_kinerja})
        
        fig_ipa = px.scatter(df_ipa, x="Kinerja (Performance)", y="Harapan (Importance)", text="Dimensi", size_max=20, size=[1]*len(df_ipa), color_discrete_sequence=[color_secondary])
        fig_ipa = elite_layout(fig_ipa, "Matriks Kuadran IPA - Customer Service XYZ")
        fig_ipa.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=2, color='black')))
        
        # Garis Salib Rata-rata
        mean_kinerja = np.mean(cs_kinerja)
        mean_harapan = np.mean(cs_harapan)
        fig_ipa.add_vline(x=mean_kinerja, line_dash="dash", line_color="#EF4444", line_width=2)
        fig_ipa.add_hline(y=mean_harapan, line_dash="dash", line_color="#EF4444", line_width=2)
        
        # Anotasi Kuadran
        fig_ipa.add_annotation(x=min(cs_kinerja), y=max(cs_harapan), text="Kritikal (Perbaiki Segera)", showarrow=False, font=dict(color="red", size=14))
        fig_ipa.add_annotation(x=max(cs_kinerja), y=max(cs_harapan), text="Pertahankan Prestasi", showarrow=False, font=dict(color="green", size=14))
        st.plotly_chart(fig_ipa, use_container_width=True)

    # --- VISUAL 21: HEATMAP TELLER vs CABANG ---
    with r2c2:
        top5_cabang = df['CABANG'].value_counts().head(10).index
        heat_df = df[df['CABANG'].isin(top5_cabang)].groupby('CABANG')[['Overall_Teller_XYZ', 'Overall_CS_XYZ', 'Overall_Satpam_XYZ']].mean()
        fig_heat = px.imshow(heat_df.T, color_continuous_scale='Blues', text_auto=".2f", aspect="auto")
        fig_heat = elite_layout(fig_heat, "Kinerja Frontliner Cabang Terpadat")
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- VISUAL 22, 23: DETAIL CA & SATPAM (Bar Horizontal) ---
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        ca_dim = ['Ketersediaan_Kecepatan', 'Ketelitian_Kompetensi', 'Sikap_Keramahan', 'Proaktif_Promosi', 'Penampilan']
        df_ca = pd.DataFrame({'Dimensi': ca_dim, 'Kinerja': [df[f'Kinerja_CA_{d}_XYZ'].mean() for d in ca_dim]}).sort_values('Kinerja')
        fig_ca = elite_layout(px.bar(df_ca, x='Kinerja', y='Dimensi', orientation='h', color_discrete_sequence=['#F59E0B'], text_auto='.2f'), "Detail Kinerja Customer Advisor (XYZ)")
        fig_ca.update_xaxes(range=[4.5, 6])
        st.plotly_chart(fig_ca, use_container_width=True)
        
    with r3c2:
        sc_dim = ['Penampilan_Satpam', 'Sikap_Keramahan_Satpam', 'Kompetensi_Tugas_Satpam', 'Kesiapan_Kehadiran_Satpam']
        df_sc = pd.DataFrame({'Dimensi': sc_dim, 'Kinerja': [df[f'Kinerja_{d}_XYZ'].mean() for d in sc_dim]}).sort_values('Kinerja')
        fig_sc = elite_layout(px.bar(df_sc, x='Kinerja', y='Dimensi', orientation='h', color_discrete_sequence=['#10B981'], text_auto='.2f'), "Detail Kinerja Satpam (XYZ)")
        fig_sc.update_xaxes(range=[4.5, 6])
        st.plotly_chart(fig_sc, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 5: FASILITAS & DIGITALISASI
# ---------------------------------------------------------------------
with tab5:
    c1, c2 = st.columns(2)
    # --- VISUAL 24: FASILITAS CABANG BAR ---
    with c1:
        kc_dim = ['Akses_Eksterior', 'Parkir', 'BankingHall_Inti', 'Fasilitas_Ekstra', 'Kemudahan_Eform', 'Toilet']
        df_kc = pd.DataFrame({'Dimensi': kc_dim, 'Bank XYZ': [df[f'Kinerja_{d}_XYZ'].mean() for d in kc_dim], 'Kompetitor': [df[f'Kinerja_{d}_Komp'].mean() for d in kc_dim]}).melt(id_vars='Dimensi', var_name='Bank', value_name='Skor')
        fig_kc = px.bar(df_kc, x='Dimensi', y='Skor', color='Bank', barmode='group', color_discrete_map={'Bank XYZ':color_primary, 'Kompetitor':'#EF4444'})
        fig_kc = elite_layout(fig_kc, "Komparasi Fasilitas Fisik Cabang")
        fig_kc.update_yaxes(range=[4.5, 6])
        st.plotly_chart(fig_kc, use_container_width=True)
        
    # --- VISUAL 25: SMART LAYANAN XYZ ---
    with c2:
        sl_dim = ['Smart_Tab', 'Signage_Table_Santer', 'Pinpad_Frontliner', 'Mesin_CRM_TCR']
        df_sl = pd.DataFrame({'Perangkat Digital': sl_dim, 'Skor': [df[f'Kinerja_SL_{d}_XYZ'].mean() for d in sl_dim]})
        fig_sl = elite_layout(px.bar(df_sl, x='Perangkat Digital', y='Skor', color_discrete_sequence=['#06B6D4'], text_auto='.2f'), "Kinerja Perangkat Smart Layanan XYZ")
        fig_sl.update_yaxes(range=[4.5, 6])
        st.plotly_chart(fig_sl, use_container_width=True)

    c3, c4 = st.columns(2)
    # --- VISUAL 26: E-FORM ADOPSI ---
    with c3:
        if 'D2' in df.columns:
            eform = df['D2'].dropna().value_counts().reset_index()
            fig_eform = px.pie(eform, names='D2', values='count', hole=0.5, color_discrete_sequence=px.colors.qualitative.Set2)
            fig_eform = elite_layout(fig_eform, "Adopsi Pembukaan Rekening E-Form (D2)")
            st.plotly_chart(fig_eform, use_container_width=True)
            
    # --- VISUAL 27: SCATTER DIGITAL VS EMOSI ---
    with c4:
        fig_dig_emo = px.scatter(df, x="Outcome_Persepsi_Digitalisasi", y="Outcome_Emosi_Positif_XYZ", color="S2_2", trendline="ols")
        fig_dig_emo = elite_layout(fig_dig_emo, "Korelasi Digitalisasi vs Emosi Positif per Usia")
        st.plotly_chart(fig_dig_emo, use_container_width=True)


# ---------------------------------------------------------------------
# TAB 6: VOICE OF CUSTOMER & TEXT ANALYTICS
# ---------------------------------------------------------------------
with tab6:
    st.markdown("#### Analisis Semantik & Keluhan Nasabah (Verbatim)")
    
    # --- VISUAL 28: HISTOGRAM NPS ---
    fig_hist_nps = px.histogram(df, x='NPS_Score', nbins=10, color='NPS_Category', color_discrete_map=nps_colors)
    fig_hist_nps = elite_layout(fig_hist_nps, "Distribusi Frekuensi Skor NPS (0-10)")
    st.plotly_chart(fig_hist_nps, use_container_width=True)
    
    st.markdown("---")
    
    # NLP Processing
    all_text = " ".join(df['G1B'].dropna().astype(str).tolist()).lower()
    
    if len(all_text.strip()) > 10:
        r2c1, r2c2 = st.columns(2)
        
        # --- VISUAL 29: WORD CLOUD ALL ---
        with r2c1:
            st.markdown("**☁️ Word Cloud (Seluruh Alasan Nasabah)**")
            wordcloud_all = WordCloud(width=800, height=400, background_color='#ffffff', colormap='Blues', max_words=100).generate(all_text)
            fig_wc_all, ax_all = plt.subplots(figsize=(8, 4))
            ax_all.imshow(wordcloud_all, interpolation='bilinear')
            ax_all.axis('off')
            fig_wc_all.patch.set_facecolor('#ffffff')
            st.pyplot(fig_wc_all)
            
        # --- VISUAL 30: WORD CLOUD DETRACTOR ---
        with r2c2:
            st.markdown("**🛑 Word Cloud (Nasabah Kecewa / Detractor)**")
            detractor_text = " ".join(df[df['NPS_Category'] == 'Detractor']['G1B'].dropna().astype(str).tolist()).lower()
            if len(detractor_text.strip()) > 5:
                wordcloud_det = WordCloud(width=800, height=400, background_color='#ffffff', colormap='Reds', max_words=80).generate(detractor_text)
                fig_wc_det, ax_det = plt.subplots(figsize=(8, 4))
                ax_det.imshow(wordcloud_det, interpolation='bilinear')
                ax_det.axis('off')
                fig_wc_det.patch.set_facecolor('#ffffff')
                st.pyplot(fig_wc_det)
            else:
                st.info("Tidak ada cukup data verbatim dari kelompok Detractor.")
                
        # --- VISUAL 31: TOP WORDS BAR ---
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        # Simple stopword removal bypass
        stopwords_id = ['yang','untuk','dengan','pada','dari','sebagai','tidak','karena','sangat','lebih','sudah','saya','bank','bisa','dan','di','ke']
        words_clean = [w for w in words if w not in stopwords_id]
        
        if words_clean:
            word_counts = Counter(words_clean).most_common(10)
            df_words = pd.DataFrame(word_counts, columns=['Terminologi', 'Frekuensi Aktual']).sort_values(by='Frekuensi Aktual', ascending=True)
            fig_bar_words = elite_layout(px.bar(df_words, x='Frekuensi Aktual', y='Terminologi', orientation='h', text='Frekuensi Aktual', color_discrete_sequence=[color_secondary]), "Top 10 Kata Terbanyak Disebut (Verbatim NPS)")
            fig_bar_words.update_traces(textposition='outside', textfont_color='#000000')
            st.plotly_chart(fig_bar_words, use_container_width=True)

    # --- VISUAL 32: RAW DATA VIEWER ---
    st.markdown("#### Tabel Penelusuran Keluhan Spesifik")
    filter_voC = st.radio("Saring Verbatim:", ["Hanya Detractor & Passive", "Semua Nasabah"], horizontal=True)
    vocab_df = df if filter_voC == "Semua Nasabah" else df[df['NPS_Category'].isin(['Detractor', 'Passive'])]
    st.dataframe(vocab_df[['CABANG', 'S1', 'S2_2', 'NPS_Score', 'NPS_Category', 'G1B']].sort_values(by='NPS_Score'), use_container_width=True, height=250, hide_index=True)

st.markdown("---")
st.caption("🚀 Developed with Streamlit, Plotly, and Advanced NLP Analytics | CX Strategy Bank XYZ")
