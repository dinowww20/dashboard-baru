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
# 1. KONFIGURASI & TEMA ELITE (KONTRAST TINGGI)
# =====================================================================
st.set_page_config(page_title="Executive CX Analytics - Bank XYZ", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #F3F4F6; }
    .metric-card {
        background-color: #ffffff; border: 1px solid #D1D5DB; padding: 20px;
        border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;
    }
    .metric-title { color: #000000; font-size: 13px; font-weight: 800; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #1D4ED8; font-size: 32px; font-weight: 900; }
    .stTabs [data-baseweb="tab-list"] { background-color: #ffffff; border-bottom: 2px solid #9CA3AF; padding: 5px 15px 0px 15px; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { color: #000000 !important; border-bottom: 3px solid #1D4ED8 !important; font-weight: 900; }
    h1, h2, h3, h4, p, span, div, label { color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

nps_colors = {'Promoter': '#10B981', 'Passive': '#F59E0B', 'Detractor': '#EF4444'}
color_primary, color_secondary = '#1E3A8A', '#3B82F6'

def elite_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#000000', weight='bold')),
        template="plotly_white", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20, l=20, r=20), font=dict(color='#000000', size=12, weight='bold'),
        hoverlabel=dict(bgcolor="#1E3A8A", font_color="#FFFFFF", font_size=13),
        legend=dict(font=dict(color="#000000"))
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#D1D5DB', zeroline=False, title_font=dict(color='#000000', weight='bold'), tickfont=dict(color='#000000', weight='bold'))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#D1D5DB', zeroline=False, title_font=dict(color='#000000', weight='bold'), tickfont=dict(color='#000000', weight='bold'))
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
        if score >= 9: return 'Promoter'
        elif score >= 7: return 'Passive'
        else: return 'Detractor'
    df['NPS_Category'] = df['NPS_Score'].apply(categorize_nps)
    df['G1B'] = df['G1B'].fillna('')
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}")
    st.stop()

# =====================================================================
# 3. SUPER SIDEBAR (CASCADING MULTI-SELECT FILTERS)
# =====================================================================
st.sidebar.markdown("### 🎛️ Filter Analitik Cerdas")
st.sidebar.caption("Kosongkan pilihan untuk menampilkan semua data.")

# 1. Filter Geografis (Bertingkat)
prov_options = sorted(df_raw['PROV'].dropna().unique().tolist())
selected_prov = st.sidebar.multiselect("📍 Filter Provinsi", prov_options, default=[])

if selected_prov:
    cabang_options = sorted(df_raw[df_raw['PROV'].isin(selected_prov)]['CABANG'].dropna().unique().tolist())
else:
    cabang_options = sorted(df_raw['CABANG'].dropna().unique().tolist())
selected_cabang = st.sidebar.multiselect("🏢 Filter Cabang", cabang_options, default=[])

# 2. Filter Demografi
selected_gender = st.sidebar.multiselect("🚻 Jenis Kelamin", sorted(df_raw['S1'].dropna().unique().tolist()), default=[])
selected_age = st.sidebar.multiselect("🎂 Rentang Usia", sorted(df_raw['S2_2'].dropna().unique().tolist()), default=[])

# 3. Filter Profil Nasabah
selected_tenure = st.sidebar.multiselect("⏳ Lama Menjadi Nasabah", sorted(df_raw['S4'].dropna().unique().tolist()), default=[])
selected_income = st.sidebar.multiselect("💰 Segmen Pendapatan", sorted(df_raw['P6'].dropna().unique().tolist()), default=[])

# LOGIKA FILTERING DATAFRAME
df = df_raw.copy()
if selected_prov: df = df[df['PROV'].isin(selected_prov)]
if selected_cabang: df = df[df['CABANG'].isin(selected_cabang)]
if selected_gender: df = df[df['S1'].isin(selected_gender)]
if selected_age: df = df[df['S2_2'].isin(selected_age)]
if selected_tenure: df = df[df['S4'].isin(selected_tenure)]
if selected_income: df = df[df['P6'].isin(selected_income)]

st.sidebar.markdown("---")
st.sidebar.success(f"📊 Volume Data Aktif: {len(df):,} Nasabah")

st.markdown("<h2 style='text-align: center; margin-bottom: 25px; font-weight: 900;'>🏦 BANK XYZ - EXECUTIVE CX DASHBOARD</h2>", unsafe_allow_html=True)

if df.empty:
    st.warning("Peringatan: Kombinasi filter menghasilkan data kosong. Silakan kurangi filter di sidebar.")
    st.stop()

# =====================================================================
# 4. TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌟 Eksekutif", "👥 Demografi", "🏆 Brand & Kompetitor", 
    "🎯 Frontliner & Matriks IPA", "🏢 Fasilitas & Digital", "💬 Voice of Customer"
])

# ---------------------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><div class='metric-title'>Net Promoter Score (0-10)</div><div class='metric-value'>{df['NPS_Score'].mean():.1f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><div class='metric-title'>Loyalitas Inti (1-6)</div><div class='metric-value'>{df['Outcome_Loyalitas_Inti_XYZ'].mean():.2f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><div class='metric-title'>Emosi Positif</div><div class='metric-value'>{df['Outcome_Emosi_Positif_XYZ'].mean():.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><div class='metric-title'>Emosi Negatif (Skor Terbalik)</div><div class='metric-value' style='color:#EF4444;'>{df['Outcome_Emosi_Negatif_XYZ'].mean():.2f}</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns([1.2, 2])
    with r2c1:
        nps_count = df[df['NPS_Category'] != 'Unknown']['NPS_Category'].value_counts().reset_index()
        fig_nps = px.pie(nps_count, values='count', names='NPS_Category', hole=0.6, color='NPS_Category', color_discrete_map=nps_colors)
        fig_nps = elite_layout(fig_nps, "Distribusi NPS Nasabah")
        fig_nps.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig_nps, use_container_width=True)

    with r2c2:
        top_cabang = df.groupby('CABANG')['Outcome_Loyalitas_Inti_XYZ'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_cabang = px.bar(top_cabang, x='Outcome_Loyalitas_Inti_XYZ', y='CABANG', orientation='h', color_discrete_sequence=[color_secondary])
        fig_cabang = elite_layout(fig_cabang, "Top 10 Cabang dg Loyalitas Tertinggi")
        fig_cabang.update_traces(texttemplate='%{x:.2f}', textposition='outside', textfont_color='#000000')
        fig_cabang.update_xaxes(range=[4, 6.5])
        st.plotly_chart(fig_cabang, use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig_scatter = px.scatter(df, x="Outcome_Emosi_Positif_XYZ", y="Outcome_Loyalitas_Inti_XYZ", color="NPS_Category", color_discrete_map=nps_colors, hover_data=['CABANG'], trendline="ols")
        fig_scatter = elite_layout(fig_scatter, "Korelasi: Emosi Positif vs Loyalitas")
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
        fig_gender = px.pie(df, names='S1', hole=0.4, color_discrete_sequence=[color_primary, '#93C5FD'])
        st.plotly_chart(elite_layout(fig_gender, "Proporsi Gender"), use_container_width=True)
    with r1c2:
        fig_age = px.bar(df['S2_2'].value_counts().reset_index(), x='S2_2', y='count', color_discrete_sequence=[color_secondary])
        st.plotly_chart(elite_layout(fig_age, "Distribusi Kelompok Usia"), use_container_width=True)
    with r1c3:
        fig_tenure = px.bar(df['S4'].value_counts().reset_index(), x='count', y='S4', orientation='h', color_discrete_sequence=['#10B981'])
        st.plotly_chart(elite_layout(fig_tenure, "Lama Menjadi Nasabah"), use_container_width=True)

    r2c1, r2c2 = st.columns([2, 1])
    with r2c1:
        fig_tree = px.treemap(df, path=[px.Constant("Nasabah"), 'PROV', 'CABANG'], color='NPS_Score', color_continuous_scale='RdYlGn')
        fig_tree.update_traces(textfont=dict(color='#000000', size=14, weight='bold'))
        st.plotly_chart(elite_layout(fig_tree, "Peta Hierarki Wilayah (Warna = Rata-rata NPS)"), use_container_width=True)
    with r2c2:
        komp = df['KOMP'].replace(' ', np.nan).dropna().value_counts().head(5).reset_index()
        fig_komp = px.pie(komp, names='KOMP', values='count', hole=0.5, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(elite_layout(fig_komp, "Top Kompetitor Utama"), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: BRAND BENCHMARKING
# ---------------------------------------------------------------------
with tab3:
    c1, c2 = st.columns(2)
    brand_dim = ['Reputasi', 'Jaringan_Fisik', 'Digital_Channel', 'Finansial_Produk', 'Kemudahan_Solusi', 'Koneksi_Emosional']
    with c1:
        fig_radar_brand = go.Figure()
        fig_radar_brand.add_trace(go.Scatterpolar(r=[df[f'Kinerja_{d}_XYZ'].mean() for d in brand_dim], theta=brand_dim, fill='toself', name='Bank XYZ', line_color=color_primary))
        fig_radar_brand.add_trace(go.Scatterpolar(r=[df[f'Kinerja_{d}_Komp'].mean() for d in brand_dim], theta=brand_dim, fill='toself', name='Kompetitor', line_color='#EF4444'))
        fig_radar_brand.update_layout(polar=dict(radialaxis=dict(visible=True, range=[4, 6])))
        st.plotly_chart(elite_layout(fig_radar_brand, "Peta Kekuatan Citra Merek (Brand)"), use_container_width=True)
        
    with c2:
        atm_dim = ['Keamanan', 'Aksesibilitas_Ketersediaan', 'Keandalan_Mesin', 'Fitur_Fungsionalitas', 'Kenyamanan_Fisik']
        fig_radar_atm = go.Figure()
        fig_radar_atm.add_trace(go.Scatterpolar(r=[df[f'Kinerja_ATM_{d}_XYZ'].mean() for d in atm_dim], theta=atm_dim, fill='toself', name='ATM XYZ', line_color=color_primary))
        fig_radar_atm.add_trace(go.Scatterpolar(r=[df[f'Kinerja_ATM_{d}_Komp'].mean() for d in atm_dim], theta=atm_dim, fill='toself', name='ATM Kompetitor', line_color='#EF4444'))
        fig_radar_atm.update_layout(polar=dict(radialaxis=dict(visible=True, range=[4, 6])))
        st.plotly_chart(elite_layout(fig_radar_atm, "Perbandingan Fitur Mesin ATM"), use_container_width=True)

    df_gap_brand = pd.DataFrame({'Dimensi': brand_dim, 'Harapan': [df[f'Harapan_{d}'].mean() for d in brand_dim], 'Kinerja XYZ': [df[f'Kinerja_{d}_XYZ'].mean() for d in brand_dim]}).melt(id_vars='Dimensi', var_name='Metrik', value_name='Skor')
    fig_gap_brand = px.bar(df_gap_brand, x='Dimensi', y='Skor', color='Metrik', barmode='group', color_discrete_map={'Harapan':'#9CA3AF', 'Kinerja XYZ':color_secondary})
    fig_gap_brand.update_traces(texttemplate='%{y:.2f}', textposition='outside', textfont_color='#000000')
    fig_gap_brand.update_yaxes(range=[4.5, 6])
    st.plotly_chart(elite_layout(fig_gap_brand, "Analisis Kesenjangan: Harapan vs Kinerja Brand XYZ"), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 4: FRONTLINER & MATRIKS IPA INTERAKTIF (THE GAME CHANGER)
# ---------------------------------------------------------------------
with tab4:
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(elite_layout(px.bar(x=['XYZ', 'Komp'], y=[df['Overall_Satpam_XYZ'].mean(), df['Overall_Satpam_Komp'].mean()], color=['XYZ','Komp'], title="Overall Satpam", color_discrete_sequence=[color_primary, '#EF4444']).update_layout(showlegend=False), "Overall Satpam"), use_container_width=True)
    c2.plotly_chart(elite_layout(px.bar(x=['XYZ', 'Komp'], y=[df['Overall_Teller_XYZ'].mean(), df['Overall_Teller_Komp'].mean()], color=['XYZ','Komp'], title="Overall Teller", color_discrete_sequence=[color_primary, '#EF4444']).update_layout(showlegend=False), "Overall Teller"), use_container_width=True)
    c3.plotly_chart(elite_layout(px.bar(x=['XYZ', 'Komp'], y=[df['Overall_CS_XYZ'].mean(), df['Overall_CS_Komp'].mean()], color=['XYZ','Komp'], title="Overall CS", color_discrete_sequence=[color_primary, '#EF4444']).update_layout(showlegend=False), "Overall CS"), use_container_width=True)

    st.markdown("---")
    st.markdown("### 🎯 Analisis Kuadran Interaktif (Importance-Performance)")
    
    # FILTER LOKAL INTERAKTIF UNTUK IPA
    segmen_ipa = st.selectbox("Bedah Touchpoint Secara Detail:", ["Customer Service", "Teller", "Satpam", "Fasilitas Cabang"])
    
    ipa_data = []
    if segmen_ipa == "Customer Service":
        dims = ['Antrian_Ketersediaan', 'Kecepatan_Kehandalan', 'Kompetensi_Solusi', 'Edukasi_Digital', 'Sikap_Keramahan', 'Cross_Selling', 'Penampilan_Atribut']
        for d in dims: ipa_data.append({'Indikator': d.replace('_', ' '), 'Harapan': df[f'Harapan_CS_{d}'].mean(), 'Kinerja': df[f'Kinerja_CS_{d}_XYZ'].mean()})
    elif segmen_ipa == "Teller":
        dims = ['Antrian_Ketersediaan', 'Kecepatan_Akurasi', 'Sikap_Keramahan', 'Edukasi_Digital', 'Penampilan_Atribut']
        for d in dims: ipa_data.append({'Indikator': d.replace('_', ' '), 'Harapan': df[f'Harapan_Teller_{d}'].mean(), 'Kinerja': df[f'Kinerja_Teller_{d}_XYZ'].mean()})
    elif segmen_ipa == "Satpam":
        pairs = [("Penampilan", "Harapan_Penampilan_Satpam", "Kinerja_Penampilan_Satpam_XYZ"), ("Sikap Keramahan", "Harapan_Sikap_Keramahan_Satpam", "Kinerja_Sikap_Keramahan_Satpam_XYZ"), ("Kompetensi Tugas", "Harapan_Kompetensi_Tugas_Satpam", "Kinerja_Kompetensi_Tugas_Satpam_XYZ"), ("Kesiapan Kehadiran", "Harapan_Kesiapan_Kehadiran_Satpam", "Kinerja_Kesiapan_Kehadiran_Satpam_XYZ")]
        for l, h, k in pairs: ipa_data.append({'Indikator': l, 'Harapan': df[h].mean(), 'Kinerja': df[k].mean()})
    elif segmen_ipa == "Fasilitas Cabang":
        pairs = [("Akses Eksterior", "Harapan_Akses_Eksterior", "Kinerja_Akses_Eksterior_XYZ"), ("Parkir", "Harapan_Parkir", "Kinerja_Parkir_XYZ"), ("Banking Hall", "Harapan_BankingHall_Inti", "Kinerja_BankingHall_Inti_XYZ"), ("Fasilitas Ekstra", "Harapan_Fasilitas_Ekstra", "Kinerja_Fasilitas_Ekstra_XYZ"), ("Toilet", "Harapan_Toilet", "Kinerja_Toilet_XYZ")]
        for l, h, k in pairs: ipa_data.append({'Indikator': l, 'Harapan': df[h].mean(), 'Kinerja': df[k].mean()})

    df_ipa = pd.DataFrame(ipa_data)
    df_ipa['Gap'] = df_ipa['Kinerja'] - df_ipa['Harapan']
    
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig_ipa = px.scatter(df_ipa, x="Kinerja", y="Harapan", text="Indikator", size_max=20, size=[1]*len(df_ipa), color_discrete_sequence=[color_secondary])
        fig_ipa = elite_layout(fig_ipa, f"Matriks Kuadran (IPA) - {segmen_ipa}")
        fig_ipa.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=2, color='black')))
        m_k, m_h = df_ipa['Kinerja'].mean(), df_ipa['Harapan'].mean()
        fig_ipa.add_vline(x=m_k, line_dash="dash", line_color="#EF4444", line_width=2)
        fig_ipa.add_hline(y=m_h, line_dash="dash", line_color="#EF4444", line_width=2)
        st.plotly_chart(fig_ipa, use_container_width=True)
        
    with r3c2:
        df_ipa = df_ipa.sort_values(by='Gap', ascending=True)
        df_ipa['Warna'] = np.where(df_ipa['Gap'] < 0, '#EF4444', '#10B981')
        fig_gap = px.bar(df_ipa, x='Gap', y='Indikator', orientation='h', text_auto='.2f')
        fig_gap = elite_layout(fig_gap, f"Skor Kesenjangan {segmen_ipa} (Minus = Perlu Perbaikan)")
        fig_gap.update_traces(marker_color=df_ipa['Warna'], textposition='outside', textfont_color='#000000')
        st.plotly_chart(fig_gap, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 5: FASILITAS & DIGITALISASI
# ---------------------------------------------------------------------
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        kc_dim = ['Akses_Eksterior', 'Parkir', 'BankingHall_Inti', 'Fasilitas_Ekstra', 'Kemudahan_Eform', 'Toilet']
        df_kc = pd.DataFrame({'Dimensi': kc_dim, 'Bank XYZ': [df[f'Kinerja_{d}_XYZ'].mean() for d in kc_dim], 'Kompetitor': [df[f'Kinerja_{d}_Komp'].mean() for d in kc_dim]}).melt(id_vars='Dimensi', var_name='Bank', value_name='Skor')
        fig_kc = px.bar(df_kc, x='Dimensi', y='Skor', color='Bank', barmode='group', color_discrete_map={'Bank XYZ':color_primary, 'Kompetitor':'#EF4444'})
        fig_kc.update_yaxes(range=[4.5, 6])
        st.plotly_chart(elite_layout(fig_kc, "Komparasi Fasilitas Fisik Cabang"), use_container_width=True)
        
    with c2:
        sl_dim = ['Smart_Tab', 'Signage_Table_Santer', 'Pinpad_Frontliner', 'Mesin_CRM_TCR']
        df_sl = pd.DataFrame({'Perangkat Digital': sl_dim, 'Skor': [df[f'Kinerja_SL_{d}_XYZ'].mean() for d in sl_dim]})
        fig_sl = px.bar(df_sl, x='Perangkat Digital', y='Skor', color_discrete_sequence=['#06B6D4'], text_auto='.2f')
        fig_sl.update_yaxes(range=[4.5, 6])
        st.plotly_chart(elite_layout(fig_sl, "Kinerja Perangkat Smart Layanan XYZ"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if 'D2' in df.columns:
            eform = df['D2'].dropna().value_counts().reset_index()
            fig_eform = px.pie(eform, names='D2', values='count', hole=0.5, color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(elite_layout(fig_eform, "Adopsi Pembukaan Rekening E-Form"), use_container_width=True)
    with c4:
        fig_dig_emo = px.scatter(df, x="Outcome_Persepsi_Digitalisasi", y="Outcome_Emosi_Positif_XYZ", color="S2_2", trendline="ols")
        st.plotly_chart(elite_layout(fig_dig_emo, "Korelasi Digitalisasi vs Emosi Positif per Usia"), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 6: VOICE OF CUSTOMER & TEXT ANALYTICS
# ---------------------------------------------------------------------
with tab6:
    st.markdown("#### Analisis Semantik & Keluhan Nasabah (Verbatim)")
    
    fig_hist_nps = px.histogram(df, x='NPS_Score', nbins=10, color='NPS_Category', color_discrete_map=nps_colors)
    st.plotly_chart(elite_layout(fig_hist_nps, "Distribusi Frekuensi Skor NPS (0-10)"), use_container_width=True)
    st.markdown("---")
    
    all_text = " ".join(df['G1B'].dropna().astype(str).tolist()).lower()
    if len(all_text.strip()) > 10:
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("**☁️ Word Cloud (Seluruh Alasan Nasabah)**")
            wordcloud_all = WordCloud(width=800, height=400, background_color='#ffffff', colormap='Blues', max_words=100).generate(all_text)
            fig_wc_all, ax_all = plt.subplots(figsize=(8, 4))
            ax_all.imshow(wordcloud_all, interpolation='bilinear')
            ax_all.axis('off'); fig_wc_all.patch.set_facecolor('#ffffff')
            st.pyplot(fig_wc_all)
            
        with r2c2:
            st.markdown("**🛑 Word Cloud (Nasabah Kecewa / Detractor)**")
            detractor_text = " ".join(df[df['NPS_Category'] == 'Detractor']['G1B'].dropna().astype(str).tolist()).lower()
            if len(detractor_text.strip()) > 5:
                wordcloud_det = WordCloud(width=800, height=400, background_color='#ffffff', colormap='Reds', max_words=80).generate(detractor_text)
                fig_wc_det, ax_det = plt.subplots(figsize=(8, 4))
                ax_det.imshow(wordcloud_det, interpolation='bilinear')
                ax_det.axis('off'); fig_wc_det.patch.set_facecolor('#ffffff')
                st.pyplot(fig_wc_det)
            else:
                st.info("Tidak ada cukup data verbatim dari kelompok Detractor.")
                
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
        stopwords_id = ['yang','untuk','dengan','pada','dari','sebagai','tidak','karena','sangat','lebih','sudah','saya','bank','bisa','dan','di','ke']
        words_clean = [w for w in words if w not in stopwords_id]
        if words_clean:
            word_counts = Counter(words_clean).most_common(10)
            df_words = pd.DataFrame(word_counts, columns=['Terminologi', 'Frekuensi']).sort_values(by='Frekuensi', ascending=True)
            fig_bar_words = px.bar(df_words, x='Frekuensi', y='Terminologi', orientation='h', text='Frekuensi', color_discrete_sequence=[color_secondary])
            fig_bar_words = elite_layout(fig_bar_words, "Top 10 Kata Terbanyak Disebut (Verbatim)")
            fig_bar_words.update_traces(textposition='outside', textfont_color='#000000')
            st.plotly_chart(fig_bar_words, use_container_width=True)

    st.markdown("#### Tabel Penelusuran Keluhan Spesifik")
    filter_voC = st.radio("Saring Verbatim:", ["Hanya Detractor & Passive", "Semua Nasabah"], horizontal=True)
    vocab_df = df if filter_voC == "Semua Nasabah" else df[df['NPS_Category'].isin(['Detractor', 'Passive'])]
    st.dataframe(vocab_df[['CABANG', 'S1', 'S2_2', 'NPS_Score', 'NPS_Category', 'G1B']].sort_values(by='NPS_Score'), use_container_width=True, height=250, hide_index=True)

st.markdown("---")
st.caption("🚀 Developed with Streamlit & Plotly | Executive Intelligence Analytics Bank XYZ")
