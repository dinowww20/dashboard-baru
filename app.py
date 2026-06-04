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
    st.error("Library 'wordcloud' atau 'matplotlib' belum terinstal. Tambahkan 'statsmodels' juga di requirements.txt!")
    st.stop()

# =====================================================================
# 1. KONFIGURASI & THEME MANAGEMENT (SAAS LIGHT PREMIUM)
# =====================================================================
st.set_page_config(page_title="Executive CX Analytics - Bank XYZ", layout="wide", initial_sidebar_state="expanded")

# CSS Khusus: Memperbaiki Kontras Sidebar & Mencegah Cropping Content
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC !important; }
    
    /* Sidebar Styling: Force High Contrast */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Metric Cards: Glassmorphism Effect */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        text-align: center;
        transition: 0.3s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); border-color: #3B82F6; }
    .metric-title { color: #64748B; font-size: 12px; font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #1E293B; font-size: 32px; font-weight: 800; }
    .metric-delta { font-size: 13px; font-weight: 600; }

    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #475569 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E40AF !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# Palet Warna Profesional
COLORS = {
    'XYZ': '#2563EB',      # Royal Blue
    'Komp': '#F43F5E',     # Rose Red
    'Neutral': '#94A3B8',  # Slate Grey
    'Success': '#10B981'   # Emerald Green
}

# Fungsi Layout Plotly (Anti-Cropping & Aesthetic)
def apply_pro_layout(fig, title="", height=450):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#0F172A', weight='bold'), x=0.05),
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=12, color='#334155'),
        margin=dict(l=80, r=50, t=100, b=80), # Margin Luas agar label tidak terpotong
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9', tickfont=dict(weight='bold'))
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', tickfont=dict(weight='bold'))
    return fig

# =====================================================================
# 2. DATA ENGINE
# =====================================================================
@st.cache_data
def load_and_clean():
    df = pd.read_excel('Dataset_Dashboard_Perfect.xlsx', engine='openpyxl')
    # Clean NPS
    df['NPS_Score'] = df['G1A'].astype(str).str.extract(r'(\d+)').astype(float)
    def cat_nps(s):
        if pd.isna(s): return 'Unknown'
        return 'Promoter' if s >= 9 else ('Passive' if s >= 7 else 'Detractor')
    df['NPS_Category'] = df['NPS_Score'].apply(cat_nps)
    return df

try:
    df_raw = load_and_clean()
except Exception as e:
    st.error(f"Gagal Load Data: {e}")
    st.stop()

# =====================================================================
# 3. SIDEBAR FILTERS (DYNAMIC & CASCADING)
# =====================================================================
with st.sidebar:
    st.markdown("### 📊 Filter Strategis")
    
    # 1. Filter Regional
    prov_sel = st.multiselect("Provinsi", sorted(df_raw['PROV'].dropna().unique()))
    df_step = df_raw[df_raw['PROV'].isin(prov_sel)] if prov_sel else df_raw
    
    cabang_sel = st.multiselect("Cabang", sorted(df_step['CABANG'].dropna().unique()))
    df_step = df_step[df_step['CABANG'].isin(cabang_sel)] if cabang_sel else df_step
    
    # 2. Filter Kompetitor (Sangat Penting untuk Benchmarking)
    komp_options = sorted(df_raw['KOMP'].dropna().unique())
    selected_bench_bank = st.selectbox("Pilih Kompetitor Utama untuk Benchmarking:", komp_options)
    
    st.markdown("---")
    # 3. Demografi
    gender_sel = st.multiselect("Gender", df_raw['S1'].unique())
    age_sel = st.multiselect("Rentang Usia", sorted(df_raw['S2_2'].dropna().unique()))
    
# Final Filtered DataFrame
df = df_step.copy()
if gender_sel: df = df[df['S1'].isin(gender_sel)]
if age_sel: df = df[df['S2_2'].isin(age_sel)]

# Subset Data Kompetitor Spesifik (Untuk perbandingan IPA & Radar)
df_komp_target = df_raw[df_raw['KOMP'] == selected_bench_bank]

# =====================================================================
# 4. DASHBOARD LAYOUT
# =====================================================================
st.markdown(f"<h1 style='text-align: center; color: #0F172A;'>🏦 Bank XYZ Intelligence vs {selected_bench_bank}</h1>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💎 Executive View", "🆚 Benchmarking", "🎯 IPA Strategy", "☁️ Voice of Customer"])

# --- TAB 1: EXECUTIVE ---
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    # Kalkulasi Dinamis
    nps_val = df['NPS_Score'].mean()
    nps_prev = df_raw['NPS_Score'].mean() # Sebagai benchmark global
    
    m1.markdown(f"<div class='metric-card'><div class='metric-title'>Net Promoter Score</div><div class='metric-value' style='color:{COLORS['XYZ']}'>{nps_val:.1f}</div><div class='metric-delta' style='color:{COLORS['Success']}'>Global Avg: {nps_prev:.1f}</div></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='metric-card'><div class='metric-title'>Loyalitas XYZ</div><div class='metric-value'>{df['Outcome_Loyalitas_Inti_XYZ'].mean():.2f}</div><div class='metric-delta'>Skala 1-6</div></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='metric-card'><div class='metric-title'>Indeks Emosi Positif</div><div class='metric-value'>{df['Outcome_Emosi_Positif_XYZ'].mean():.2f}</div><div class='metric-delta'>Target: >5.0</div></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='metric-card'><div class='metric-title'>Digital Adoption</div><div class='metric-value'>{df['Outcome_Persepsi_Digitalisasi'].mean():.2f}</div><div class='metric-delta'>Persepsi Cabang</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1.5, 2.5])
    with c1:
        # Donut NPS kontras
        fig_nps = px.pie(df, names='NPS_Category', hole=0.6, color='NPS_Category', color_discrete_map={'Promoter':'#10B981','Passive':'#F59E0B','Detractor':'#EF4444'})
        st.plotly_chart(apply_pro_layout(fig_nps, "Sentimen Nasabah"), use_container_width=True)
    with c2:
        # Heatmap Korelasi Makro
        corr_cols = ['Outcome_Loyalitas_Inti_XYZ', 'Outcome_Emosi_Positif_XYZ', 'Outcome_Persepsi_Digitalisasi', 'Overall_CS_XYZ', 'Overall_Teller_XYZ']
        fig_corr = px.imshow(df[corr_cols].corr(), text_auto=".2f", color_continuous_scale='RdBu_r', labels=dict(color="Korelasi"))
        st.plotly_chart(apply_pro_layout(fig_corr, "Faktor Pendorong Loyalitas (Correlation Matrix)"), use_container_width=True)

# --- TAB 2: BENCHMARKING (RINGKAS & INSIGHTFUL) ---
with tab2:
    st.markdown(f"### 🚀 Performa Bank XYZ vs {selected_bench_bank}")
    
    # 1. MERGED OVERALL SCORES (Gambar 2 diperbaiki)
    # Menampilkan semua touchpoint dalam satu bar chart grup
    tp_labels = ['Satpam', 'Teller', 'Customer Service', 'ATM']
    tp_xyz = [df['Overall_Satpam_XYZ'].mean(), df['Overall_Teller_XYZ'].mean(), df['Overall_CS_XYZ'].mean(), df['Overall_ATM_XYZ'].mean()]
    tp_komp = [df_komp_target['Overall_Satpam_Komp'].mean(), df_komp_target['Overall_Teller_Komp'].mean(), df_komp_target['Overall_CS_Komp'].mean(), df_komp_target['Overall_ATM_Komp'].mean()]
    
    fig_overall = go.Figure(data=[
        go.Bar(name='Bank XYZ', x=tp_labels, y=tp_xyz, marker_color=COLORS['XYZ'], text=np.round(tp_xyz, 2), textposition='auto'),
        go.Bar(name=selected_bench_bank, x=tp_labels, y=tp_komp, marker_color=COLORS['Komp'], text=np.round(tp_komp, 2), textposition='auto')
    ])
    fig_overall.update_layout(barmode='group', yaxis_range=[0, 6.5])
    st.plotly_chart(apply_pro_layout(fig_overall, "Touchpoint Performance Scoreboard"), use_container_width=True)
    
    # 2. RADAR BRAND POWER (Gambar 1 diperbaiki)
    c1, c2 = st.columns(2)
    brand_dim = ['Reputasi', 'Jaringan_Fisik', 'Digital_Channel', 'Finansial_Produk', 'Kemudahan_Solusi', 'Koneksi_Emosional']
    with c1:
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[df[f'Kinerja_{d}_XYZ'].mean() for d in brand_dim], theta=brand_dim, fill='toself', name='XYZ', line_color=COLORS['XYZ']))
        fig_radar.add_trace(go.Scatterpolar(r=[df_komp_target[f'Kinerja_{d}_Komp'].mean() for d in brand_dim], theta=brand_dim, fill='toself', name=selected_bench_bank, line_color=COLORS['Komp']))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[4, 6])), margin=dict(l=100, r=100, t=100, b=100))
        st.plotly_chart(apply_pro_layout(fig_radar, "Radar Citra Merek (Brand Power)"), use_container_width=True)
    with c2:
        # Analisis Gap vs Kompetitor Terpilih
        gap_vals = np.array(tp_xyz) - np.array(tp_komp)
        fig_gap = px.bar(x=tp_labels, y=gap_vals, color=gap_vals > 0, 
                         color_discrete_map={True: COLORS['Success'], False: COLORS['Komp']},
                         labels={'x':'Layanan','y':'Gap Score'})
        st.plotly_chart(apply_pro_layout(fig_gap, f"Win/Loss Gap vs {selected_bench_bank}"), use_container_width=True)

# --- TAB 3: IPA STRATEGY (ULTIMATE INTERACTIVE) ---
with tab3:
    st.markdown("### 🎯 Competitive IPA Matrix")
    st.caption("Membandingkan Kepentingan (Harapan) dengan Kinerja XYZ dan Kompetitor Pilihan.")
    
    target_tp = st.selectbox("Pilih Dimensi Layanan:", ["Customer Service", "Teller", "Satpam", "Fasilitas"])
    
    # Data Mapping Dinamis
    ipa_map = {
        "Customer Service": ['Antrian_Ketersediaan', 'Kecepatan_Kehandalan', 'Kompetensi_Solusi', 'Sikap_Keramahan', 'Penampilan_Atribut'],
        "Teller": ['Antrian_Ketersediaan', 'Kecepatan_Akurasi', 'Sikap_Keramahan', 'Penampilan_Atribut'],
        "Satpam": ['Penampilan_Satpam', 'Sikap_Keramahan_Satpam', 'Kompetensi_Tugas_Satpam', 'Kesiapan_Kehadiran_Satpam'],
        "Fasilitas": ['Akses_Eksterior', 'Parkir', 'BankingHall_Inti', 'Toilet']
    }
    
    # Menghitung Data IPA
    plot_data = []
    prefix = "CS" if target_tp == "Customer Service" else ("Teller" if target_tp == "Teller" else "")
    
    for dim in ipa_map[target_tp]:
        if target_tp in ["Customer Service", "Teller"]:
            h = df[f'Harapan_{prefix}_{dim}'].mean()
            k_xyz = df[f'Kinerja_{prefix}_{dim}_XYZ'].mean()
            k_komp = df_komp_target[f'Kinerja_{prefix}_{dim}_Komp'].mean()
        else:
            h = df[f'Harapan_{dim}'].mean()
            k_xyz = df[f'Kinerja_{dim}_XYZ'].mean()
            k_komp = df_komp_target[f'Kinerja_{dim}_Komp'].mean()
            
        plot_data.append({'Atribut': dim.replace('_', ' '), 'Importance': h, 'XYZ': k_xyz, 'Competitor': k_komp})
    
    df_ipa = pd.DataFrame(plot_data)
    
    # Plotly Scatter IPA
    fig_ipa = go.Figure()
    # Tambahkan XYZ Points
    fig_ipa.add_trace(go.Scatter(x=df_ipa['XYZ'], y=df_ipa['Importance'], mode='markers+text', 
                                 text=df_ipa['Atribut'], name='Bank XYZ', marker=dict(size=15, color=COLORS['XYZ'])))
    # Tambahkan Kompetitor Points (sebagai 'X' atau simbol lain untuk kontras)
    fig_ipa.add_trace(go.Scatter(x=df_ipa['Competitor'], y=df_ipa['Importance'], mode='markers', 
                                 name=selected_bench_bank, marker=dict(size=12, symbol='x', color=COLORS['Komp'])))
    
    # Draw Quadrant Lines (XYZ Based)
    avg_perf = df_ipa['XYZ'].mean()
    avg_imp = df_ipa['Importance'].mean()
    fig_ipa.add_vline(x=avg_perf, line_dash="dash", line_color="#94A3B8")
    fig_ipa.add_hline(y=avg_imp, line_dash="dash", line_color="#94A3B8")
    
    # Quadrant Labels
    fig_ipa.add_annotation(x=avg_perf-0.2, y=avg_imp+0.2, text="URGENT (Kinerja Rendah, Harapan Tinggi)", showarrow=False, font=dict(color="red"))
    fig_ipa.add_annotation(x=avg_perf+0.2, y=avg_imp+0.2, text="STRENGTH (Pertahankan)", showarrow=False, font=dict(color="green"))
    
    st.plotly_chart(apply_pro_layout(fig_ipa, f"Competitive IPA Matrix: XYZ vs {selected_bench_bank}"), use_container_width=True)
    
    # Insight Section
    st.info(f"💡 **Key Insight:** Atribut yang titik Birunya (XYZ) berada di sebelah kiri titik Merah ({selected_bench_bank}) adalah area di mana Anda tertinggal secara kompetitif.")

# --- TAB 4: VOC ---
with tab4:
    st.markdown("### 💬 Voice of Customer & NLP")
    c1, c2 = st.columns(2)
    # WordCloud
    with c1:
        text = " ".join(df['G1B'].astype(str))
        wc = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
        st.pyplot(fig_wc)
    with c2:
        # Analisis Sentimen Sederhana / Top Kata
        words = re.findall(r'\b\w{4,}\b', text.lower())
        stop = ['yang','dengan','pada','untuk','bank','nasabah','sudah','saya']
        top_words = Counter([w for w in words if w not in stop]).most_common(10)
        df_top = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])
        st.plotly_chart(apply_pro_layout(px.bar(df_top, x='Frekuensi', y='Kata', orientation='h'), "Top Keyword Feedback"), use_container_width=True)

st.markdown("---")
st.caption("🚀 V4 Ultimate CX Dashboard | Built for Bank XYZ Executive Strategy")
