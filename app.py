import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from collections import Counter

# Library untuk Word Cloud
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except ImportError:
    st.error("Library 'wordcloud' atau 'matplotlib' belum terinstal. Tambahkan di requirements.txt")
    st.stop()

# ================= 1. KONFIGURASI & TEMA =================
st.set_page_config(page_title="Hospital Analytics Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS Profesional: Bersih, Minimalis, dan Elegan
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }
    .metric-title {
        color: #64748b;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 28px;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f8fafc;
        border-bottom: 2px solid #0f172a;
    }
    h1, h2, h3, h4 {
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

# Warna Standar Corporate untuk Kategori NPS
nps_colors = {'Promoter': '#22c55e', 'Passive': '#94a3b8', 'Detractor': '#ef4444'}
primary_color = '#0f172a'
secondary_color = '#3b82f6'

# ================= 2. LOAD & PRE-PROCESS DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("Streamlit.csv", sep=";")
    df['Branch'] = df['Branch'].str.replace('Always Healthy Hospital ', '')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H.%M', errors='coerce')
    df['Date'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour
    
    bins = [0, 25, 40, 55, 100]
    labels = ['<25 (Gen Z)', '26-40 (Millennials)', '41-55 (Gen X)', '>55 (Boomers)']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    def categorize_nps(score):
        if score >= 9: return 'Promoter'
        elif score >= 7: return 'Passive'
        else: return 'Detractor'
    df['NPS_Category'] = df['NPS'].apply(categorize_nps)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memproses data: {e}")
    st.stop()

services = ['Registration', 'Doctor Consultation', 'Nurse Service', 'Pharmacy Service', 
            'Laboratory', 'Emergency Response', 'Billing Process', 'Facility Cleanliness', 
            'Staff Friendliness', 'Waiting Time']

# ================= 3. SIDEBAR & FILTER =================
st.sidebar.markdown("### Parameter Filter")
st.sidebar.markdown("---")
min_date, max_date = df['Date'].min(), df['Date'].max()
selected_dates = st.sidebar.date_input("Rentang Waktu", [min_date, max_date], min_value=min_date, max_value=max_date)
start_date, end_date = selected_dates if len(selected_dates) == 2 else (min_date, max_date)

selected_branch = st.sidebar.selectbox("Fasilitas / Cabang", ["Semua Cabang"] + sorted(list(df['Branch'].unique())))
selected_gender = st.sidebar.selectbox("Jenis Kelamin", ["Semua", "Male", "Female"])

df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
if selected_branch != "Semua Cabang":
    df_filtered = df_filtered[df_filtered['Branch'] == selected_branch]
if selected_gender != "Semua":
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]

st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>Hospital Analytics Command Center</h2>", unsafe_allow_html=True)

if df_filtered.empty:
    st.warning("Data tidak tersedia untuk kombinasi filter yang dipilih.")
    st.stop()

# ================= 4. TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary", 
    "Service Deep Dive", 
    "Loyalty & Retention", 
    "Voice of Customer", 
    "Text Analytics"
])

# ----------------- TAB 1: EXECUTIVE SUMMARY -----------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Pasien</div><div class='metric-value'>{len(df_filtered):,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-title'>Skor NPS Rata-rata</div><div class='metric-value'>{df_filtered['NPS'].mean():.1f}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-title'>Indeks Kepuasan (CSI)</div><div class='metric-value'>{df_filtered['CSI'].mean():.2f}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-title'>Persentase Promoter</div><div class='metric-value'>{(len(df_filtered[df_filtered['NPS_Category'] == 'Promoter']) / len(df_filtered)) * 100:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    t1_r1_c1, t1_r1_c2 = st.columns([2, 1])
    with t1_r1_c1:
        trend_data = df_filtered.groupby('Date').agg(Total=('NPS', 'count'), NPS_Avg=('NPS', 'mean')).reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=trend_data['Date'], y=trend_data['Total'], name='Volume Kunjungan', marker_color='#cbd5e1', yaxis='y1'))
        fig_trend.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['NPS_Avg'], name='NPS Rata-rata', mode='lines+markers', line=dict(color=primary_color, width=3), yaxis='y2'))
        fig_trend.update_layout(template="plotly_white", title="Tren Kunjungan vs Skor Kepuasan", yaxis=dict(title='Volume'), yaxis2=dict(title='NPS', overlaying='y', side='right', range=[0, 10]), margin=dict(t=40, b=0))
        st.plotly_chart(fig_trend, use_container_width=True)
    with t1_r1_c2:
        nps_counts = df_filtered['NPS_Category'].value_counts().reset_index()
        fig_donut = px.pie(nps_counts, values='count', names='NPS_Category', hole=0.6, title="Komposisi Segmen Pasien", color='NPS_Category', color_discrete_map=nps_colors)
        fig_donut.update_layout(template="plotly_white", margin=dict(t=40, b=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    t1_r2_c1, t1_r2_c2 = st.columns([1, 1.5])
    with t1_r2_c1:
        branch_nps = df_filtered.groupby('Branch')['NPS'].mean().sort_values(ascending=True).reset_index()
        fig_lead = px.bar(branch_nps, x='NPS', y='Branch', orientation='h', title="Perbandingan Kinerja Fasilitas", color='NPS', color_continuous_scale='Blues')
        fig_lead.update_layout(template="plotly_white", coloraxis_showscale=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_lead, use_container_width=True)
    with t1_r2_c2:
        fig_tree = px.treemap(df_filtered, path=[px.Constant("Distribusi Pasien"), 'Branch', 'Gender', 'NPS_Category'], title="Struktur Demografi & Sentimen", color='NPS', color_continuous_scale='RdYlGn')
        fig_tree.update_layout(template="plotly_white", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_tree, use_container_width=True)

# ----------------- TAB 2: SERVICE DEEP DIVE -----------------
with tab2:
    t2_r1_c1, t2_r1_c2 = st.columns(2)
    with t2_r1_c1:
        service_means = df_filtered[services].mean().reset_index()
        service_means.columns = ['Layanan', 'Skor']
        fig_radar = px.line_polar(service_means, r='Skor', theta='Layanan', line_close=True, title="Kinerja Layanan Menyeluruh")
        fig_radar.update_traces(fill='toself', line_color=secondary_color, fillcolor='rgba(59, 130, 246, 0.2)')
        fig_radar.update_layout(template="plotly_white", polar=dict(radialaxis=dict(range=[1, 5])), margin=dict(t=40, b=0))
        st.plotly_chart(fig_radar, use_container_width=True)
    with t2_r1_c2:
        corr_data = df_filtered[services + ['NPS']].corr()['NPS'].drop('NPS').sort_values(ascending=True)
        fig_driver = px.bar(corr_data, orientation='h', color=corr_data.values, color_continuous_scale='Blues', title="Analisis Dampak Layanan Terhadap NPS")
        fig_driver.update_layout(template="plotly_white", xaxis_title="Koefisien Korelasi", coloraxis_showscale=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_driver, use_container_width=True)

    heatmap_data = df_filtered.groupby('Branch')[services].mean()
    fig_heat = px.imshow(heatmap_data.T, color_continuous_scale='RdBu', text_auto=".1f", title="Matriks Kinerja Kritis: Layanan vs Fasilitas", aspect="auto")
    fig_heat.update_layout(template="plotly_white", margin=dict(t=40, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    t2_r3_c1, t2_r3_c2 = st.columns(2)
    with t2_r3_c1:
        fig_box = px.box(df_filtered, x='Branch', y='Waiting Time', color='Branch', title="Distribusi Waktu Tunggu Operasional")
        fig_box.update_layout(template="plotly_white", showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_box, use_container_width=True)
    with t2_r3_c2:
        doc_nurse = df_filtered.groupby('Branch')[['Doctor Consultation', 'Nurse Service']].mean().reset_index()
        fig_dn = go.Figure(data=[
            go.Bar(name='Konsultasi Dokter', x=doc_nurse['Branch'], y=doc_nurse['Doctor Consultation'], marker_color=primary_color),
            go.Bar(name='Layanan Perawat', x=doc_nurse['Branch'], y=doc_nurse['Nurse Service'], marker_color=secondary_color)
        ])
        fig_dn.update_layout(template="plotly_white", barmode='group', title="Komparasi Kinerja Medis", yaxis=dict(range=[1, 5]), margin=dict(t=40, b=0))
        st.plotly_chart(fig_dn, use_container_width=True)

# ----------------- TAB 3: LOYALTY & RETENTION -----------------
with tab3:
    t3_r1_c1, t3_r1_c2 = st.columns([1.5, 1])
    with t3_r1_c1:
        bubble_data = df_filtered.groupby('Branch').agg(CSI=('CSI', 'mean'), Loyalty=('Loyalty', 'mean'), CES=('CES', 'mean')).reset_index()
        fig_bub = px.scatter(bubble_data, x="CSI", y="Loyalty", size="CES", color="Branch", hover_name="Branch", size_max=35, title="Pemetaan Posisi Strategis (CSI vs Loyalty)")
        fig_bub.add_vline(x=bubble_data['CSI'].mean(), line_dash="dash", line_color="#cbd5e1")
        fig_bub.add_hline(y=bubble_data['Loyalty'].mean(), line_dash="dash", line_color="#cbd5e1")
        fig_bub.update_layout(template="plotly_white", margin=dict(t=40, b=0))
        st.plotly_chart(fig_bub, use_container_width=True)
    with t3_r1_c2:
        macro_corr = df_filtered[['NPS', 'CSI', 'Loyalty', 'CES']].corr()
        fig_mcorr = px.imshow(macro_corr, text_auto=".2f", color_continuous_scale='Blues', title="Korelasi Indikator Makro")
        fig_mcorr.update_layout(template="plotly_white", margin=dict(t=40, b=0))
        st.plotly_chart(fig_mcorr, use_container_width=True)

    t3_r2_c1, t3_r2_c2, t3_r2_c3 = st.columns(3)
    with t3_r2_c1:
        age_loyalty = df_filtered.groupby('Age_Group')['Loyalty'].mean().reset_index()
        fig_al = px.bar(age_loyalty, x='Age_Group', y='Loyalty', title="Tingkat Retensi Berdasarkan Generasi", color='Loyalty', color_continuous_scale='Blues')
        fig_al.update_layout(template="plotly_white", coloraxis_showscale=False, yaxis=dict(range=[1, 5]), margin=dict(t=40, b=0))
        st.plotly_chart(fig_al, use_container_width=True)
    with t3_r2_c2:
        fig_ces = px.histogram(df_filtered, x='CES', color='NPS_Category', barmode='group', title="Distribusi Tingkat Usaha Pasien (CES)", color_discrete_map=nps_colors)
        fig_ces.update_layout(template="plotly_white", margin=dict(t=40, b=0))
        st.plotly_chart(fig_ces, use_container_width=True)
    with t3_r2_c3:
        trend_loyalty = df_filtered.groupby('Date')['Loyalty'].mean().reset_index()
        fig_tl = px.line(trend_loyalty, x='Date', y='Loyalty', markers=True, title="Tren Retensi Operasional")
        fig_tl.update_layout(template="plotly_white", yaxis=dict(range=[1, 5]), margin=dict(t=40, b=0))
        st.plotly_chart(fig_tl, use_container_width=True)

# ----------------- TAB 4: VOICE OF CUSTOMER -----------------
with tab4:
    t4_r1_c1, t4_r1_c2 = st.columns([1, 1])
    with t4_r1_c1:
        hour_nps = df_filtered.groupby('Hour')['NPS'].mean().reset_index()
        fig_hour = px.line(hour_nps, x='Hour', y='NPS', markers=True, title="Fluktuasi Kepuasan Berdasarkan Jam Operasional", line_shape='spline')
        fig_hour.update_layout(template="plotly_white", xaxis=dict(tickmode='linear', tick0=0, dtick=2), margin=dict(t=40, b=0))
        st.plotly_chart(fig_hour, use_container_width=True)
    with t4_r1_c2:
        detractors = df_filtered[df_filtered['NPS_Category'] == 'Detractor']
        if not detractors.empty:
            fig_detage = px.histogram(detractors, x='Age', nbins=15, title="Distribusi Usia Segmen Detractor", color_discrete_sequence=[nps_colors['Detractor']])
            fig_detage.update_layout(template="plotly_white", margin=dict(t=40, b=0))
            st.plotly_chart(fig_detage, use_container_width=True)
        else:
            st.info("Kondisi Optimal: Tidak ada data pasien pada segmen Detractor di rentang waktu ini.")

    st.markdown("#### Log Evaluasi Kualitatif Pasien")
    if 'Improvement_Feedback' in df_filtered.columns:
        cat = st.radio("Filter Sentimen Evaluasi:", ["Seluruh Data", "Khusus Detractor"], horizontal=True)
        vocab_df = df_filtered if cat == "Seluruh Data" else detractors
        st.dataframe(vocab_df[['Datetime', 'Branch', 'Gender', 'Age', 'NPS', 'Improvement_Feedback']].sort_values(by='NPS'), use_container_width=True, height=250, hide_index=True)
    
    fig_stack = px.histogram(df_filtered, y="Branch", color="NPS_Category", orientation='h', title="Proporsi Sentimen Berdasarkan Fasilitas", color_discrete_map=nps_colors)
    fig_stack.update_layout(template="plotly_white", barmode='stack', margin=dict(t=40, b=0))
    st.plotly_chart(fig_stack, use_container_width=True)

# ----------------- TAB 5: TEXT ANALYTICS -----------------
with tab5:
    st.markdown("#### Analisis Frekuensi Kata Kunci Kualitatif")
    
    if 'Improvement_Feedback' in df_filtered.columns:
        all_text = " ".join(df_filtered['Improvement_Feedback'].dropna().astype(str).tolist()).lower()
        
        if len(all_text.strip()) > 0:
            t5_r1_c1, t5_r1_c2 = st.columns(2)
            
            with t5_r1_c1:
                st.markdown("**Pemetaan Kata (Seluruh Pasien)**")
                wordcloud_all = WordCloud(width=800, height=500, background_color='white', colormap='Blues', max_words=100).generate(all_text)
                fig_wc_all, ax_all = plt.subplots(figsize=(8, 5))
                ax_all.imshow(wordcloud_all, interpolation='bilinear')
                ax_all.axis('off')
                st.pyplot(fig_wc_all)
                
            with t5_r1_c2:
                st.markdown("**Pemetaan Kata (Khusus Detractor)**")
                detractor_text = " ".join(df_filtered[df_filtered['NPS_Category'] == 'Detractor']['Improvement_Feedback'].dropna().astype(str).tolist()).lower()
                
                if len(detractor_text.strip()) > 0:
                    wordcloud_det = WordCloud(width=800, height=500, background_color='white', colormap='Reds', max_words=100).generate(detractor_text)
                    fig_wc_det, ax_det = plt.subplots(figsize=(8, 5))
                    ax_det.imshow(wordcloud_det, interpolation='bilinear')
                    ax_det.axis('off')
                    st.pyplot(fig_wc_det)
                else:
                    st.info("Data evaluasi dari segmen Detractor tidak tersedia.")
                    
            st.markdown("---")
            st.markdown("#### Distribusi Frekuensi Istilah Teratas")
            
            words = re.findall(r'\b\w{4,}\b', all_text)
            
            if words:
                word_counts = Counter(words).most_common(10)
                df_words = pd.DataFrame(word_counts, columns=['Istilah', 'Frekuensi'])
                df_words = df_words.sort_values(by='Frekuensi', ascending=True)
                
                fig_bar_words = px.bar(
                    df_words, x='Frekuensi', y='Istilah', orientation='h',
                    color='Frekuensi', color_continuous_scale='Blues', text='Frekuensi'
                )
                fig_bar_words.update_traces(textposition='outside')
                fig_bar_words.update_layout(template="plotly_white", coloraxis_showscale=False, margin=dict(t=40, b=0))
                st.plotly_chart(fig_bar_words, use_container_width=True)
            else:
                st.info("Kuantitas data teks tidak mencukupi untuk ekstraksi frasa.")

        else:
            st.warning("Data kualitatif (teks) tidak ditemukan pada rentang filter saat ini.")
    else:
        st.warning("Variabel teks evaluasi tidak ditemukan dalam struktur data.")
