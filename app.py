import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. Konfigurasi Halaman 
st.set_page_config(page_title="Advanced Analytics RS", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk mempercantik Metrics
st.markdown("""
    <style>
    .metric-card {background-color: #ffffff; border-top: 4px solid #1E3A8A; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;}
    .metric-title {color: #6c757d; font-size: 13px; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;}
    .metric-value {color: #1E3A8A; font-size: 32px; font-weight: 900;}
    </style>
""", unsafe_allow_html=True)

# 2. Fungsi Load & Pre-process Data
@st.cache_data
def load_data():
    df = pd.read_csv("Streamlit.csv", sep=";")
    df['Branch'] = df['Branch'].str.replace('Always Healthy Hospital ', '')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H.%M', errors='coerce')
    df['Date'] = df['Datetime'].dt.date
    
    # Kategori NPS
    def categorize_nps(score):
        if score >= 9: return 'Promoter'
        elif score >= 7: return 'Passive'
        else: return 'Detractor'
    df['NPS_Category'] = df['NPS'].apply(categorize_nps)
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memproses data. Error: {e}")
    st.stop()

# Daftar Layanan
services = ['Registration', 'Doctor Consultation', 'Nurse Service', 'Pharmacy Service', 
            'Laboratory', 'Emergency Response', 'Billing Process', 'Facility Cleanliness', 
            'Staff Friendliness', 'Waiting Time']

# 3. Sidebar Filter Utama (Berlaku untuk semua Tab)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.markdown("## ⚙️ Filter Global")

min_date, max_date = df['Date'].min(), df['Date'].max()
selected_dates = st.sidebar.date_input("Rentang Tanggal:", [min_date, max_date], min_value=min_date, max_value=max_date)
start_date, end_date = selected_dates if len(selected_dates) == 2 else (min_date, max_date)

all_branches = ["Semua Cabang"] + sorted(list(df['Branch'].unique()))
selected_branch = st.sidebar.selectbox("Cabang:", all_branches)

selected_gender = st.sidebar.selectbox("Gender:", ["Semua", "Male", "Female"])
age_range = st.sidebar.slider("Rentang Umur:", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))

# Terapkan Filter
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
if selected_branch != "Semua Cabang":
    df_filtered = df_filtered[df_filtered['Branch'] == selected_branch]
if selected_gender != "Semua":
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
df_filtered = df_filtered[(df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1])]

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏥 Hospital Command Center Analytics</h1>", unsafe_allow_html=True)
st.markdown("---")

if df_filtered.empty:
    st.warning("Tidak ada data pada filter yang dipilih.")
    st.stop()

# 4. MEMBUAT TABS
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Summary", "🔍 Analisis Layanan (Deep Dive)", "❤️ Loyalitas & Retensi", "💬 Voice of Customer"])

# ================= TAB 1: EXECUTIVE SUMMARY =================
with tab1:
    # Metrik Baris Atas
    total_resp = len(df_filtered)
    avg_nps = df_filtered['NPS'].mean()
    avg_csi = df_filtered['CSI'].mean()
    promoter_pct = (len(df_filtered[df_filtered['NPS_Category'] == 'Promoter']) / total_resp) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Pasien</div><div class='metric-value'>{total_resp:,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-title'>Skor NPS (0-10)</div><div class='metric-value'>{avg_nps:.1f}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-title'>Indeks CSI (1-5)</div><div class='metric-value'>{avg_csi:.2f}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-title'>Promoter (Loyal)</div><div class='metric-value'>{promoter_pct:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    t1_col1, t1_col2 = st.columns([2, 1.5])
    with t1_col1:
        # Tren Harian Bar & Line (Volume vs Kepuasan)
        trend_data = df_filtered.groupby('Date').agg(Total=('NPS', 'count'), NPS_Avg=('NPS', 'mean')).reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=trend_data['Date'], y=trend_data['Total'], name='Volume Pasien', marker_color='#a0c4ff', yaxis='y1'))
        fig_trend.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['NPS_Avg'], name='Rata-rata NPS', mode='lines+markers', line=dict(color='#1E3A8A', width=3), yaxis='y2'))
        fig_trend.update_layout(
            title="Tren Volume Kunjungan vs Skor Kepuasan (NPS)",
            yaxis=dict(title='Volume Pasien', side='left'),
            yaxis2=dict(title='Skor NPS', side='right', overlaying='y', range=[0, 10]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with t1_col2:
        # Sunburst Chart: Demografi -> Gender -> NPS
        fig_sunburst = px.sunburst(
            df_filtered, path=['Branch', 'Gender', 'NPS_Category'], 
            title="Komposisi Demografi & Sentimen",
            color='NPS_Category',
            color_discrete_map={'Promoter':'#28a745', 'Passive':'#ffc107', 'Detractor':'#dc3545'}
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)

# ================= TAB 2: ANALISIS LAYANAN =================
with tab2:
    st.markdown("### 🗺️ Peta Kinerja Cabang vs Unit Layanan")
    st.info("💡 **Cara baca heatmap:** Warna merah/gelap menunjukkan layanan dengan nilai kritis di cabang tertentu. Area ini butuh intervensi manajemen segera!")
    
    # Heatmap Rata-rata per layanan per cabang
    heatmap_data = df_filtered.groupby('Branch')[services].mean()
    fig_heat = px.imshow(
        heatmap_data.T, 
        color_continuous_scale='RdYlGn',
        aspect="auto",
        text_auto=".1f",
        title="Heatmap Kinerja Layanan (Skala 1-5)"
    )
    fig_heat.update_layout(xaxis_title="Cabang Rumah Sakit", yaxis_title="Unit Layanan")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    t2_col1, t2_col2 = st.columns(2)
    
    with t2_col1:
        st.markdown("#### 🎯 Driver Analysis (Apa Pendorong Utama NPS?)")
        corr_data = df_filtered[services + ['NPS']].corr()['NPS'].drop('NPS').sort_values(ascending=True)
        fig_driver = px.bar(corr_data, orientation='h', color=corr_data.values, color_continuous_scale='Blues')
        fig_driver.update_layout(xaxis_title="Kekuatan Korelasi dengan NPS", yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig_driver, use_container_width=True)
        
    with t2_col2:
        st.markdown("#### 📦 Variasi Waktu Tunggu (Waiting Time)")
        fig_box = px.box(df_filtered, x='Branch', y='Waiting Time', color='Branch', 
                         title="Sebaran Penilaian Waktu Tunggu per Cabang")
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ================= TAB 3: LOYALITAS & RETENSI =================
with tab3:
    t3_col1, t3_col2 = st.columns([1.5, 1])
    
    with t3_col1:
        st.markdown("### 🔮 Matriks Loyalitas vs Kepuasan (CSI)")
        st.caption("Ukuran gelembung menunjukkan skor *Customer Effort Score* (CES). Semakin besar, semakin mudah layanan diakses.")
        # Bubble Chart
        bubble_data = df_filtered.groupby('Branch').agg(
            NPS=('NPS', 'mean'), CSI=('CSI', 'mean'), 
            Loyalty=('Loyalty', 'mean'), CES=('CES', 'mean'),
            Total_Pasien=('NPS', 'count')
        ).reset_index()
        
        fig_bubble = px.scatter(
            bubble_data, x="CSI", y="Loyalty", size="CES", color="Branch",
            hover_name="Branch", size_max=40,
            labels={"CSI": "Skor Kepuasan (CSI)", "Loyalty": "Tingkat Loyalitas"},
            title="Posisi Cabang Berdasarkan Kepuasan & Retensi"
        )
        # Tambah garis kuadran rata-rata
        fig_bubble.add_vline(x=bubble_data['CSI'].mean(), line_width=2, line_dash="dash", line_color="red")
        fig_bubble.add_hline(y=bubble_data['Loyalty'].mean(), line_width=2, line_dash="dash", line_color="red")
        st.plotly_chart(fig_bubble, use_container_width=True)

    with t3_col2:
        st.markdown("### 🚧 Analisis Customer Effort Score (CES)")
        st.caption("Seberapa besar *usaha* yang dikeluarkan pasien?")
        fig_ces = px.histogram(df_filtered, x="CES", color="NPS_Category", barmode="group",
                               color_discrete_map={'Promoter':'#28a745', 'Passive':'#ffc107', 'Detractor':'#dc3545'})
        fig_ces.update_layout(xaxis_title="Skor Upaya Pelanggan (CES)", yaxis_title="Jumlah Pasien")
        st.plotly_chart(fig_ces, use_container_width=True)

# ================= TAB 4: VOICE OF CUSTOMER =================
with tab4:
    st.markdown("### 📢 Analisis Komentar & Umpan Balik Pasien")
    
    if 'Improvement_Feedback' in df_filtered.columns:
        # Filter interaktif khusus di tab ini
        feedback_category = st.radio("Saring berdasarkan kategori pasien:", ["Semua Pasien", "Hanya Detractor (Kecewa)", "Hanya Promoter (Puas)"], horizontal=True)
        
        if feedback_category == "Hanya Detractor (Kecewa)":
            f_df = df_filtered[df_filtered['NPS_Category'] == 'Detractor']
        elif feedback_category == "Hanya Promoter (Puas)":
            f_df = df_filtered[df_filtered['NPS_Category'] == 'Promoter']
        else:
            f_df = df_filtered
            
        f_df = f_df[['Date', 'Branch', 'NPS', 'CSI', 'Improvement_Feedback']].sort_values(by='NPS')
        f_df.columns = ['Tanggal', 'Cabang', 'Skor NPS', 'Skor CSI', 'Komentar Pasien']
        
        st.dataframe(f_df, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("Data komentar tidak tersedia.")
