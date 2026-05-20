import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Konfigurasi Halaman & Tema Premium
st.set_page_config(page_title="Dashboard Analitik RS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card {background-color: #ffffff; border-left: 5px solid #1E3A8A; padding: 15px; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .metric-title {color: #6c757d; font-size: 14px; font-weight: bold; text-transform: uppercase;}
    .metric-value {color: #1E3A8A; font-size: 28px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# 2. Fungsi Membaca & Pre-processing Data
@st.cache_data
def load_data():
    data = pd.read_csv("Streamlit.csv", sep=";")
    
    # Membersihkan nama cabang
    data['Branch'] = data['Branch'].str.replace('Always Healthy Hospital ', '')
    
    # Mengubah format Datetime menjadi tipe data tanggal yang benar
    data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%Y %H.%M', errors='coerce')
    data['Date'] = data['Datetime'].dt.date
    
    # Membuat Kategori NPS
    def categorize_nps(score):
        if score >= 9: return 'Promoter (Loyal)'
        elif score >= 7: return 'Passive (Netral)'
        else: return 'Detractor (Kecewa)'
    data['NPS_Category'] = data['NPS'].apply(categorize_nps)
    
    return data

try:
    df = load_data()
except Exception as e:
    st.error("Gagal memproses data. Pastikan format Datetime sesuai.")
    st.stop()

# 3. Sidebar Filter yang Komprehensif
st.sidebar.markdown("## 🔍 Panel Filter Data")
st.sidebar.markdown("---")

# Filter Rentang Waktu
min_date = df['Date'].min()
max_date = df['Date'].max()
selected_dates = st.sidebar.date_input("Pilih Rentang Waktu:", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = min_date, max_date

# Filter Cabang
all_branches = ["Semua Cabang"] + sorted(list(df['Branch'].unique()))
selected_branch = st.sidebar.selectbox("Pilih Cabang:", all_branches)

# Filter Gender & Umur
selected_gender = st.sidebar.selectbox("Gender:", ["Semua", "Male", "Female"])
age_range = st.sidebar.slider("Rentang Umur:", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))

# Menerapkan Filter ke Dataframe
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
if selected_branch != "Semua Cabang":
    df_filtered = df_filtered[df_filtered['Branch'] == selected_branch]
if selected_gender != "Semua":
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
df_filtered = df_filtered[(df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1])]

# 4. Header & Metrik Utama
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>📈 Dashboard Analitik & Performa Layanan Rumah Sakit</h1>", unsafe_allow_html=True)
st.markdown("---")

if df_filtered.empty:
    st.warning("Tidak ada data pada rentang filter yang dipilih.")
    st.stop()

# Menghitung Metrik Lanjutan
total_resp = len(df_filtered)
avg_nps = df_filtered['NPS'].mean()
avg_csi = df_filtered['CSI'].mean()
promoter_pct = (len(df_filtered[df_filtered['NPS_Category'] == 'Promoter (Loyal)']) / total_resp) * 100

col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Responden</div><div class='metric-value'>{total_resp:,}</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><div class='metric-title'>Rata-rata NPS (0-10)</div><div class='metric-value'>{avg_nps:.1f}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><div class='metric-title'>Indeks Kepuasan / CSI (1-5)</div><div class='metric-value'>{avg_csi:.2f}</div></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><div class='metric-title'>Persentase Promoters</div><div class='metric-value'>{promoter_pct:.1f}%</div></div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# 5. VISUALISASI MENDALAM (DEEP DIVE ANALYTICS)

st.markdown("### 1️⃣ Analisis Tren & Segmen Pasien")
row1_col1, row1_col2 = st.columns([2, 1])

with row1_col1:
    # Time Series (Tren Waktu)
    trend_df = df_filtered.groupby('Date')[['NPS', 'CSI']].mean().reset_index()
    fig_trend = px.line(trend_df, x='Date', y=['NPS', 'CSI'], markers=True, 
                        title="Pergerakan Rata-rata Skor NPS & CSI Harian",
                        labels={"value": "Skor Rata-rata", "Date": "Tanggal", "variable": "Metrik"})
    fig_trend.update_layout(legend_title="", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("🔍 *Analisis: Identifikasi apakah terjadi lonjakan/penurunan kepuasan pada tanggal tertentu (misal saat libur panjang).*")

with row1_col2:
    # NPS Composition (Pie)
    nps_counts = df_filtered['NPS_Category'].value_counts().reset_index()
    nps_counts.columns = ['Kategori', 'Jumlah']
    fig_nps = px.pie(nps_counts, values='Jumlah', names='Kategori', 
                     color='Kategori',
                     color_discrete_map={'Promoter (Loyal)':'#28a745', 'Passive (Netral)':'#ffc107', 'Detractor (Kecewa)':'#dc3545'},
                     hole=0.5, title="Komposisi Pelanggan (NPS)")
    st.plotly_chart(fig_nps, use_container_width=True)
    st.caption("🔍 *Analisis: Fokus pada zona merah (Detractor) yang berpotensi menyebarkan Word-of-Mouth negatif.*")

st.markdown("---")

st.markdown("### 2️⃣ Analisis Diagnostik Kinerja Layanan")
row2_col1, row2_col2 = st.columns([1, 1])

# Mengelompokkan kolom layanan
services = ['Registration', 'Doctor Consultation', 'Nurse Service', 'Pharmacy Service', 
            'Laboratory', 'Emergency Response', 'Billing Process', 'Facility Cleanliness', 
            'Staff Friendliness', 'Waiting Time']

with row2_col1:
    # Radar Chart - Overview Kinerja
    service_means = df_filtered[services].mean().reset_index()
    service_means.columns = ['Layanan', 'Skor Rata-rata']
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=service_means['Skor Rata-rata'],
        theta=service_means['Layanan'],
        fill='toself',
        name='Skor Layanan',
        line_color='#007A87'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
        title="Jaring Kinerja: Kekuatan & Kelemahan Departemen"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("🔍 *Analisis: Area yang mengerucut ke dalam menandakan departemen dengan layanan di bawah ekspektasi.*")

with row2_col2:
    # Correlation/Driver Analysis - INSIGHT PALING PENTING
    st.markdown("##### 🔑 Layanan Apa yang Paling Mempengaruhi Kepuasan (NPS)?")
    
    # Hitung korelasi masing-masing layanan dengan NPS keseluruhan
    corr_data = df_filtered[services + ['NPS']].corr()['NPS'].drop('NPS').sort_values(ascending=True)
    corr_df = corr_data.reset_index()
    corr_df.columns = ['Layanan', 'Dampak thd NPS (Korelasi)']
    
    fig_driver = px.bar(corr_df, x='Dampak thd NPS (Korelasi)', y='Layanan', orientation='h',
                        color='Dampak thd NPS (Korelasi)', color_continuous_scale="Viridis",
                        title="Driver Analysis: Prioritas Perbaikan")
    fig_driver.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_driver, use_container_width=True)
    st.caption("🔍 *Analisis: Layanan dengan batang terpanjang adalah kunci utama kepuasan pasien. Jika ini jelek, NPS pasti anjlok! Perbaiki layanan ini lebih dulu.*")

st.markdown("---")

# 6. Tabel Ekstraksi Feedback
st.markdown("### 💬 Suara Pasien (Voice of Customer)")
if 'Improvement_Feedback' in df_filtered.columns:
    # Menampilkan yang benar-benar memberi kritik (Detractors) agar mudah dievaluasi
    detractors_feedback = df_filtered[df_filtered['NPS_Category'] == 'Detractor (Kecewa)']
    if not detractors_feedback.empty:
        st.error(f"⚠️ Ditemukan {len(detractors_feedback)} pasien kecewa (Detractor). Berikut masukan mereka:")
        st.dataframe(detractors_feedback[['Date', 'Branch', 'NPS', 'Improvement_Feedback']], use_container_width=True, hide_index=True)
    else:
        st.success("Tidak ada pasien kategori Detractor pada filter ini!")
        st.dataframe(df_filtered[['Date', 'Branch', 'NPS', 'Improvement_Feedback']].head(10), use_container_width=True, hide_index=True)
