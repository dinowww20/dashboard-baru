import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Konfigurasi Halaman & Tema Premium
st.set_page_config(page_title="Dashboard Rumah Sakit", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk mempercantik tampilan kartu (Cards) dan layout
st.markdown("""
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #007A87;
    }
    .metric-label {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Fungsi Membaca Data
@st.cache_data
def load_data():
    data = pd.read_csv("Streamlit.csv", sep=";")
    # Bersihkan nama cabang dari prefix jika ada
    data['Branch'] = data['Branch'].str.replace('Always Healthy Hospital ', '')
    return data

try:
    df = load_data()
except Exception as e:
    st.error("Data Streamlit.csv belum diupload dengan benar atau format terganggu!")
    st.stop()

# 3. Sidebar Filter yang Interaktif & Ramah Pengguna
st.sidebar.markdown("## 🔍 Panel Filter Data")
st.sidebar.markdown("---")

# Filter Cabang Rumah Sakit
all_branches = ["Semua Cabang"] + sorted(list(df['Branch'].unique()))
selected_branch = st.sidebar.selectbox("Pilih Cabang Rumah Sakit:", all_branches)

# Filter Gender
selected_gender = st.sidebar.radio("Pilih Jenis Kelamin Pasien:", ["Semua", "Male", "Female"], horizontal=True)

# Filter Umur dengan Slider
min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
selected_age = st.sidebar.slider("Rentang Umur Pasien:", min_age, max_age, (min_age, max_age))

# Mengaplikasikan filter ke Dataframe
df_filtered = df.copy()
if selected_branch != "Semua Cabang":
    df_filtered = df_filtered[df_filtered['Branch'] == selected_branch]
if selected_gender != "Semua":
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
df_filtered = df_filtered[(df_filtered['Age'] >= selected_age[0]) & (df_filtered['Age'] <= selected_age[1])]


# 4. Header Utama Dashboard
st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom: 30px;'>📊 Dashboard Analisis Kepuasan & Kinerja Layanan Rumah Sakit</h1>", unsafe_allow_html=True)

# 5. Baris Pertama: Ringkasan Indikator Utama (KPI Metrics)
total_respondents = len(df_filtered)
avg_nps = df_filtered['NPS'].mean() if total_respondents > 0 else 0
avg_csi = df_filtered['CSI'].mean() if total_respondents > 0 else 0
avg_waiting = df_filtered['Waiting Time'].mean() if total_respondents > 0 else 0

m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>👥 TOTAL RESPONDEN</div><div class='metric-value'>{total_respondents:,}</div></div>", unsafe_allow_html=True)
with m_col2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>📈 RATA-RATA NPS</div><div class='metric-value'>{avg_nps:.2f} <span style='font-size:16px; color:#6c757d;'>/10</span></div></div>", unsafe_allow_html=True)
with m_col3:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>⭐ RATA-RATA CSI</div><div class='metric-value'>{avg_csi:.2f} <span style='font-size:16px; color:#6c757d;'>/5</span></div></div>", unsafe_allow_html=True)
with m_col4:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>⏱️ SKOR WAKTU TUNGGU</div><div class='metric-value'>{avg_waiting:.2f} <span style='font-size:16px; color:#6c757d;'>/5</span></div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# 6. Baris Kedua: Grafik Demografi & Sebaran Kepuasan
row2_col1, row2_col2 = st.columns([3, 2])

with row2_col1:
    st.markdown("### 🔀 Tren Kepuasan (NPS) Berdasarkan Umur & Gender")
    if total_respondents > 0:
        fig_scatter = px.scatter(
            df_filtered, x="Age", y="NPS", color="Gender",
            hover_data=["Branch"],
            color_discrete_map={"Male": "#007A87", "Female": "#FF8A8A"},
            labels={"Age": "Umur Pasien", "NPS": "Skor Kepuasan (NPS)"}
        )
        fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10))
        fig_scatter.update_xaxes(showgrid=True, gridcolor="#e9ecef")
        fig_scatter.update_yaxes(showgrid=True, gridcolor="#e9ecef")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Data tidak tersedia untuk kombinasi filter ini.")

with row2_col2:
    st.markdown("### 🏥 Sebaran Responden per Cabang")
    if total_respondents > 0:
        branch_counts = df_filtered['Branch'].value_counts().reset_index()
        branch_counts.columns = ['Cabang', 'Jumlah']
        fig_pie = px.pie(
            branch_counts, values='Jumlah', names='Cabang',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Data tidak tersedia.")


# 7. Baris Ketiga: Analisis Kepuasan Mendalam per Unit Layanan (Insight Utama!)
st.markdown("---")
st.markdown("### 🏢 Nilai Kepuasan Rata-rata Berdasarkan Unit Layanan")

if total_respondents > 0:
    # Mengambil kolom-kolom layanan medis utama
    service_cols = {
        "Pendaftaran": "Registration",
        "Konsultasi Dokter": "Doctor Consultation",
        "Pelayanan Perawat": "Nurse Service",
        "Farmasi / Obat": "Pharmacy Service",
        "Laboratorium": "Laboratory",
        "Respon Gawat Darurat (UGD)": "Emergency Response",
        "Proses Pembayaran (Kasir)": "Billing Process",
        "Kebersihan Fasilitas": "Facility Cleanliness",
        "Keramahan Staf": "Staff Friendliness"
    }
    
    # Hitung rata-rata skor untuk masing-masing layanan
    service_means = []
    service_names = []
    
    for display_name, col_name in service_cols.items():
        if col_name in df_filtered.columns:
            service_means.append(df_filtered[col_name].mean())
            service_names.append(display_name)
            
    # Susun ke dalam dataframe baru untuk divisualisasikan
    df_services = pd.DataFrame({"Layanan": service_names, "Rata-rata Skor": service_means})
    df_services = df_services.sort_values(by="Rata-rata Skor", ascending=True) # Urutkan dari yang terendah untuk evaluasi
    
    # Bikin Grafik Batang Horizontal berwarna Gradasi Teal-Navy
    fig_bar = px.bar(
        df_services, x="Rata-rata Skor", y="Layanan", orientation='h',
        color="Rata-rata Skor",
        color_continuous_scale=["#FF8A8A", "#007A87", "#1E3A8A"],
        range_x=[1, 5],
        labels={"Rata-rata Skor": "Skor Kepuasan Pasien (Skala 1-5)"}
    )
    fig_bar.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, b=10))
    fig_bar.update_xaxes(showgrid=True, gridcolor="#e9ecef")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("💡 *Insight: Layanan dengan batang warna kemerahan/berada di posisi atas memiliki skor kepuasan terendah dan memerlukan evaluasi segera oleh manajemen.*")
else:
    st.info("Data tidak tersedia.")


# 8. Baris Keempat: Tabel Umpan Balik / Saran Pasien (Feedback Komentar)
st.markdown("---")
st.markdown("### 💬 Umpan Balik Langsung & Saran Perbaikan dari Pasien")

if total_respondents > 0 and 'Improvement_Feedback' in df_filtered.columns:
    # Filter keluhan pasien yang bukan saran standar/bagus saja
    feedback_df = df_filtered[['Datetime', 'Branch', 'Gender', 'Age', 'Improvement_Feedback']].copy()
    feedback_df.columns = ['Tanggal', 'Cabang', 'Gender', 'Umur', 'Komentar / Saran Pasien']
    
    # Tampilkan tabel yang bisa di-scroll dan difilter kata kuncinya oleh manajemen rumah sakit
    st.dataframe(feedback_df, use_container_width=True, hide_index=True)
else:
    st.info("Kolom kritik/saran tidak ditemukan atau data kosong.")
