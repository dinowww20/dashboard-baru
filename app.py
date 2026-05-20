import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard Rumah Sakit", layout="wide")
st.title("📊 Dashboard Analisis Kepuasan Pasien")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Streamlit.csv", sep=";")

try:
    df = load_data()
except Exception as e:
    st.error("Data Streamlit.csv belum diupload ke GitHub!")
    st.stop()

# Sidebar untuk Filter
st.sidebar.header("Pilihan Filter")
branch = st.sidebar.selectbox("Pilih Cabang Rumah Sakit:", ["Semua Cabang"] + list(df['Branch'].unique()))

if branch != "Semua Cabang":
    df = df[df['Branch'] == branch]

# Menampilkan Metrik Utama
col1, col2, col3 = st.columns(3)
col1.metric("Total Responden", len(df))
col2.metric("Rata-rata NPS", round(df['NPS'].mean(), 2))
col3.metric("Rata-rata CSI", round(df['CSI'].mean(), 2))

st.markdown("---")

# Menampilkan Grafik
st.subheader("Tren Kepuasan (NPS) Berdasarkan Umur")
fig1 = px.scatter(df, x="Age", y="NPS", color="Gender", hover_data=["Branch"])
st.plotly_chart(fig1)

st.subheader("Distribusi Penilaian Layanan Perawat")
fig2 = px.histogram(df, x="Nurse Service", color="Gender", barmode="group")
st.plotly_chart(fig2)

# Menampilkan Tabel Data
st.markdown("---")
st.subheader("Tabel Data Mentah")
st.dataframe(df)
