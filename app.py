import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ================= 1. KONFIGURASI & TEMA =================
st.set_page_config(page_title="Advanced Hospital Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .metric-card {background-color: #ffffff; border-top: 4px solid #1E3A8A; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;}
    .metric-title {color: #6c757d; font-size: 13px; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;}
    .metric-value {color: #1E3A8A; font-size: 32px; font-weight: 900;}
    h3 {color: #007A87;}
    </style>
""", unsafe_allow_html=True)

# ================= 2. LOAD & PRE-PROCESS DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("Streamlit.csv", sep=";")
    df['Branch'] = df['Branch'].str.replace('Always Healthy Hospital ', '')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H.%M', errors='coerce')
    df['Date'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour
    
    # Kategori Umur
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
st.sidebar.markdown("## ⚙️ Filter Global Analytics")
min_date, max_date = df['Date'].min(), df['Date'].max()
selected_dates = st.sidebar.date_input("Rentang Tanggal:", [min_date, max_date], min_value=min_date, max_value=max_date)
start_date, end_date = selected_dates if len(selected_dates) == 2 else (min_date, max_date)

selected_branch = st.sidebar.selectbox("Pilih Cabang:", ["Semua Cabang"] + sorted(list(df['Branch'].unique())))
selected_gender = st.sidebar.selectbox("Gender:", ["Semua", "Male", "Female"])

# Filter Logic
df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
if selected_branch != "Semua Cabang":
    df_filtered = df_filtered[df_filtered['Branch'] == selected_branch]
if selected_gender != "Semua":
    df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]

st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏥 Hospital Command Center 360°</h1>", unsafe_allow_html=True)
st.markdown("---")

if df_filtered.empty:
    st.warning("Data kosong untuk filter ini.")
    st.stop()

# ================= 4. TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Executive Summary", "🔍 Service Deep Dive", "❤️ Loyalty & Retention", "💬 Voice of Customer (VOC)"])

# ----------------- TAB 1: EXECUTIVE SUMMARY -----------------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Pasien</div><div class='metric-value'>{len(df_filtered):,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-title'>Skor NPS (0-10)</div><div class='metric-value'>{df_filtered['NPS'].mean():.1f}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-title'>Indeks CSI (1-5)</div><div class='metric-value'>{df_filtered['CSI'].mean():.2f}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-title'>% Promoter</div><div class='metric-value'>{(len(df_filtered[df_filtered['NPS_Category'] == 'Promoter']) / len(df_filtered)) * 100:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    t1_r1_c1, t1_r1_c2 = st.columns([2, 1])
    with t1_r1_c1:
        # V1: Trend Volume vs NPS (Dual Axis)
        trend_data = df_filtered.groupby('Date').agg(Total=('NPS', 'count'), NPS_Avg=('NPS', 'mean')).reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=trend_data['Date'], y=trend_data['Total'], name='Volume Pasien', marker_color='#a0c4ff', yaxis='y1'))
        fig_trend.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['NPS_Avg'], name='NPS', mode='lines+markers', line=dict(color='#1E3A8A', width=3), yaxis='y2'))
        fig_trend.update_layout(title="1. Tren Harian: Volume vs NPS", yaxis=dict(title='Volume'), yaxis2=dict(title='NPS', overlaying='y', side='right', range=[0, 10]), plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0))
        st.plotly_chart(fig_trend, use_container_width=True)
    with t1_r1_c2:
        # V2: NPS Composition Donut
        nps_counts = df_filtered['NPS_Category'].value_counts().reset_index()
        fig_donut = px.pie(nps_counts, values='count', names='NPS_Category', hole=0.6, title="2. Komposisi Kategori NPS", color='NPS_Category', color_discrete_map={'Promoter':'#28a745', 'Passive':'#ffc107', 'Detractor':'#dc3545'})
        fig_donut.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig_donut, use_container_width=True)

    t1_r2_c1, t1_r2_c2 = st.columns([1, 1.5])
    with t1_r2_c1:
        # V3: Leaderboard Cabang
        branch_nps = df_filtered.groupby('Branch')['NPS'].mean().sort_values(ascending=True).reset_index()
        fig_lead = px.bar(branch_nps, x='NPS', y='Branch', orientation='h', title="3. Peringkat Cabang (NPS)", color='NPS', color_continuous_scale='blues')
        fig_lead.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_lead, use_container_width=True)
    with t1_r2_c2:
        # V4: Tree Map Demografi
        fig_tree = px.treemap(df_filtered, path=[px.Constant("Semua Pasien"), 'Branch', 'Gender', 'NPS_Category'], title="4. Peta Demografi & Sentimen", color='NPS', color_continuous_scale='RdYlGn')
        fig_tree.update_layout(margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_tree, use_container_width=True)

# ----------------- TAB 2: SERVICE DEEP DIVE -----------------
with tab2:
    t2_r1_c1, t2_r1_c2 = st.columns(2)
    with t2_r1_c1:
        # V1: Radar Chart Overall Service
        service_means = df_filtered[services].mean().reset_index()
        service_means.columns = ['Layanan', 'Skor']
        fig_radar = px.line_polar(service_means, r='Skor', theta='Layanan', line_close=True, title="1. Radar Kinerja Menyeluruh (Skala 1-5)")
        fig_radar.update_traces(fill='toself', line_color='#007A87')
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[1, 5])), margin=dict(t=40, b=0))
        st.plotly_chart(fig_radar, use_container_width=True)
    with t2_r1_c2:
        # V2: Driver Analysis (Korelasi Layanan ke NPS)
        corr_data = df_filtered[services + ['NPS']].corr()['NPS'].drop('NPS').sort_values(ascending=True)
        fig_driver = px.bar(corr_data, orientation='h', color=corr_data.values, color_continuous_scale='Viridis', title="2. Driver Analysis: Dampak Layanan thd NPS")
        fig_driver.update_layout(xaxis_title="Kekuatan Dampak (Korelasi)", coloraxis_showscale=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_driver, use_container_width=True)

    # V3: Heatmap Kinerja Cabang vs Layanan
    heatmap_data = df_filtered.groupby('Branch')[services].mean()
    fig_heat = px.imshow(heatmap_data.T, color_continuous_scale='RdYlGn', text_auto=".1f", title="3. Heatmap Peta Kritis: Layanan vs Cabang", aspect="auto")
    fig_heat.update_layout(margin=dict(t=40, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("🔍 *Analisis: Area berwarna MERAH adalah titik lemah operasional yang butuh perbaikan secepatnya.*")

    t2_r3_c1, t2_r3_c2 = st.columns(2)
    with t2_r3_c1:
        # V4: Waktu Tunggu Boxplot
        fig_box = px.box(df_filtered, x='Branch', y='Waiting Time', color='Branch', title="4. Distribusi & Outlier Waktu Tunggu")
        fig_box.update_layout(showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_box, use_container_width=True)
    with t2_r3_c2:
        # V5: Perbandingan Dokter vs Perawat
        doc_nurse = df_filtered.groupby('Branch')[['Doctor Consultation', 'Nurse Service']].mean().reset_index()
        fig_dn = go.Figure(data=[
            go.Bar(name='Dokter', x=doc_nurse['Branch'], y=doc_nurse['Doctor Consultation'], marker_color='#1E3A8A'),
            go.Bar(name='Perawat', x=doc_nurse['Branch'], y=doc_nurse['Nurse Service'], marker_color='#007A87')
        ])
        fig_dn.update_layout(barmode='group', title="5. Komparasi Kepuasan: Dokter vs Perawat", yaxis=dict(range=[1, 5]), margin=dict(t=40, b=0))
        st.plotly_chart(fig_dn, use_container_width=True)

# ----------------- TAB 3: LOYALTY & RETENTION -----------------
with tab3:
    t3_r1_c1, t3_r1_c2 = st.columns([1.5, 1])
    with t3_r1_c1:
        # V1: Kuadran Loyalitas (Bubble)
        bubble_data = df_filtered.groupby('Branch').agg(CSI=('CSI', 'mean'), Loyalty=('Loyalty', 'mean'), CES=('CES', 'mean')).reset_index()
        fig_bub = px.scatter(bubble_data, x="CSI", y="Loyalty", size="CES", color="Branch", hover_name="Branch", size_max=35, title="1. Matriks Kuadran: CSI vs Loyalty (Size: Effort)")
        fig_bub.add_vline(x=bubble_data['CSI'].mean(), line_dash="dash", line_color="gray")
        fig_bub.add_hline(y=bubble_data['Loyalty'].mean(), line_dash="dash", line_color="gray")
        fig_bub.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig_bub, use_container_width=True)
    with t3_r1_c2:
        # V2: Heatmap Korelasi Makro
        macro_corr = df_filtered[['NPS', 'CSI', 'Loyalty', 'CES']].corr()
        fig_mcorr = px.imshow(macro_corr, text_auto=".2f", color_continuous_scale='Blues', title="2. Korelasi Metrik Makro")
        fig_mcorr.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig_mcorr, use_container_width=True)

    t3_r2_c1, t3_r2_c2, t3_r2_c3 = st.columns(3)
    with t3_r2_c1:
        # V3: Loyalitas Berdasarkan Umur
        age_loyalty = df_filtered.groupby('Age_Group')['Loyalty'].mean().reset_index()
        fig_al = px.bar(age_loyalty, x='Age_Group', y='Loyalty', title="3. Loyalitas per Generasi Umur", color='Loyalty', color_continuous_scale='Teal')
        fig_al.update_layout(coloraxis_showscale=False, yaxis=dict(range=[1, 5]), margin=dict(t=40, b=0))
        st.plotly_chart(fig_al, use_container_width=True)
    with t3_r2_c2:
        # V4: Sebaran Customer Effort Score (CES)
        fig_ces = px.histogram(df_filtered, x='CES', color='NPS_Category', barmode='group', title="4. Distribusi Upaya Pelanggan", color_discrete_map={'Promoter':'#28a745', 'Passive':'#ffc107', 'Detractor':'#dc3545'})
        fig_ces.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig_ces, use_container_width=True)
    with t3_r2_c3:
        # V5: Trend Loyalitas
        trend_loyalty = df_filtered.groupby('Date')['Loyalty'].mean().reset_index()
        fig_tl = px.line(trend_loyalty, x='Date', y='Loyalty', markers=True, title="5. Tren Retensi/Loyalitas Harian")
        fig_tl.update_layout(yaxis=dict(range=[1, 5]), margin=dict(t=40, b=0))
        st.plotly_chart(fig_tl, use_container_width=True)

# ----------------- TAB 4: VOICE OF CUSTOMER -----------------
with tab4:
    t4_r1_c1, t4_r1_c2 = st.columns([1, 1])
    with t4_r1_c1:
        # V1: Kepuasan Berdasarkan Jam Kunjungan (Kapan layanan terburuk?)
        hour_nps = df_filtered.groupby('Hour')['NPS'].mean().reset_index()
        fig_hour = px.line(hour_nps, x='Hour', y='NPS', markers=True, title="1. Kapan Kepuasan Menurun? (NPS per Jam)", line_shape='spline')
        fig_hour.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2, title="Jam (Format 24H)"), margin=dict(t=40, b=0))
        st.plotly_chart(fig_hour, use_container_width=True)
    with t4_r1_c2:
        # V2: Karakteristik Umur Pasien Kecewa
        detractors = df_filtered[df_filtered['NPS_Category'] == 'Detractor']
        if not detractors.empty:
            fig_detage = px.histogram(detractors, x='Age', nbins=15, title="2. Profil Umur Pasien Detractor (Kecewa)", color_discrete_sequence=['#dc3545'])
            fig_detage.update_layout(margin=dict(t=40, b=0))
            st.plotly_chart(fig_detage, use_container_width=True)
        else:
            st.success("Tidak ada Detractor di rentang waktu ini!")

    # V3 & V4: Tabel Feedback & Volume Feedback per Cabang
    st.markdown("### 3. Log Umpan Balik Kualitatif (VOC Data)")
    if 'Improvement_Feedback' in df_filtered.columns:
        cat = st.radio("Saring Sentimen Komentar:", ["Semua", "Detractor (Merah)"], horizontal=True)
        vocab_df = df_filtered if cat == "Semua" else detractors
        st.dataframe(vocab_df[['Datetime', 'Branch', 'Gender', 'Age', 'NPS', 'Improvement_Feedback']].sort_values(by='NPS'), use_container_width=True, height=250, hide_index=True)
    
    # V5: Stacked Bar Kategori NPS per Cabang
    fig_stack = px.histogram(df_filtered, y="Branch", color="NPS_Category", orientation='h', title="4. Distribusi Sentimen per Cabang (Volume)", color_discrete_map={'Promoter':'#28a745', 'Passive':'#ffc107', 'Detractor':'#dc3545'})
    fig_stack.update_layout(barmode='stack', margin=dict(t=40, b=0))
    st.plotly_chart(fig_stack, use_container_width=True)
