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
st.set_page_config(page_title="Executive Analytics", layout="wide", initial_sidebar_state="expanded")

# CSS Elite Corporate: Mewah, Lembut, dan Sangat Profesional
st.markdown("""
    <style>
    /* Latar belakang aplikasi sedikit keabu-abuan agar kartu putih lebih menonjol */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Desain Kartu Metrik Elite */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #f1f5f9;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -4px rgba(15, 23, 42, 0.08);
    }
    .metric-title {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 34px;
        font-weight: 800;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Styling Tabs yang Elegan */
    .stTabs [data-baseweb="tab-list"] {
        gap: 32px;
        border-bottom: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 54px;
        background-color: transparent;
        border-radius: 0;
        padding-top: 12px;
        padding-bottom: 12px;
        color: #64748b;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        border-bottom: 3px solid #0f172a !important;
        font-weight: 600;
        background-color: transparent !important;
    }
    
    /* Tipografi Header */
    h1, h2, h3, h4 {
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    hr {
        border-color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Palet Warna Elite / Muted
nps_colors = {
    'Promoter': '#059669', # Emerald Muted
    'Passive': '#94a3b8',  # Silver/Slate
    'Detractor': '#e11d48' # Crimson/Rose
}
primary_color = '#1e293b'   # Midnight Blue
secondary_color = '#0284c7' # Elegant Ocean Blue

# Fungsi bantuan untuk mengatur tema layout plotly secara global
def elite_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#0f172a', family='Helvetica Neue')),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=20, l=20, r=20),
        font=dict(color='#475569'),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Helvetica Neue")
    )
    # Menyamarkan gridlines agar lebih bersih
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9', zeroline=False)
    return fig

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
st.sidebar.markdown("<h3 style='color: #0f172a;'>Parameter Analisis</h3>", unsafe_allow_html=True)
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

st.markdown("<h2 style='text-align: center; margin-bottom: 40px; color: #0f172a;'>Executive Command Center</h2>", unsafe_allow_html=True)

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
    col1.markdown(f"<div class='metric-card'><div class='metric-title'>Total Pengunjung</div><div class='metric-value'>{len(df_filtered):,}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-card'><div class='metric-title'>Net Promoter Score</div><div class='metric-value'>{df_filtered['NPS'].mean():.1f}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-card'><div class='metric-title'>Indeks Kepuasan</div><div class='metric-value'>{df_filtered['CSI'].mean():.2f}</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-card'><div class='metric-title'>Tingkat Loyalitas</div><div class='metric-value'>{(len(df_filtered[df_filtered['NPS_Category'] == 'Promoter']) / len(df_filtered)) * 100:.1f}%</div></div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)

    t1_r1_c1, t1_r1_c2 = st.columns([2.5, 1.5])
    with t1_r1_c1:
        trend_data = df_filtered.groupby('Date').agg(Total=('NPS', 'count'), NPS_Avg=('NPS', 'mean')).reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(x=trend_data['Date'], y=trend_data['Total'], name='Volume Kunjungan', marker_color='#e2e8f0', yaxis='y1'))
        fig_trend.add_trace(go.Scatter(x=trend_data['Date'], y=trend_data['NPS_Avg'], name='NPS Rata-rata', mode='lines+markers', line=dict(color=primary_color, width=3), yaxis='y2'))
        fig_trend = elite_layout(fig_trend, "Dinamika Kunjungan & Skor Kepuasan Operasional")
        fig_trend.update_layout(yaxis=dict(title='Volume Pasien', showgrid=False), yaxis2=dict(title='Skor NPS', overlaying='y', side='right', range=[0, 10], showgrid=False))
        st.plotly_chart(fig_trend, use_container_width=True)
    with t1_r1_c2:
        nps_counts = df_filtered['NPS_Category'].value_counts().reset_index()
        fig_donut = px.pie(nps_counts, values='count', names='NPS_Category', hole=0.7, color='NPS_Category', color_discrete_map=nps_colors)
        fig_donut = elite_layout(fig_donut, "Distribusi Segmen Pelanggan")
        fig_donut.update_traces(textposition='outside', textinfo='percent+label', hoverinfo='label+value+percent')
        fig_donut.update_layout(showlegend=False) # Hilangkan legend agar lebih bersih
        st.plotly_chart(fig_donut, use_container_width=True)

    t1_r2_c1, t1_r2_c2 = st.columns([1.5, 2.5])
    with t1_r2_c1:
        branch_nps = df_filtered.groupby('Branch')['NPS'].mean().sort_values(ascending=True).reset_index()
        fig_lead = px.bar(branch_nps, x='NPS', y='Branch', orientation='h', color='NPS', color_continuous_scale='Teal')
        fig_lead = elite_layout(fig_lead, "Peringkat Kinerja Fasilitas (Berdasarkan NPS)")
        fig_lead.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_lead, use_container_width=True)
    with t1_r2_c2:
        fig_tree = px.treemap(df_filtered, path=[px.Constant("Semua Pasien"), 'Branch', 'Gender', 'NPS_Category'], color='NPS', color_continuous_scale='Teal')
        fig_tree = elite_layout(fig_tree, "Peta Hierarki Demografi & Sentimen")
        st.plotly_chart(fig_tree, use_container_width=True)

# ----------------- TAB 2: SERVICE DEEP DIVE -----------------
with tab2:
    t2_r1_c1, t2_r1_c2 = st.columns(2)
    with t2_r1_c1:
        service_means = df_filtered[services].mean().reset_index()
        service_means.columns = ['Layanan', 'Skor']
        fig_radar = px.line_polar(service_means, r='Skor', theta='Layanan', line_close=True)
        fig_radar.update_traces(fill='toself', line_color=secondary_color, fillcolor='rgba(2, 132, 199, 0.15)', line_width=2)
        fig_radar = elite_layout(fig_radar, "Spektrum Kinerja Layanan Menyeluruh")
        fig_radar.update_layout(polar=dict(radialaxis=dict(range=[1, 5], gridcolor='#e2e8f0'), angularaxis=dict(gridcolor='#e2e8f0')))
        st.plotly_chart(fig_radar, use_container_width=True)
    with t2_r1_c2:
        corr_data = df_filtered[services + ['NPS']].corr()['NPS'].drop('NPS').sort_values(ascending=True)
        fig_driver = px.bar(corr_data, orientation='h', color=corr_data.values, color_continuous_scale='Blues')
        fig_driver = elite_layout(fig_driver, "Analisis Faktor Pendorong Kepuasan (Key Drivers)")
        fig_driver.update_layout(xaxis_title="Koefisien Korelasi", coloraxis_showscale=False)
        st.plotly_chart(fig_driver, use_container_width=True)

    heatmap_data = df_filtered.groupby('Branch')[services].mean()
    fig_heat = px.imshow(heatmap_data.T, color_continuous_scale='Teal', text_auto=".1f", aspect="auto")
    fig_heat = elite_layout(fig_heat, "Matriks Panas: Evaluasi Layanan Lintas Fasilitas")
    st.plotly_chart(fig_heat, use_container_width=True)

    t2_r3_c1, t2_r3_c2 = st.columns(2)
    with t2_r3_c1:
        fig_box = px.box(df_filtered, x='Branch', y='Waiting Time', color='Branch', color_discrete_sequence=px.colors.qualitative.Prism)
        fig_box = elite_layout(fig_box, "Distribusi Waktu Tunggu Operasional")
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with t2_r3_c2:
        doc_nurse = df_filtered.groupby('Branch')[['Doctor Consultation', 'Nurse Service']].mean().reset_index()
        fig_dn = go.Figure(data=[
            go.Bar(name='Konsultasi Dokter', x=doc_nurse['Branch'], y=doc_nurse['Doctor Consultation'], marker_color=primary_color),
            go.Bar(name='Layanan Perawat', x=doc_nurse['Branch'], y=doc_nurse['Nurse Service'], marker_color=secondary_color)
        ])
        fig_dn = elite_layout(fig_dn, "Komparasi Evaluasi Medis Profesional")
        fig_dn.update_layout(barmode='group', yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig_dn, use_container_width=True)

# ----------------- TAB 3: LOYALTY & RETENTION -----------------
with tab3:
    t3_r1_c1, t3_r1_c2 = st.columns([1.5, 1])
    with t3_r1_c1:
        bubble_data = df_filtered.groupby('Branch').agg(CSI=('CSI', 'mean'), Loyalty=('Loyalty', 'mean'), CES=('CES', 'mean')).reset_index()
        fig_bub = px.scatter(bubble_data, x="CSI", y="Loyalty", size="CES", color="Branch", hover_name="Branch", size_max=35, color_discrete_sequence=px.colors.qualitative.Safe)
        fig_bub = elite_layout(fig_bub, "Matriks Kuadran Strategis (Kepuasan vs Retensi)")
        fig_bub.add_vline(x=bubble_data['CSI'].mean(), line_dash="dash", line_color="#cbd5e1", line_width=2)
        fig_bub.add_hline(y=bubble_data['Loyalty'].mean(), line_dash="dash", line_color="#cbd5e1", line_width=2)
        st.plotly_chart(fig_bub, use_container_width=True)
    with t3_r1_c2:
        macro_corr = df_filtered[['NPS', 'CSI', 'Loyalty', 'CES']].corr()
        fig_mcorr = px.imshow(macro_corr, text_auto=".2f", color_continuous_scale='Blues')
        fig_mcorr = elite_layout(fig_mcorr, "Korelasi Indikator Strategis Makro")
        st.plotly_chart(fig_mcorr, use_container_width=True)

    t3_r2_c1, t3_r2_c2, t3_r2_c3 = st.columns(3)
    with t3_r2_c1:
        age_loyalty = df_filtered.groupby('Age_Group')['Loyalty'].mean().reset_index()
        fig_al = px.bar(age_loyalty, x='Age_Group', y='Loyalty', color='Loyalty', color_continuous_scale='Blues')
        fig_al = elite_layout(fig_al, "Profil Retensi Berdasarkan Generasi")
        fig_al.update_layout(coloraxis_showscale=False, yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig_al, use_container_width=True)
    with t3_r2_c2:
        fig_ces = px.histogram(df_filtered, x='CES', color='NPS_Category', barmode='group', color_discrete_map=nps_colors)
        fig_ces = elite_layout(fig_ces, "Distribusi Beban Upaya Pelanggan (CES)")
        st.plotly_chart(fig_ces, use_container_width=True)
    with t3_r2_c3:
        trend_loyalty = df_filtered.groupby('Date')['Loyalty'].mean().reset_index()
        fig_tl = px.line(trend_loyalty, x='Date', y='Loyalty', markers=True, line_shape='spline', line=dict(color=secondary_color, width=3))
        fig_tl = elite_layout(fig_tl, "Tren Retensi & Loyalitas Periode Aktif")
        fig_tl.update_layout(yaxis=dict(range=[1, 5]))
        st.plotly_chart(fig_tl, use_container_width=True)

# ----------------- TAB 4: VOICE OF CUSTOMER -----------------
with tab4:
    t4_r1_c1, t4_r1_c2 = st.columns([1, 1])
    with t4_r1_c1:
        hour_nps = df_filtered.groupby('Hour')['NPS'].mean().reset_index()
        fig_hour = px.line(hour_nps, x='Hour', y='NPS', markers=True, line_shape='spline', line=dict(color=primary_color, width=3))
        fig_hour = elite_layout(fig_hour, "Volatilitas Sentimen Berdasarkan Jam Operasional")
        fig_hour.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=2))
        st.plotly_chart(fig_hour, use_container_width=True)
    with t4_r1_c2:
        detractors = df_filtered[df_filtered['NPS_Category'] == 'Detractor']
        if not detractors.empty:
            fig_detage = px.histogram(detractors, x='Age', nbins=15, color_discrete_sequence=[nps_colors['Detractor']])
            fig_detage = elite_layout(fig_detage, "Profil Demografi Segmen Kritis (Detractor)")
            st.plotly_chart(fig_detage, use_container_width=True)
        else:
            st.info("Kondisi Optimal: Tidak terdeteksi pasien pada segmen Detractor di rentang waktu ini.")

    st.markdown("<h4 style='margin-top: 20px;'>Tinjauan Data Kualitatif</h4>", unsafe_allow_html=True)
    if 'Improvement_Feedback' in df_filtered.columns:
        cat = st.radio("Saring Sentimen Evaluasi:", ["Seluruh Data Kualitatif", "Fokus Segmen Kritis (Detractor)"], horizontal=True)
        vocab_df = df_filtered if cat == "Seluruh Data Kualitatif" else detractors
        
        # Tampilkan tabel tanpa index dan styling yang lebih bersih
        st.dataframe(
            vocab_df[['Datetime', 'Branch', 'Gender', 'Age', 'NPS', 'Improvement_Feedback']].sort_values(by='NPS'), 
            use_container_width=True, height=250, hide_index=True
        )
    
    fig_stack = px.histogram(df_filtered, y="Branch", color="NPS_Category", orientation='h', color_discrete_map=nps_colors)
    fig_stack = elite_layout(fig_stack, "Proporsi Sentimen Berdasarkan Wilayah Operasional")
    fig_stack.update_layout(barmode='stack')
    st.plotly_chart(fig_stack, use_container_width=True)

# ----------------- TAB 5: TEXT ANALYTICS -----------------
with tab5:
    st.markdown("<h4 style='margin-bottom: 20px;'>Analisis Semantik & Ekstraksi Kata Kunci</h4>", unsafe_allow_html=True)
    
    if 'Improvement_Feedback' in df_filtered.columns:
        all_text = " ".join(df_filtered['Improvement_Feedback'].dropna().astype(str).tolist()).lower()
        
        if len(all_text.strip()) > 0:
            t5_r1_c1, t5_r1_c2 = st.columns(2)
            
            with t5_r1_c1:
                st.markdown("<p style='font-weight: 600; color: #64748b; font-size: 14px; text-transform: uppercase;'>Pemetaan Leksikal (Global)</p>", unsafe_allow_html=True)
                wordcloud_all = WordCloud(width=800, height=500, background_color='#f8fafc', colormap='ocean', max_words=100, contour_width=0).generate(all_text)
                fig_wc_all, ax_all = plt.subplots(figsize=(8, 5))
                ax_all.imshow(wordcloud_all, interpolation='bilinear')
                ax_all.axis('off')
                fig_wc_all.patch.set_facecolor('#f8fafc')
                st.pyplot(fig_wc_all)
                
            with t5_r1_c2:
                st.markdown("<p style='font-weight: 600; color: #64748b; font-size: 14px; text-transform: uppercase;'>Pemetaan Leksikal (Segmen Kritis)</p>", unsafe_allow_html=True)
                detractor_text = " ".join(df_filtered[df_filtered['NPS_Category'] == 'Detractor']['Improvement_Feedback'].dropna().astype(str).tolist()).lower()
                
                if len(detractor_text.strip()) > 0:
                    wordcloud_det = WordCloud(width=800, height=500, background_color='#f8fafc', colormap='Reds', max_words=100).generate(detractor_text)
                    fig_wc_det, ax_det = plt.subplots(figsize=(8, 5))
                    ax_det.imshow(wordcloud_det, interpolation='bilinear')
                    ax_det.axis('off')
                    fig_wc_det.patch.set_facecolor('#f8fafc')
                    st.pyplot(fig_wc_det)
                else:
                    st.info("Insufisiensi data kualitatif pada segmen kritis untuk dilakukan pemetaan.")
                    
            st.markdown("<hr>", unsafe_allow_html=True)
            
            words = re.findall(r'\b\w{4,}\b', all_text)
            
            if words:
                word_counts = Counter(words).most_common(10)
                df_words = pd.DataFrame(word_counts, columns=['Terminologi', 'Frekuensi Aktual'])
                df_words = df_words.sort_values(by='Frekuensi Aktual', ascending=True)
                
                fig_bar_words = px.bar(
                    df_words, x='Frekuensi Aktual', y='Terminologi', orientation='h',
                    color='Frekuensi Aktual', color_continuous_scale='Teal', text='Frekuensi Aktual'
                )
                fig_bar_words = elite_layout(fig_bar_words, "Distribusi Terminologi Teratas")
                fig_bar_words.update_traces(textposition='outside')
                fig_bar_words.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_bar_words, use_container_width=True)
            else:
                st.info("Kuantitas data teks tidak mencukupi untuk ekstraksi analitik.")

        else:
            st.warning("Data kualitatif (teks) tidak ditemukan pada rentang filter saat ini.")
    else:
        st.warning("Variabel teks evaluasi tidak ditemukan dalam struktur dataset.")
