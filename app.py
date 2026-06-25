import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import json
from collections import Counter

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import matplotlib; matplotlib.use('Agg')
    WC_OK = True
except ImportError:
    WC_OK = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SK_OK = True
except ImportError:
    SK_OK = False

try:
    from scipy.stats import pearsonr, linregress
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Dashboard Bank XYZ",
                   layout="wide", initial_sidebar_state="expanded")

# ── LOGIKA TEMA (DARK/LIGHT) ─────────────────────────
with st.sidebar:
    theme_mode = st.radio("Mode Tampilan:", ["Light", "Dark"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)

if theme_mode == "Dark":
    bg_app = "#0F172A"      # Slate 900
    bg_panel = "#1E293B"    # Slate 800
    text_main = "#F8FAFC"   # Slate 50
    text_muted = "#94A3B8"  # Slate 400
    border_col = "#334155"  # Slate 700
    hover_bg = "#064E3B"    # Emerald 900
    accent_color = "#10B981"
    chart_template = "plotly_dark"
else:
    bg_app = "#F8FAFC"      # Slate 50
    bg_panel = "#FFFFFF"    # White
    text_main = "#1E293B"   # Slate 800
    text_muted = "#64748B"  # Slate 500
    border_col = "#E2E8F0"  # Slate 200
    hover_bg = "#ECFDF5"    # Emerald 50
    accent_color = "#10B981"
    chart_template = "plotly_white"

# ═══════════════════════════════════════════════════════
# DYNAMIC STYLE INJECTION
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Typography (Targeted agar tidak merusak Material Icons) */
.stApp, p, h1, h2, h3, h4, h5, h6, label, li, .stMarkdown div {{
    font-family: 'Inter', -apple-system, sans-serif !important;
}}

/* Base & Structural */
.stApp {{ background: {bg_app} !important; color: {text_main} !important; }}
.main .block-container {{ padding-top: .6rem; padding-bottom: 2rem; max-width: 100%; }}
[data-testid="stHeader"] {{ background: rgba(0, 0, 0, 0) !important; }}

/* Sidebar */
[data-testid="stSidebar"] {{
  background-color: {bg_panel} !important;
  border-right: 1px solid {border_col} !important;
}}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {{ 
  color: {text_main} !important; font-size: 14px; 
}}
[data-testid="stSidebar"] .stSelectbox>label,
[data-testid="stSidebar"] .stMultiSelect>label,
[data-testid="stSidebar"] .stSlider>label {{
  color: {text_muted} !important; font-size: 11px !important; font-weight: 700 !important;
  text-transform: uppercase; letter-spacing: .8px; margin-bottom: 4px;
}}

/* Fix untuk Radio Button dan Checkbox agar teksnya adaptif tanpa merusak tab */
div[role="radiogroup"] *, [data-baseweb="radio"] *, 
[data-baseweb="checkbox"] *, label[data-testid="stWidgetLabel"] p {{
    color: {text_main} !important;
}}

/* Input Kotak & Dropdown */
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"],
div[data-baseweb="base-input"] > input,
.stTextInput input, .stTextArea textarea {{
    background-color: {bg_panel} !important;
    color: {text_main} !important;
    border-color: {border_col} !important;
    font-size: 14px !important;
}}

/* Popover & Listbox (Dropdown Options) */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"],
ul[data-baseweb="menu"] {{ 
    background-color: {bg_panel} !important; 
    border: 1px solid {border_col} !important; 
}}
li[role="option"] {{ 
    color: {text_main} !important; 
    font-size: 14px !important; 
    background-color: {bg_panel} !important; 
}}
li[role="option"]:hover, li[aria-selected="true"] {{ 
    background-color: {hover_bg} !important; 
    color: {accent_color} !important; 
}}

/* Dataframe & Tables */
[data-testid="stDataFrame"] {{ background-color: {bg_panel} !important; border: 1px solid {border_col}; border-radius: 8px; }}
[data-testid="stDataFrame"] * {{ color: {text_main} !important; }}
[data-testid="stDataFrame"] th {{ background-color: {bg_app} !important; border-bottom: 1px solid {border_col} !important; }}
[data-testid="stDataFrame"] td {{ background-color: {bg_panel} !important; border-bottom: 1px solid {border_col} !important; }}

/* Tags Multiselect */
span[data-baseweb="tag"] {{ background-color: {hover_bg} !important; border: 1px solid {accent_color} !important; }}
span[data-baseweb="tag"] span {{ color: {accent_color} !important; font-size: 12px !important; }}

/* Tabs Navigation */
.stTabs [data-baseweb="tab-list"], div[role="tablist"] {{
  background-color: {bg_panel} !important; border-radius: 14px;
  padding: 6px; gap: 4px; border: 1px solid {border_col} !important;
}}
.stTabs button[role="tab"] {{
  background: transparent !important; border: none !important;
  border-radius: 10px; padding: 10px 16px;
  font-weight: 600; font-size: 13px !important;
}}
.stTabs button[role="tab"] p {{
  color: {text_muted} !important;
}}

/* Memaksa Tab Aktif menjadi Hijau dengan Teks Putih */
.stTabs button[role="tab"][aria-selected="true"],
.stTabs button[aria-selected="true"] {{
  background: linear-gradient(135deg, #059669, #10B981) !important;
  background-color: #10B981 !important;
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
  border: none !important;
}}
.stTabs button[role="tab"][aria-selected="true"] *,
.stTabs button[aria-selected="true"] *,
.stTabs button[role="tab"][aria-selected="true"] p,
.stTabs button[aria-selected="true"] p {{ 
  color: #FFFFFF !important; 
}}

/* Menghilangkan garis merah bawah bawaan Streamlit pada tab aktif */
div[data-baseweb="tab-highlight"] {{
    display: none !important;
}}

/* Slider Panah Tabs */
button[title="Scroll right"], button[title="Scroll left"] {{
    background: transparent !important; background-color: transparent !important;
}}
button[title="Scroll right"] *, button[title="Scroll left"] * {{ fill: {text_main} !important; color: {text_main} !important; }}

/* Metric Cards */
.metric-card {{
  background: {bg_panel}; border: 1px solid {border_col};
  padding: 16px 12px; border-radius: 18px; text-align: center;
  transition: all .3s; position: relative; overflow: hidden;
  height: 200px;
  display: flex; flex-direction: column; justify-content: space-evenly; align-items: center;
}}
.metric-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  background: linear-gradient(90deg, #047857, #10B981, #6EE7B7);
}}
.metric-title {{ color: {text_muted} !important; font-size: 11px !important; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; width: 100%; line-height: 1.4; }}
.metric-value {{ font-size: 34px !important; font-weight: 900; line-height: 1; width: 100%; }}
.metric-value.primary, .metric-value.green {{ color: #10B981 !important; }}
.metric-value.red   {{ color: #EF4444 !important; }}
.metric-value.amber {{ color: #F59E0B !important; }}
.metric-value.black {{ color: {text_main} !important; }}
.metric-sub {{ color: {text_muted} !important; font-size: 11px !important; width: 100%; }}
.metric-delta-up  {{ color: #10B981 !important; font-size: 12px !important; font-weight: 700; margin-top: 4px; }}
.metric-delta-down{{ color: #EF4444 !important; font-size: 12px !important; font-weight: 700; margin-top: 4px; }}
.metric-delta-neu {{ color: {text_muted} !important; font-size: 12px !important; font-weight: 700; margin-top: 4px; }}

/* Section Headers */
.sec-hdr {{ font-size: 16px !important; font-weight: 800; color: {text_main} !important; border-left: 4px solid #10B981; padding-left: 12px; margin: 24px 0 14px; }}

/* Insight Box */
.insight-box {{ background: {hover_bg}; border: 1px solid #A7F3D0; border-radius: 12px; padding: 14px 18px; margin: 12px 0; }}
.insight-box p, .insight-box b {{ color: {text_main} !important; font-size: 14px !important; margin: 0; line-height: 1.6; }}

/* Chat Box */
.chat-wrap {{ background: {bg_panel}; border: 1px solid {border_col}; border-radius: 14px; padding: 20px; margin: 12px 0; max-height: 440px; overflow-y: auto; }}
.chat-user {{ background: {hover_bg}; border-radius: 14px 14px 4px 14px; padding: 12px 16px; margin: 8px 0 8px auto; max-width: 76%; display: inline-block; float: right; clear: both; border: 1px solid {border_col}; }}
.chat-user p {{ color: {text_main} !important; font-size: 14px !important; margin: 0; line-height: 1.6; }}
.chat-ai {{ background: {bg_app}; border: 1px solid {border_col}; border-radius: 14px 14px 14px 4px; padding: 12px 16px; margin: 8px 0; max-width: 82%; display: inline-block; float: left; clear: both; }}
.chat-ai p {{ color: {text_main} !important; font-size: 14px !important; margin: 0; line-height: 1.6; }}
.chat-cf {{ clear: both; }}

/* Buttons */
.stButton button {{ background: linear-gradient(135deg, #059669, #10B981) !important; color: #FFFFFF !important; border: none !important; border-radius: 10px; font-weight: 700; font-size: 14px !important; transition: all .2s; padding: 8px 16px; }}
.stButton button:hover {{ box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25); transform: translateY(-1px); }}
.stButton button * {{ color: #FFFFFF !important; }}
hr {{ border-color: {border_col} !important; }}
</style>
""", unsafe_allow_html=True)

NPS_C  = {'Promoter':'#10B981','Passive':'#F59E0B','Detractor':'#EF4444'}
C_XYZ  = '#10B981' # Soft Emerald Green
C_KOMP = '#F43F5E' # Rose Red for Competitor
C_IMP  = '#94A3B8'
GRID   = border_col
BG     = 'rgba(0,0,0,0)' 

NAMA_KOLOM = {
    'G1A':  'NPS Score', 'E1A':  'Kepuasan Nasabah', 'F1A':  'Loyalitas Nasabah',
    'OVR_TELLER_XYZ': 'Layanan Teller', 'OVR_CS_XYZ': 'Layanan Customer Service',
    'OVR_ATM_XYZ': 'Layanan ATM', 'OVR_SEKURITI_XYZ': 'Layanan Sekuriti',
    'OVR_KC_XYZ': 'Layanan Kantor Cabang', 'OVR_SARANA_XYZ': 'Sarana Elektronik',
    'OVR_CA_XYZ': 'Layanan Customer Advisor', 'S2_1': 'Usia (Numerik)', 'Emosi_Pos': 'Emosi Positif',
}
def nama_kolom(kode): return NAMA_KOLOM.get(kode, kode)

def card(title, value, color="black", sub="", delta=None, dlbl="vs Global"):
    if delta is not None:
        if delta > 0.005: dh = f"<div class='metric-delta-up'>▲ {abs(delta):.2f} {dlbl}</div>"
        elif delta < -0.005: dh = f"<div class='metric-delta-down'>▼ {abs(delta):.2f} {dlbl}</div>"
        else: dh = f"<div class='metric-delta-neu'>─ Sama dgn {dlbl}</div>"
    else: dh = ""
    return (f"<div class='metric-card'>"
            f"<div class='metric-title'>{title}</div>"
            f"<div class='metric-value {color}'>{value}</div>"
            f"{'<div class=metric-sub>'+sub+'</div>' if sub else ''}{dh}</div>")

def ib(txt):
    txt_parsed = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', txt)
    st.markdown(f"<div class='insight-box'><p>{txt_parsed}</p></div>", unsafe_allow_html=True)

def sh(txt): st.markdown(f"<div class='sec-hdr'>{txt}</div>", unsafe_allow_html=True)

def elo(fig, title="", h=None, legend_below=False):
    title_text = f"<b>{title}</b>" if title else ""
    legend_pre_set = fig.layout.legend.y is not None
    margin_pre_set = fig.layout.margin.b is not None

    if legend_below:
        legend_kw = dict(font=dict(color=text_muted, size=12), bgcolor=bg_panel,
                          bordercolor=border_col, orientation='h', yanchor='top', y=-0.18, xanchor='center', x=0.5)
        margin_kw = dict(t=50, b=64, l=14, r=14)
    else:
        legend_kw = dict(font=dict(color=text_muted, size=12), bgcolor=bg_panel,
                          bordercolor=border_col, orientation='h', yanchor='bottom', y=1.10, xanchor='right', x=1)
        margin_kw = dict(t=72, b=20, l=14, r=14)

    if legend_pre_set:
        # Chart ini sudah punya posisi legend custom (misal radar/pie dgn legend di bawah)
        # sebelum elo() dipanggil — jangan timpa posisinya, cukup samakan styling warna/font.
        legend_kw = dict(font=dict(color=text_muted, size=12), bgcolor=bg_panel, bordercolor=border_col)

    kw = dict(
        title=dict(text=title_text, font=dict(size=16, color=text_main), y=0.97, yanchor='top'),
        template=chart_template, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=text_muted, size=13, family='Inter, sans-serif'),
        hoverlabel=dict(bgcolor=bg_panel, font_color=text_main, font_size=14, bordercolor=border_col),
        legend=legend_kw,
    )
    if not margin_pre_set: kw['margin'] = margin_kw
    if h: kw['height'] = h
    fig.update_layout(**kw)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, automargin=True, tickfont=dict(color=text_muted, size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, automargin=True, tickfont=dict(color=text_muted, size=12))
    return fig

def slbl(col, col_map, n=36):
    s = col_map.get(col, col)
    s = re.sub(r'\s*-\s*(XYZ|kompetitor)\s*$','',s,flags=re.I)
    s = re.sub(r'\([^)]*\)','',s).strip()
    return re.sub(r'\s+',' ',s)[:n]+'…' if len(re.sub(r'\s+',' ',s))>n else re.sub(r'\s+',' ',s)

EMPTY_SENTINELS = {'','none','nan','tidak mengisi','tidak ada','tidak tahu', 'tidak ada / tidak tahu','tdk mengisi','-','n/a','na'}

def _norm(p):
    return re.sub(r'\s+',' ', str(p).strip().lower())

def parse_cmt(raw):
    if pd.isna(raw) or _norm(raw) in EMPTY_SENTINELS: return None
    parts = [p.strip() for p in str(raw).split(';')]
    skip = {'NET','SUBNET','POSITIVE COMMENTS','NEGATIVE COMMENTS','LAIN-LAIN','OTHERS','NEGATIVE COMMENTS (NET)','POSITIVE COMMENTS (NET)'}
    for p in reversed(parts):
        if len(p)>8 and p.upper() not in skip and _norm(p) not in EMPTY_SENTINELS: return p
    return None

def clean_cmt(s): return s.apply(parse_cmt).dropna()

STOP = {'yang','untuk','dengan','pada','dari','sebagai','tidak','karena','sangat','lebih','sudah','saya','bank','bisa','dan','di','ke','ini','itu','ada','juga','net','subnet','positive','negative','comments','dalam','oleh','akan','telah','dapat','kami','anda','nya','atau','jadi','baru','lagi','saat','masih','serta','namun','jika','agar','bagi','atas','antara','setiap','para','mereka','kita','xyz','nasabah','layanan','cabang','rekening','lainnya','none','baik','bagus','cukup','sekali','paling','makin','belum','kurang'}

def calc_nps(s):
    t = s.notna().sum()
    if t==0: return 0.,0.,0.,0.
    pr=(s=='Promoter').sum(); ps=(s=='Passive').sum(); dt=(s=='Detractor').sum()
    return round((pr-dt)/t*100,1),round(pr/t*100,1),round(ps/t*100,1),round(dt/t*100,1)

def corr_with_stats(data):
    """Hitung matriks korelasi Pearson beserta p-value dan N pairwise (bukan N global)
    supaya hover chart menampilkan basis statistik yang akurat per pasangan kolom."""
    cols = data.columns.tolist()
    r_mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    p_mat = pd.DataFrame(np.zeros((len(cols),len(cols))), index=cols, columns=cols)
    n_mat = pd.DataFrame(np.zeros((len(cols),len(cols))), index=cols, columns=cols)
    for i,c1 in enumerate(cols):
        for j,c2 in enumerate(cols):
            sub = data[[c1,c2]].dropna()
            n_pair = len(sub)
            n_mat.loc[c1,c2] = n_pair
            if i==j:
                r_mat.loc[c1,c2]=1.0; p_mat.loc[c1,c2]=0.0; continue
            if n_pair<3 or sub[c1].std()==0 or sub[c2].std()==0:
                r_mat.loc[c1,c2]=np.nan; p_mat.loc[c1,c2]=np.nan; continue
            if SCIPY_OK:
                r,p = pearsonr(sub[c1],sub[c2])
            else:
                r,p = sub[c1].corr(sub[c2]), np.nan
            r_mat.loc[c1,c2]=r; p_mat.loc[c1,c2]=p
    return r_mat, p_mat, n_mat

def scatter_ols_manual(data, xcol, ycol, xlabel, ylabel, np_seed=42):
    """Scatter dgn jitter visual ringan pada Y diskrit (utk keterbacaan titik yg numpuk)
    + garis OLS dihitung dari data ASLI tanpa jitter (scipy.stats.linregress, tidak
    bergantung statsmodels spt trendline='ols' bawaan plotly yg bisa crash jika
    statsmodels tidak terinstal). Warna titik berdasar kategori NPS bila kolom
    G1A_CAT tersedia di data, kalau tidak pakai warna tunggal."""
    has_cat = 'G1A_CAT' in data.columns
    cols = [xcol,ycol] + (['G1A_CAT'] if has_cat else [])
    sub = data[cols].dropna()
    fig = go.Figure()
    rng = np.random.RandomState(np_seed)
    y_jitter = sub[ycol].values.astype(float) + rng.uniform(-0.12,0.12,size=len(sub))
    if has_cat:
        for cat in ['Promoter','Passive','Detractor']:
            m = sub['G1A_CAT']==cat
            if m.sum()==0: continue
            fig.add_trace(go.Scatter(x=sub.loc[m,xcol],y=y_jitter[m.values],mode='markers', name=cat,marker=dict(color=NPS_C.get(cat),size=6,opacity=0.45), hovertemplate=f'{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{customdata:.1f}}<extra></extra>', customdata=sub.loc[m,ycol]))
    else:
        fig.add_trace(go.Scatter(x=sub[xcol],y=y_jitter,mode='markers', name=ylabel,marker=dict(color=C_XYZ,size=6,opacity=0.45), hovertemplate=f'{xlabel}: %{{x:.2f}}<br>{ylabel}: %{{customdata:.1f}}<extra></extra>', customdata=sub[ycol]))
    r_line = None
    if SCIPY_OK and len(sub)>=3 and sub[xcol].std()>0:
        slope,intercept,r_val,p_val,_ = linregress(sub[xcol],sub[ycol])
        xr = np.linspace(sub[xcol].min(),sub[xcol].max(),50)
        fig.add_trace(go.Scatter(x=xr,y=intercept+slope*xr,mode='lines',name='OLS (data asli)', line=dict(color=text_main,width=2,dash='dash'), hovertemplate='Garis OLS<extra></extra>'))
        r_line = (r_val,p_val)
    fig.update_layout(xaxis_title=xlabel,yaxis_title=ylabel)
    return fig, r_line, len(sub)

# ═══════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('data/df_clean.csv', low_memory=False)
    cm = pd.read_csv('data/col_mapping.csv').set_index('kode')['nama_panjang'].to_dict()
    np.random.seed(42)
    months = pd.date_range('2023-01', periods=12, freq='MS')
    df['Periode'] = np.random.choice([m.strftime('%Y-%m') for m in months], size=len(df))
    return df, cm

try:
    df_raw, col_map = load_data()
except Exception as e:
    st.error(f"Gagal memuat data: {e}"); st.stop()

g_nps,_,_,_ = calc_nps(df_raw['G1A_CAT'])
g_sat = df_raw['E1A'].mean()
g_loy = df_raw['F1A'].mean()

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    periods = sorted(df_raw['Periode'].unique())
    sel_per = st.select_slider("Periode", key="sl_per", options=periods, value=(periods[0], periods[-1]))
    per_range = periods[periods.index(sel_per[0]):periods.index(sel_per[1])+1]

    st.markdown("---")
    st.markdown("**Benchmark**")
    NO_KOMP_LABEL = "Tidak Ada Kompetitor"
    komp_opts   = sorted([k for k in df_raw['KOMP'].dropna().unique() if k != NO_KOMP_LABEL])
    target_komp = st.selectbox("Kompetitor:", key="sb_komp", options=["Seluruh Kompetitor"]+komp_opts)

    st.markdown("---")
    st.markdown("**Lokasi**")
    sel_prov = st.multiselect("Provinsi", key="ms_prov", options=sorted(df_raw['PROV'].dropna().unique()), help="💡 Ketik beberapa huruf untuk mencari dari daftar.")
    kpool = df_raw[df_raw['PROV'].isin(sel_prov)] if sel_prov else df_raw
    sel_kota = st.multiselect("Kab/Kota", key="ms_kota", options=sorted(kpool['KABKOTA'].dropna().unique()), help="💡 Ketik beberapa huruf untuk mencari dari daftar. Pilih Provinsi dulu untuk mempersempit pilihan.")
    cpool = kpool[kpool['KABKOTA'].isin(sel_kota)] if sel_kota else kpool
    cab_opts=sorted(cpool['CABANG'].dropna().unique())
    sel_cab = st.multiselect(f"Cabang ({len(cab_opts)} opsi)", key="ms_cab", options=cab_opts, help="💡 Ketik nama cabang untuk mencari (misal 'Bandung'). Pilih Provinsi/Kab-Kota dulu untuk mempersempit daftar ini.")

    st.markdown("---")
    with st.expander("Profil Responden"):
        sel_gender    = st.multiselect("Gender",key="ms_gd",options=sorted(df_raw['S1'].dropna().unique()))
        sel_usia      = st.multiselect("Usia",key="ms_us",options=sorted(df_raw['S2_2'].dropna().unique()))
        sel_tenure    = st.multiselect("Lama Nasabah",key="ms_tn",options=sorted(df_raw['S4'].dropna().unique()))
        sel_frek      = st.multiselect("Frekuensi",key="ms_fk",options=sorted(df_raw['S7'].dropna().unique()))
        sel_panel     = st.multiselect("Panel",key="ms_pl",options=sorted(df_raw['PANEL'].dropna().unique()))
        sel_pekerjaan = st.multiselect("Pekerjaan",key="ms_pk",options=sorted(df_raw['P4'].dropna().unique()))
        sel_pendidikan= st.multiselect("Pendidikan",key="ms_pd",options=sorted(df_raw['P3'].dropna().unique()))
        sel_status    = st.multiselect("Status Nikah",key="ms_st",options=sorted(df_raw['P1'].dropna().unique()))

    with st.expander("Perilaku Perbankan"):
        sel_bsimpan = st.multiselect("Bank Utama Simpan",key="ms_bs",options=sorted(df_raw['A1B'].dropna().unique()))
        sel_btrans  = st.multiselect("Bank Utama Transaksi",key="ms_bt",options=sorted(df_raw['A1C'].dropna().unique()))
        sel_npscat  = st.multiselect("Kategori NPS",key="ms_nc",options=['Promoter','Passive','Detractor'])

    with st.expander("Filter Skor"):
        nps_r = st.slider("NPS",0,10,(0,10),key="sl_np")
        sat_r = st.slider("Kepuasan",1,6,(1,6),key="sl_st")
        loy_r = st.slider("Loyalitas",1,6,(1,6),key="sl_ly")
    st.markdown("---")

# apply filter
df = df_raw[df_raw['Periode'].isin(per_range)].copy()
for col,sel in [('PROV',sel_prov),('KABKOTA',sel_kota),('CABANG',sel_cab),
    ('S1',sel_gender),('S2_2',sel_usia),('S4',sel_tenure),('S7',sel_frek),
    ('PANEL',sel_panel),('P4',sel_pekerjaan),('P3',sel_pendidikan),
    ('P1',sel_status),('A1B',sel_bsimpan),('A1C',sel_btrans)]:
    if sel: df = df[df[col].isin(sel)]
if sel_npscat: df = df[df['G1A_CAT'].isin(sel_npscat)]
df = df[df['G1A'].between(*nps_r)&df['E1A'].between(*sat_r)&df['F1A'].between(*loy_r)]

df_komp = df.copy() if target_komp=="Seluruh Kompetitor" else df[df['KOMP']==target_komp]
df_hk   = df_komp[df_komp['KOMP'].notna() & (df_komp['KOMP']!=NO_KOMP_LABEL)]

with st.sidebar:
    st.success(f"Total: {len(df):,} Responden aktif")
    if len(df_hk): st.info(f"Total: {len(df_hk):,} dengan Kompetitor")
    if len(df)<30: st.warning("Perhatian: Sampel < 30")

if df.empty: st.warning("Data kosong, silakan sesuaikan filter."); st.stop()

def get_dm(dfr):
    return {
      "Kantor Cabang":{"imp":[f"T_KC1_{i}" for i in range(1,36) if f"T_KC1_{i}" in dfr.columns], "xyz":[c for c in dfr.columns if c.startswith("T_KC2_") and c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116","T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and int(c.split("_")[-1])%3==2], "komp":[c for c in dfr.columns if c.startswith("T_KC2_") and c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116","T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and int(c.split("_")[-1])%3==0]},
      "Sekuriti":{"imp":[f"T_SC1_{i}" for i in range(1,16) if f"T_SC1_{i}" in dfr.columns], "xyz":[c for c in dfr.columns if c.startswith("T_SC2_") and c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1])%3==2], "komp":[c for c in dfr.columns if c.startswith("T_SC2_") and c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1])%3==0]},
      "Teller":{"imp":[f"T_TL2_{i}" for i in range(1,20) if f"T_TL2_{i}" in dfr.columns], "xyz":[c for c in dfr.columns if c.startswith("T_TL3_") and c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1])%3==2], "komp":[c for c in dfr.columns if c.startswith("T_TL3_") and c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1])%3==0]},
      "Customer Service":{"imp":[f"T_CS2_{i}" for i in range(1,24) if f"T_CS2_{i}" in dfr.columns], "xyz":[c for c in dfr.columns if c.startswith("T_CS3_") and c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1])%3==2], "komp":[c for c in dfr.columns if c.startswith("T_CS3_") and c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1])%3==0]},
      "Customer Advisor":{"imp":[f"T_CA1_{i}" for i in range(1,20) if f"T_CA1_{i}" in dfr.columns], "xyz":[c for c in dfr.columns if c.startswith("T_CA2_") and c!="T_CA2_20"], "komp":[]},
      "ATM":{"imp":[f"T_AT2_{i}" for i in range(1,19) if f"T_AT2_{i}" in dfr.columns], "xyz":[c for c in dfr.columns if c.startswith("T_AT3_") and c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1])%3==2], "komp":[c for c in dfr.columns if c.startswith("T_AT3_") and c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1])%3==0]},
    }
DM = get_dm(df_raw)

def build_ctx(dff, dhk):
    ns,pp,pv,pd2 = calc_nps(dff['G1A_CAT'])
    nk,pk,pvk,dk  = calc_nps(dhk['G1C_CAT']) if len(dhk)>0 else (0,0,0,0)
    oc = [c for c in dff.columns if c.startswith("OVR_") and "_XYZ" in c]
    om = {nama_kolom(c.replace("OVR_","").replace("_XYZ","")):round(dff[c].mean(),2) for c in oc if c in dff.columns}
    tc = [c for c in dff.columns if c.startswith("OVR_") and "_XYZ" in c and c not in ["OVR_KC_OPERASIONAL_XYZ","OVR_KC_PARKIR_XYZ","OVR_KC_BANKINGHALL_XYZ","OVR_KC_TOILET_XYZ"]]
    cs = dff.groupby('CABANG')[tc].mean().mean(axis=1) if tc else pd.Series(dtype=float)
    cs_n = dff.groupby('CABANG').size()
    cs_rel = cs[cs_n>=10]  # hanya cabang dgn sampel cukup, konsisten dgn perbaikan #2 sebelumnya
    t5 = [(idx,round(v,2),int(cs_n[idx])) for idx,v in cs_rel.nlargest(5).items()] if len(cs_rel)>0 else []
    b5 = [(idx,round(v,2),int(cs_n[idx])) for idx,v in cs_rel.nsmallest(5).items()] if len(cs_rel)>0 else []
    pc = clean_cmt(dff[dff['G1A_CAT']=='Promoter']['G1B']).head(5).tolist() if 'G1B' in dff.columns else []
    dc = clean_cmt(dff[dff['G1A_CAT']=='Detractor']['G1B']).head(5).tolist() if 'G1B' in dff.columns else []

    # Breakdown per provinsi (NPS/Kepuasan/Loyalitas) — biasanya <20 provinsi, aman utk konteks
    prov_lines = []
    if 'PROV' in dff.columns:
        pg = dff.groupby('PROV').agg(NPS=('G1A','mean'),Kepuasan=('E1A','mean'),N=('SERIAL','count')).round(2)
        prov_lines = [f"{idx}: NPS={r['NPS']}, Kepuasan={r['Kepuasan']}, N={int(r['N'])}" for idx,r in pg.iterrows()]

    # Breakdown demografi ringkas: gender, kategori usia, pendidikan -> NPS
    demo_lines = []
    for col,label in [('S1','Gender'),('S2_2','Usia'),('P3','Pendidikan'),('S7','Frekuensi Transaksi')]:
        if col in dff.columns:
            dg = dff.groupby(col)['G1A'].agg(['mean','count'])
            dg = dg[dg['count']>=10]
            if len(dg)>0:
                items = [f"{idx}(N={int(r['count'])}):{round(r['mean'],1)}" for idx,r in dg.iterrows()]
                demo_lines.append(f"{label} -> NPS: " + ", ".join(items))

    # Waktu tunggu Teller/CS: aktual vs toleransi, dan jumlah cabang yg melebihi toleransi
    wait_lines = []
    if 'PANEL' in dff.columns and 'TL5' in dff.columns:
        dft = dff[dff['PANEL'].astype(str).str.contains('Teller',case=False,na=False)].dropna(subset=['TL5','TL6'])
        if len(dft)>0:
            wait_lines.append(f"Teller: aktual rata-rata={dft['TL5'].mean():.1f} mnt (median={dft['TL5'].median():.1f}), toleransi={dft['TL6'].mean():.1f} mnt, N={len(dft)}")
        dfc = dff[dff['PANEL'].astype(str).str.contains('CS',case=False,na=False)].dropna(subset=['CS5','CS6'])
        if len(dfc)>0:
            wait_lines.append(f"CS: aktual rata-rata={dfc['CS5'].mean():.1f} mnt (median={dfc['CS5'].median():.1f}), toleransi={dfc['CS6'].mean():.1f} mnt, N={len(dfc)}")

    # Emosi positif/negatif XYZ vs kompetitor (proxy dari kolom T_I1A_* bila ada)
    emo_line = ""
    epc=[c for c in["T_I1A_2","T_I1A_5","T_I1A_8","T_I1A_11","T_I1A_14","T_I1A_17","T_I1A_20","T_I1A_23","T_I1A_26"] if c in dff.columns]
    enc=[c for c in["T_I1A_29","T_I1A_32","T_I1A_35","T_I1A_38","T_I1A_41","T_I1A_44","T_I1A_47"] if c in dff.columns]
    if epc and enc:
        emo_line = f"- Emosi Positif XYZ (skala 1-6): {dff[epc].mean(axis=1).mean():.2f} | Emosi Negatif XYZ: {dff[enc].mean(axis=1).mean():.2f}"

    ctx_parts = [
        f"DATA RINGKASAN BANK XYZ (filter aktif: {target_komp}):",
        f"- Total responden: {len(dff):,}",
        f"- NPS XYZ: {ns} (Promoter {pp}%, Passive {pv}%, Detractor {pd2}%)",
        f"- NPS Kompetitor: {nk} (Promoter {pk}%, Passive {pvk}%, Detractor {dk}%) dari N={len(dhk):,} responden yang punya data kompetitor",
        f"- Gap NPS: {round(ns-nk,1)} poin",
        f"- Kepuasan XYZ (skala 1-6): {round(dff['E1A'].mean(),2)} | Loyalitas XYZ: {round(dff['F1A'].mean(),2)}",
        f"- Skor dimensi layanan XYZ (skala 1-6): {json.dumps(om)}",
    ]
    if emo_line: ctx_parts.append(emo_line)
    if t5: ctx_parts.append(f"- Top 5 cabang terbaik (N≥10): {t5}")
    if b5: ctx_parts.append(f"- Bottom 5 cabang (N≥10): {b5}")
    if prov_lines: ctx_parts.append("- Breakdown per provinsi:\n  " + "\n  ".join(prov_lines))
    if demo_lines: ctx_parts.append("- Breakdown demografi:\n  " + "\n  ".join(demo_lines))
    if wait_lines: ctx_parts.append("- Waktu tunggu:\n  " + "\n  ".join(wait_lines))
    ctx_parts.append(f"- Sampel alasan Promoter (verbatim): {pc}")
    ctx_parts.append(f"- Sampel alasan Detractor (verbatim): {dc}")
    ctx_parts.append("\nCatatan: hanya cabang dengan N≥10 responden yang ditampilkan di ranking cabang, karena cabang dengan sampel lebih kecil rata-ratanya kurang stabil secara statistik. Bila ditanya cabang spesifik yang tidak tercantum, sampaikan bahwa data rinci cabang tersebut tidak tersedia dalam ringkasan ini.")
    return "\n".join(ctx_parts)

def call_ai(msgs, ctx):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key: return "GROQ_API_KEY belum dikonfigurasi di Streamlit Secrets."
        from groq import Groq
        client = Groq(api_key=api_key)
        sys_p = ("Kamu adalah analis Customer Experience (CX) senior ahli perbankan Indonesia. "
            "Analisis data survei kepuasan nasabah Bank XYZ di bawah dan berikan insight actionable. "
            "Jawab dalam Bahasa Indonesia profesional. Format: maksimal 4 paragraf pendek, langsung ke point.\n\n"
            "ATURAN PENTING:\n"
            "1. HANYA gunakan angka yang ada di KONTEKS DATA di bawah. Jangan mengarang atau menebak angka yang tidak tercantum.\n"
            "2. Jika ditanya sesuatu yang datanya tidak ada di konteks (misal cabang spesifik yang tidak tercantum di Top/Bottom 5), katakan dengan jujur bahwa data rinci itu tidak tersedia dalam ringkasan saat ini — jangan mengarang jawaban.\n"
            "3. Manfaatkan breakdown provinsi, demografi, dan waktu tunggu di konteks bila pertanyaan relevan dengan itu, jangan hanya mengulang angka NPS/Kepuasan/Loyalitas global.\n"
            "4. Sebutkan N (jumlah responden) yang relevan saat menyebut suatu angka, terutama bila N kecil sehingga butuh kehati-hatian interpretasi.\n\n"
            f"KONTEKS DATA:\n{ctx}")
        response = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role":"system","content":sys_p}] + msgs, max_tokens=900, temperature=0.4)
        return response.choices[0].message.content
    except Exception as e: return f"Gagal menghubungi AI: {str(e)}"

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<div style='text-align:center;padding:16px 0 24px; margin-bottom:12px'>
<h1 style='font-weight:900;letter-spacing:-1.5px;color:{text_main}!important;font-size:42px;margin:0;'>
Dashboard Bank XYZ
</h1>
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════
tabs = st.tabs(["Home","Kinerja Layanan","Kompetitor",
    "Touchpoint","Emosi & Loyalitas","Digitalisasi",
    "Clustering","Profil & Segmen","Voice of Customer","AI Assistant"])
t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = tabs

# ═══════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE
# ═══════════════════════════════════════════════════════
with t1:
    ns,pp,pv,pd2 = calc_nps(df['G1A_CAT'])
    nk,pk,pvk,dk = calc_nps(df_hk['G1C_CAT']) if len(df_hk)>0 else (0,0,0,0)
    sat  = df['E1A'].mean(); loy = df['F1A'].mean()
    sat_k = df_hk['E1B'].mean() if len(df_hk)>0 else np.nan
    gap   = ns - nk
    n_tot = len(df); n_hk = len(df_hk)

    color_ns = "green" if ns >= g_nps else ("red" if ns < g_nps else "black")
    color_gap = "green" if gap > 0 else ("red" if gap < 0 else "black")
    color_sat = "green" if sat >= g_sat else ("red" if sat < g_sat else "black")
    color_loy = "green" if loy >= g_loy else ("red" if loy < g_loy else "black")

    sh("Key Performance Indicators")
    k = st.columns(8)
    k[0].markdown(card("NPS Score XYZ",  f"{ns:.0f}", color_ns, f"N={n_tot:,} · P:{pp:.0f}% D:{pd2:.0f}%",delta=ns-g_nps), unsafe_allow_html=True)
    k[1].markdown(card("NPS Kompetitor", f"{nk:.0f}","black", f"N={n_hk:,} · P:{pk:.0f}% D:{dk:.0f}%"), unsafe_allow_html=True)
    k[2].markdown(card("Gap NPS",f"+{gap:.0f}" if gap>=0 else f"{gap:.0f}", color_gap, "XYZ − Kompetitor"), unsafe_allow_html=True)
    k[3].markdown(card("Kepuasan XYZ",   f"{sat:.2f}", color_sat, f"N={n_tot:,} · Skala 1–6",delta=sat-g_sat), unsafe_allow_html=True)
    k[4].markdown(card("Kepuasan Komp", f"{sat_k:.2f}" if not np.isnan(sat_k) else "N/A","black", f"N={n_hk:,} · Skala 1–6"), unsafe_allow_html=True)
    k[5].markdown(card("Loyalitas XYZ",  f"{loy:.2f}", color_loy, f"N={n_tot:,} · Skala 1–6",delta=loy-g_loy), unsafe_allow_html=True)
    k[6].markdown(card("Promoter XYZ",   f"{pp:.0f}%","green", f"{int(df['G1A_CAT'].eq('Promoter').sum())} dari {n_tot:,} orang"), unsafe_allow_html=True)
    k[7].markdown(card("Total Responden",f"{n_tot:,}","black", f"{n_hk:,} dengan data kompetitor"), unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if gap>50:   ib(f"XYZ unggul sangat signifikan — gap NPS **{gap:.0f} poin** dari {n_tot:,} responden. Promoter {pp:.0f}% vs kompetitor {pk:.0f}%.")
    elif gap>20: ib(f"XYZ unggul **{gap:.0f} poin** NPS ({n_tot:,} responden). Pertahankan momentum dan tingkatkan dimensi dengan gap terkecil.")
    else:        ib(f"Gap NPS hanya **{gap:.0f} poin** ({n_tot:,} responden) — persaingan sangat ketat. Perlu diferensiasi lebih tajam.")

    st.markdown("<br>",unsafe_allow_html=True)
    r1,r2 = st.columns([1,2.2])
    with r1:
        sh("NPS Composition")
        for cat_s, lbl, nv, nn in [(df['G1A_CAT'],"Bank XYZ",ns,n_tot), (df_hk['G1C_CAT'] if len(df_hk)>0 else pd.Series(dtype=str), target_komp[:20], nk, n_hk)]:
            if len(cat_s)>0:
                cp = cat_s.value_counts().reset_index(); cp.columns=['K','N']
                fd = px.pie(cp,values='N',names='K',hole=0.65,color='K', color_discrete_map=NPS_C)
                fd.update_traces(textposition='inside',textinfo='percent+label', marker=dict(line=dict(color=bg_panel,width=2)), hovertemplate='<b>%{label}</b><br>Jumlah: %{value} responden<br>Persentase: %{percent}<extra></extra>')
                fd.add_annotation(text=f"{lbl}<br><b>{nv:.0f}</b><br><span style='font-size:11px'>N={nn:,}</span>", x=0.5,y=0.5,showarrow=False,font=dict(size=14,color=text_main))
                fd.update_layout(height=260, margin=dict(t=22,b=6,l=0,r=0), showlegend=False)
                st.plotly_chart(elo(fd),use_container_width=True)

    with r2:
        sh("Scorecard Dimensi XYZ vs Kompetitor")
        sc_rows=[
            ("Kantor Cabang",   "OVR_KC_XYZ",           "OVR_KC_KOM"),
            ("  ↳ Operasional", "OVR_KC_OPERASIONAL_XYZ","OVR_KC_OPERASIONAL_KOM"),
            ("  ↳ Parkir",      "OVR_KC_PARKIR_XYZ",    "OVR_KC_PARKIR_KOM"),
            ("  ↳ Banking Hall","OVR_KC_BANKINGHALL_XYZ","OVR_KC_BANKINGHALL_KOM"),
            ("  ↳ Toilet",      "OVR_KC_TOILET_XYZ",    "OVR_KC_TOILET_KOM"),
            ("Sekuriti",        "OVR_SEKURITI_XYZ",     "OVR_SEKURITI_KOM"),
            ("Teller",          "OVR_TELLER_XYZ",       "OVR_TELLER_KOM"),
            ("Customer Service","OVR_CS_XYZ",           "OVR_CS_KOM"),
            ("Customer Advisor","OVR_CA_XYZ",           None),
            ("Sarana Elektronik","OVR_SARANA_XYZ",      None),
            ("ATM",             "OVR_ATM_XYZ",          "OVR_ATM_KOM"),
        ]
        rows=[]
        for lb,cx,ck in sc_rows:
            xv = df[cx].mean() if cx in df.columns else np.nan
            kv = df_hk[ck].mean() if ck and ck in df_hk.columns and len(df_hk)>0 else np.nan
            nx = df[cx].notna().sum() if cx in df.columns else 0
            nk2= df_hk[ck].notna().sum() if ck and ck in df_hk.columns and len(df_hk)>0 else 0
            gp = xv-kv if not(np.isnan(xv) or np.isnan(kv)) else np.nan
            rows.append({"Dimensi":lb,"XYZ":round(xv,2), "Kompetitor":round(kv,2) if not np.isnan(kv) else None, "Gap":round(gp,2) if not np.isnan(gp) else None, "N_XYZ":nx,"N_Komp":nk2})
        sdf = pd.DataFrame(rows)
        fig_sc=go.Figure()
        fig_sc.add_trace(go.Bar(name='Bank XYZ',y=sdf['Dimensi'],x=sdf['XYZ'], orientation='h',marker_color=C_XYZ, text=sdf['XYZ'],texttemplate='%{x:.2f}',textposition='outside', customdata=sdf[['XYZ','Kompetitor','Gap','N_XYZ']].values, hovertemplate='<b>%{y}</b><br>XYZ: %{customdata[0]:.2f}<br>Komp: %{customdata[1]}<br>Gap: %{customdata[2]}<br>N responden: %{customdata[3]:.0f}<extra></extra>'))
        fig_sc.add_trace(go.Bar(name='Kompetitor',y=sdf['Dimensi'], x=pd.to_numeric(sdf['Kompetitor'],errors='coerce'), orientation='h',marker_color=C_KOMP,opacity=0.85, customdata=sdf[['Kompetitor','N_Komp']].values, hovertemplate='<b>%{y}</b><br>Kompetitor: %{customdata[0]:.2f}<br>N responden: %{customdata[1]:.0f}<extra></extra>'))
        fig_sc.update_layout(barmode='group',xaxis_range=[4,6.8],height=520)
        st.plotly_chart(elo(fig_sc,f"Skor Rata-rata per Dimensi (N≈{n_tot:,}, bervariasi per dimensi — lihat hover)"),use_container_width=True)

    r3,r4=st.columns(2)
    with r3:
        sh("Top & Bottom 5 Cabang")
        MIN_N_CABANG=15
        oc=[c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c and c not in ["OVR_KC_OPERASIONAL_XYZ","OVR_KC_PARKIR_XYZ","OVR_KC_BANKINGHALL_XYZ","OVR_KC_TOILET_XYZ"]]
        if oc:
            cs2=df.groupby('CABANG').agg(Skor=(oc[0],'mean'),N=('SERIAL','count')).reset_index()
            if len(oc)>1: cs2['Skor']=df.groupby('CABANG')[oc].mean().mean(axis=1).values
            cs2_rel=cs2[cs2['N']>=MIN_N_CABANG]
            n_excluded_cab=len(cs2)-len(cs2_rel)
            if len(cs2_rel)>=2:
                tb=pd.concat([cs2_rel.nlargest(5,'Skor').assign(S='Top 5'), cs2_rel.nsmallest(5,'Skor').assign(S='Bottom 5')])
                ftb=px.bar(tb,x='Skor',y='CABANG',color='S',orientation='h',text='Skor', color_discrete_map={'Top 5':C_XYZ,'Bottom 5':C_KOMP})
                ftb.update_traces(texttemplate='%{x:.2f}',textposition='outside', customdata=tb['N'].values, hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<br>N responden: %{customdata}<extra></extra>')
                ftb.update_xaxes(range=[4.5,6.6])
                st.plotly_chart(elo(ftb,h=380),use_container_width=True)
                st.caption(f"Hanya cabang dengan N≥{MIN_N_CABANG} responden yang dirank ({n_excluded_cab} cabang dengan sampel lebih kecil dikeluarkan dari ranking ini karena rata-ratanya kurang stabil).")
            else:
                st.info(f"Tidak cukup cabang dengan N≥{MIN_N_CABANG} responden untuk membuat ranking yang andal pada filter saat ini.")

    with r4:
        sh("Matriks Korelasi Antar Dimensi")
        cm_map={'NPS':'G1A','Kepuasan':'E1A','Loyalitas':'F1A', 'Teller':'OVR_TELLER_XYZ','CS':'OVR_CS_XYZ', 'ATM':'OVR_ATM_XYZ','Sekuriti':'OVR_SEKURITI_XYZ','KC':'OVR_KC_XYZ'}
        vm={k:v for k,v in cm_map.items() if v in df.columns}
        cdf_=df[list(vm.values())].copy(); cdf_.columns=list(vm.keys())
        r_mat,p_mat,n_mat=corr_with_stats(cdf_)
        sig_mask=(p_mat<0.05) | (np.eye(len(r_mat.columns),dtype=bool))
        fco=px.imshow(r_mat,text_auto=".2f", color_continuous_scale='Greens',aspect='auto',zmin=-1,zmax=1)
        customdata_corr=np.dstack([p_mat.values, n_mat.values, sig_mask.values])
        fco.update_traces(customdata=customdata_corr, hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Korelasi (r): %{z:.3f}<br>' + ('p-value: %{customdata[0]:.4f}<br>' if SCIPY_OK else '') + 'N (pairwise): %{customdata[1]:.0f}<extra></extra>')
        st.plotly_chart(elo(fco,f"Korelasi Pearson antar Dimensi (N≈{n_tot:,}, p-value & N tepat di hover)",h=380),use_container_width=True)
        if SCIPY_OK:
            n_not_sig=int((~sig_mask.values).sum()/2 - len(r_mat)/2) if len(r_mat)>0 else 0
            st.caption("Korelasi dihitung Pearson dengan N pairwise (per pasangan kolom, exclude data kosong). " + (f"Beberapa pasangan tidak signifikan secara statistik (p≥0.05) — perlakukan koefisiennya sebagai indikatif saja." if (~sig_mask.values).any() else "Semua pasangan signifikan secara statistik (p<0.05)."))
        else:
            st.caption("Korelasi dihitung Pearson dengan N pairwise (per pasangan kolom, exclude data kosong). Install scipy untuk menampilkan p-value signifikansi.")

    sh("Tren NPS, Kepuasan & Loyalitas per Periode")
    tren=df.groupby('Periode').agg(NPS=('G1A','mean'),Kepuasan=('E1A','mean'), Loyalitas=('F1A','mean'),N=('SERIAL','count')).reset_index()
    ftr=go.Figure()
    for cn,cc,nl in [('NPS',C_XYZ,'NPS'),('Kepuasan','#34D399','Kepuasan'),('Loyalitas','#FBBF24','Loyalitas')]:
        ftr.add_trace(go.Scatter(x=tren['Periode'],y=tren[cn],mode='lines+markers', name=nl,line=dict(color=cc,width=2.5),marker=dict(size=7), customdata=tren['N'], hovertemplate=f'<b>{nl}</b><br>Periode: %{{x}}<br>Nilai: %{{y:.2f}}<br>N responden: %{{customdata}}<extra></extra>'))
    st.plotly_chart(elo(ftr,"",360),use_container_width=True)

    sh("NPS per Provinsi")
    pn=df.groupby('PROV').agg(NPS=('G1A','mean'),Kepuasan=('E1A','mean'), Loyalitas=('F1A','mean'),N=('SERIAL','count')).reset_index().sort_values('NPS')
    fpn=px.bar(pn,x='NPS',y='PROV',orientation='h', color='NPS',color_continuous_scale='Greens',text='NPS')
    fpn.update_traces(texttemplate='%{x:.1f}',textposition='outside', customdata=pn[['Kepuasan','Loyalitas','N']].values, hovertemplate='<b>%{y}</b><br>NPS: %{x:.2f}<br>Kepuasan: %{customdata[0]:.2f}<br>Loyalitas: %{customdata[1]:.2f}<br>N responden: %{customdata[2]}<extra></extra>')
    fpn.update_xaxes(range=[7,11])
    st.plotly_chart(elo(fpn,h=max(360,len(pn)*34)),use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — KINERJA LAYANAN
# ═══════════════════════════════════════════════════════
with t2:
    sh("Heatmap Kinerja Cabang")
    hm1,hm2=st.columns([2,1])
    with hm1:
        oa=[c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c]
        ol={c:c.replace("OVR_","").replace("_XYZ","").replace("_"," ").title() for c in oa}
        sel_hm=st.multiselect("Dimensi heatmap:",key="ms_hm",options=list(ol.keys()), default=oa,format_func=lambda x:ol[x])
    with hm2:
        mr=st.slider("Min. responden per cabang:",3,30,15,key="sl_mr")
        st.caption("Default 15 — median N per cabang di data ini sekitar 14, di bawah itu rata-rata cenderung tidak stabil.")
        shm=st.selectbox("Urutkan:",key="sb_shm",options=["Rata-rata"]+[ol[c] for c in sel_hm if c in ol])

    if sel_hm:
        hf=df.groupby('CABANG').filter(lambda x:len(x)>=mr)
        n_per_cab=hf.groupby('CABANG').size().rename('N_Resp')
        hd=hf.groupby('CABANG')[sel_hm].mean().round(2)
        hd.columns=[ol[c] for c in hd.columns]
        hd=hd.loc[(hd.mean(axis=1).sort_values(ascending=False).index if shm=="Rata-rata" else hd.sort_values(shm,ascending=False).index)]
        fhm=px.imshow(hd,text_auto=".2f",color_continuous_scale='RdYlGn',aspect='auto',zmin=4,zmax=6)
        fhm.update_traces(hovertemplate='Cabang: <b>%{y}</b><br>Dimensi: <b>%{x}</b><br>Skor: <b>%{z:.2f}</b><extra></extra>')
        fhm.update_layout(height=max(400,len(hd)*22))
        st.plotly_chart(elo(fhm,"Heatmap Skor OVR (Merah=Rendah, Hijau=Tinggi)"),use_container_width=True)
        if len(hd)>0:
            wb=hd.mean(axis=1).idxmax(); ww=hd.mean(axis=1).idxmin()
            nb=n_per_cab.get(wb,0); nw=n_per_cab.get(ww,0)
            ib(f"Performa terbaik: **{wb}** (skor {hd.loc[wb].mean():.2f}, N={nb}). Perlu perhatian: **{ww}** (skor {hd.loc[ww].mean():.2f}, N={nw}).")

    st.markdown("---")
    sh("Drill-Down Item Level per Dimensi")
    dc1,dc2,dc3=st.columns(3)
    with dc1: sel_dim=st.selectbox("Dimensi:",key="sb_dim2",options=list(DM.keys()))
    with dc2: dm2=st.radio("Tampilan:",["Bar Chart","Scatter IPA"],horizontal=True,key="rd_dm2")
    with dc3:
        sel_c2=st.selectbox("Filter Cabang:",key="sb_c2", options=["Semua"]+sorted(df['CABANG'].dropna().unique().tolist()), help="💡 Ketik nama cabang untuk mencari.")

    dfd=df if sel_c2=="Semua" else df[df['CABANG']==sel_c2]
    dfdk=df_hk if sel_c2=="Semua" else df_hk[df_hk['CABANG']==sel_c2]
    n_dfd=len(dfd); n_dfdk=len(dfdk)
    di2=DM[sel_dim]
    ic2=[c for c in di2["imp"] if c in dfd.columns]
    xc2=[c for c in di2["xyz"] if c in dfd.columns]
    kc2=[c for c in di2["komp"] if c in dfdk.columns]
    mn2=min(len(ic2),len(xc2))

    if mn2>0:
        lb2=[slbl(c,col_map,42) for c in ic2[:mn2]]
        iv2=[dfd[c].mean() for c in ic2[:mn2]]
        xv2=[dfd[c].mean() for c in xc2[:mn2]]
        kv2=[dfdk[c].mean() if c in dfdk.columns else np.nan for c in kc2[:mn2]] if kc2 else []
        n2=[dfd[c].notna().sum() for c in ic2[:mn2]]

        if dm2=="Bar Chart":
            fdd=go.Figure()
            fdd.add_trace(go.Bar(name='Tingkat Kepentingan',x=iv2,y=lb2,orientation='h', marker_color=C_IMP,opacity=0.75, customdata=n2, hovertemplate='<b>%{y}</b><br>Kepentingan: %{x:.2f}<br>N: %{customdata}<extra></extra>'))
            fdd.add_trace(go.Bar(name='Kepuasan XYZ',x=xv2,y=lb2,orientation='h', marker_color=C_XYZ, customdata=n2, hovertemplate='<b>%{y}</b><br>Kepuasan XYZ: %{x:.2f}<br>N: %{customdata}<extra></extra>'))
            if kv2:
                fdd.add_trace(go.Bar(name=f'Kepuasan {target_komp}',x=kv2,y=lb2,orientation='h', marker_color=C_KOMP,opacity=0.85, customdata=[dfdk[c].notna().sum() if c in dfdk.columns else 0 for c in kc2[:mn2]], hovertemplate=f'<b>%{{y}}</b><br>Kepuasan Komp: %{{x:.2f}}<br>N: %{{customdata}}<extra></extra>'))
            fdd.update_layout(barmode='group',xaxis_range=[3,6.8], height=max(420,mn2*30),yaxis=dict(automargin=True,tickfont=dict(size=11)))
            st.plotly_chart(elo(fdd,f"Kepentingan vs Kepuasan — {sel_dim} (N={n_dfd:,})"), use_container_width=True)

            sh("Gap: Kepuasan − Kepentingan")
            gd2=pd.DataFrame({'Item':lb2,'Gap':[x-i for x,i in zip(xv2,iv2)],'N':n2}).sort_values('Gap')
            gd2['W']=np.where(gd2['Gap']<0,C_KOMP,C_XYZ)
            fg2=px.bar(gd2,x='Gap',y='Item',orientation='h',text='Gap')
            fg2.update_traces(marker_color=gd2['W'],texttemplate='%{x:.2f}',textposition='outside', customdata=gd2['N'].values, hovertemplate='<b>%{y}</b><br>Gap: %{x:.3f}<br>N: %{customdata}<extra></extra>')
            fg2.update_layout(height=max(380,mn2*28),yaxis=dict(automargin=True,tickfont=dict(size=11)))
            st.plotly_chart(elo(fg2),use_container_width=True)
            worst=gd2.iloc[0]
            if worst['Gap']<0: ib(f"Item paling kritis: **{worst['Item']}** (gap {worst['Gap']:.2f}, N={worst['N']}) — kepentingan tinggi, kepuasan rendah.")

        else:
            idf2=pd.DataFrame({'Item':lb2,'Kepentingan':iv2,'Kepuasan':xv2,'N':n2})
            mi2,ms2=np.nanmean(iv2),np.nanmean(xv2)
            BUF2=0.05*(np.nanmax(xv2)-np.nanmin(xv2)) if np.nanmax(xv2)>np.nanmin(xv2) else 0
            def q2(r):
                if abs(r['Kepentingan']-mi2)<=BUF2 and abs(r['Kepuasan']-ms2)<=BUF2: return "Netral"
                if r['Kepentingan']>=mi2 and r['Kepuasan']<ms2: return "Perbaiki"
                elif r['Kepentingan']>=mi2: return "Pertahankan"
                elif r['Kepuasan']<ms2: return "Rendah"
                else: return "Berlebihan"
            idf2['Kuadran']=idf2.apply(q2,axis=1)
            qc2={"Perbaiki":C_KOMP,"Pertahankan":"#10B981","Rendah":C_IMP,"Berlebihan":"#F59E0B","Netral":"#9CA3AF"}
            
            fi2=px.scatter(idf2,x='Kepuasan',y='Kepentingan', color='Kuadran',color_discrete_map=qc2)
            if kv2:
                fi2.add_trace(go.Scatter(x=kv2,y=iv2,mode='markers',name=target_komp,text=lb2, marker=dict(size=10,symbol='x',color=C_KOMP,line=dict(width=2)), hovertemplate='<b>%{text}</b><br>Kepuasan Komp: %{x:.2f}<extra></extra>'))
            fi2.update_traces(marker=dict(size=10), customdata=idf2[['Item','N']].values, hovertemplate='<b>%{customdata[0]}</b><br>Kepentingan: %{y:.2f}<br>Kepuasan: %{x:.2f}<br>N: %{customdata[1]}<extra></extra>', selector=dict(mode='markers+text'))
            fi2.add_vline(x=ms2,line_dash="dash",line_color=border_col)
            fi2.add_hline(y=mi2,line_dash="dash",line_color=border_col)
            fi2.update_layout(height=560)
            st.plotly_chart(elo(fi2,f"IPA Matrix — {sel_dim} (N={n_dfd:,})",legend_below=True),use_container_width=True)

            sh("Prioritas per Kuadran")
            qcols=st.columns(5)
            for col_q,qn,qcv in zip(qcols, ["Perbaiki","Pertahankan","Rendah","Berlebihan","Netral"], [C_KOMP,"#10B981",C_IMP,"#F59E0B","#9CA3AF"]):
                its=idf2[idf2['Kuadran']==qn]['Item'].tolist()
                if not its: continue
                with col_q:
                    items_html="".join([f"<div style='color:{text_main}!important;font-size:13px;padding:4px 0;border-bottom:1px solid {border_col}'>{it}</div>" for it in its])
                    st.markdown(f"<div style='background:{qcv}15;border:1px solid {qcv}40;border-radius:10px;padding:12px 14px;min-height:80px'><div style='color:{qcv}!important;font-weight:800;font-size:13px;margin-bottom:10px'>{qn}</div>{items_html}</div>", unsafe_allow_html=True)

    st.markdown("---")
    sh("Analisis Waktu Tunggu")
    st.caption("Mean waktu tunggu sensitif terhadap outlier ekstrem (misal antrian tidak biasa); median ditampilkan sebagai pembanding yang lebih tahan outlier.")
    wt1,wt2,wt3=st.columns(3)
    with wt1:
        dft2=df[df['PANEL']=='Teller'].dropna(subset=['TL5','TL6'])
        if len(dft2)>0:
            wd=pd.DataFrame({'Metrik':['Aktual (mean)','Aktual (median)','Toleransi'],'Menit':[dft2['TL5'].mean(),dft2['TL5'].median(),dft2['TL6'].mean()]})
            fw=px.bar(wd,x='Metrik',y='Menit',color='Metrik',text='Menit', color_discrete_map={'Aktual (mean)':C_KOMP,'Aktual (median)':'#F59E0B','Toleransi':C_XYZ})
            fw.update_traces(texttemplate='%{y:.1f} mnt',textposition='outside', hovertemplate='<b>%{x}</b><br>%{y:.2f} menit<br>' f'Berdasarkan {len(dft2):,} responden Teller<extra></extra>')
            fw.update_yaxes(range=[0,dft2['TL6'].mean()*1.7])
            st.plotly_chart(elo(fw,f"Teller: Aktual vs Toleransi (N={len(dft2):,})",legend_below=True),use_container_width=True)
            gap_mm=dft2['TL5'].mean()-dft2['TL5'].median()
            ib(f"Teller: **{dft2['TL5'].mean():.1f} mnt** rata-rata (median {dft2['TL5'].median():.1f} mnt, toleransi {dft2['TL6'].mean():.1f} mnt, N={len(dft2):,})." + (f" Selisih mean-median {gap_mm:.1f} mnt mengindikasikan ada outlier waktu tunggu yang menarik rata-rata ke atas." if gap_mm>2 else ""))
    with wt2:
        dfc2=df[df['PANEL']=='CS'].dropna(subset=['CS5','CS6'])
        if len(dfc2)>0:
            wd2=pd.DataFrame({'Metrik':['Aktual (mean)','Aktual (median)','Toleransi'],'Menit':[dfc2['CS5'].mean(),dfc2['CS5'].median(),dfc2['CS6'].mean()]})
            fw2=px.bar(wd2,x='Metrik',y='Menit',color='Metrik',text='Menit', color_discrete_map={'Aktual (mean)':C_KOMP,'Aktual (median)':'#F59E0B','Toleransi':C_XYZ})
            fw2.update_traces(texttemplate='%{y:.1f} mnt',textposition='outside', hovertemplate='<b>%{x}</b><br>%{y:.2f} menit<br>' f'Berdasarkan {len(dfc2):,} responden CS<extra></extra>')
            fw2.update_yaxes(range=[0,dfc2['CS6'].mean()*1.7])
            st.plotly_chart(elo(fw2,f"CS: Aktual vs Toleransi (N={len(dfc2):,})",legend_below=True),use_container_width=True)
    with wt3:
        JAM_LABEL_SHORT={
            'Pagi hari jam 8.00 - 10.00':'08:00-10:00',
            'Pagi hari jam 10.00 - 12.00':'10:00-12:00',
            'Pada saat jam makan siang jam 12.00 - 13.00':'12:00-13:00',
            'Pada siang hari jam 13.00-14.00':'13:00-14:00',
            'Sore hari jam 14.00-15.00':'14:00-15:00',
        }
        JAM_ORDER=list(JAM_LABEL_SHORT.values())
        jam_excl={'tidak mengisi','tidak ada','-','n/a','na',''}
        tl1_valid=df['TL1'].dropna()[~df['TL1'].dropna().str.strip().str.lower().isin(jam_excl)]
        cs1_valid=df['CS1'].dropna()[~df['CS1'].dropna().str.strip().str.lower().isin(jam_excl)]
        jt=tl1_valid.value_counts().reset_index(); jt.columns=['Jam','Teller']
        jc=cs1_valid.value_counts().reset_index(); jc.columns=['Jam','CS']
        jd=pd.merge(jt,jc,on='Jam',how='outer').fillna(0)
        jd['Jam_Short']=jd['Jam'].map(JAM_LABEL_SHORT).fillna(jd['Jam'])
        jd['_ord']=jd['Jam_Short'].apply(lambda x: JAM_ORDER.index(x) if x in JAM_ORDER else 99)
        jd=jd.sort_values('_ord')
        fj=go.Figure()
        fj.add_trace(go.Bar(name='Teller',x=jd['Jam_Short'],y=jd['Teller'],marker_color=C_XYZ, customdata=jd['Jam'], hovertemplate='<b>%{customdata}</b><br>Teller: %{y:.0f} responden<extra></extra>'))
        fj.add_trace(go.Bar(name='CS',x=jd['Jam_Short'],y=jd['CS'],marker_color=C_KOMP, customdata=jd['Jam'], hovertemplate='<b>%{customdata}</b><br>CS: %{y:.0f} responden<extra></extra>'))
        fj.update_layout(barmode='group',xaxis_tickangle=-20,height=300)
        st.plotly_chart(elo(fj,"Distribusi Jam Sibuk"),use_container_width=True)
        if len(jt)>0:
            top_jam=jt.iloc[0]
            top_jam_short=JAM_LABEL_SHORT.get(top_jam['Jam'],top_jam['Jam'])
            ib(f"Jam paling sibuk Teller: **{top_jam_short}** ({top_jam['Teller']:.0f} responden).")

    sh("Waktu Tunggu per Cabang")
    wtp=st.radio("Panel:",["Teller","CS"],horizontal=True,key="rd_wtp2")
    if wtp=="Teller": wcd=df[df['PANEL']=='Teller'].groupby('CABANG').agg(Aktual=('TL5','mean'),Toleransi=('TL6','mean'),N=('SERIAL','count')).reset_index()
    else: wcd=df[df['PANEL']=='CS'].groupby('CABANG').agg(Aktual=('CS5','mean'),Toleransi=('CS6','mean'),N=('SERIAL','count')).reset_index()
    wcd=wcd[wcd['N']>=5].sort_values('Aktual',ascending=False)
    if len(wcd)>0:
        fwc=go.Figure()
        fwc.add_trace(go.Bar(name='Aktual',x=wcd['CABANG'],y=wcd['Aktual'],marker_color=C_KOMP, customdata=wcd['N'], hovertemplate='<b>%{x}</b><br>Aktual: %{y:.1f} mnt<br>N responden: %{customdata}<extra></extra>'))
        fwc.add_trace(go.Bar(name='Toleransi',x=wcd['CABANG'],y=wcd['Toleransi'],marker_color=C_XYZ,opacity=0.7, customdata=wcd['N'], hovertemplate='<b>%{x}</b><br>Toleransi: %{y:.1f} mnt<br>N responden: %{customdata}<extra></extra>'))
        fwc.update_layout(barmode='overlay',xaxis_tickangle=-30,height=380)
        st.plotly_chart(elo(fwc,f"Waktu Tunggu {wtp} per Cabang"),use_container_width=True)
        n_small=(wcd['N']<10).sum()
        if n_small>0: st.caption(f"⚠️ {n_small} dari {len(wcd)} cabang di atas punya N<10 responden — rata-rata waktu tunggunya rawan ditarik oleh satu-dua kasus ekstrem, interpretasikan dengan hati-hati.")
        ot=wcd[wcd['Aktual']>wcd['Toleransi']]
        if len(ot)>0: ib(f"**{len(ot)} cabang** melebihi toleransi {wtp}: {', '.join(ot['CABANG'].tolist()[:5])}.")

# ═══════════════════════════════════════════════════════
# TAB 3 — BRAND & KOMPETITOR
# ═══════════════════════════════════════════════════════
with t3:
    bic=[f"T_C1A_{i}" for i in range(1,25) if f"T_C1A_{i}" in df.columns]
    bxc=sorted([c for c in df.columns if c.startswith("T_C1B_") and int(c.split("_")[-1])%3==2],key=lambda x:int(x.split("_")[-1]))
    bkc=sorted([c for c in df.columns if c.startswith("T_C1B_") and int(c.split("_")[-1])%3==0],key=lambda x:int(x.split("_")[-1]))
    blb_full=[slbl(c,col_map,55) for c in bic]
    blb=[slbl(c,col_map,50) for c in bic]
    seen={}; blb_unique=[]
    for l in blb:
        if l in seen: seen[l]+=1; blb_unique.append(f"{l} ({seen[l]})")
        else: seen[l]=0; blb_unique.append(l)
    blb=blb_unique
    biv=[df[c].mean() for c in bic]; bxv=[df[c].mean() for c in bxc[:len(bic)]]; bkv=[df_hk[c].mean() if c in df_hk.columns else np.nan for c in bkc[:len(bic)]]
    bn_xyz=[df[c].notna().sum() for c in bxc[:len(bic)]]; bn_komp=[df_hk[c].notna().sum() if c in df_hk.columns else 0 for c in bkc[:len(bic)]]

    sh("Perbandingan 24 Atribut Brand — XYZ vs Kompetitor")
    brand_view=st.radio("Tampilan:",["Bar Comparison","Radar Chart"],horizontal=True,key="rd_brand3")
    st.markdown("<br>",unsafe_allow_html=True)

    if brand_view=="Bar Comparison":
        fbc=go.Figure()
        fbc.add_trace(go.Bar(name='Kepentingan', x=biv, y=blb, orientation='h', marker_color=C_IMP, opacity=0.75, customdata=[[f,n] for f,n in zip(blb_full,bn_xyz)], hovertemplate='<b>%{customdata[0]}</b><br>Kepentingan: %{x:.2f}<br>N: %{customdata[1]}<extra></extra>'))
        fbc.add_trace(go.Bar(name='Kepuasan XYZ', x=bxv, y=blb, orientation='h', marker_color=C_XYZ, customdata=[[f,n] for f,n in zip(blb_full,bn_xyz)], hovertemplate='<b>%{customdata[0]}</b><br>Kepuasan XYZ: %{x:.2f}<br>N: %{customdata[1]}<extra></extra>'))
        fbc.add_trace(go.Bar(name=f'Kepuasan {target_komp}', x=[v if not np.isnan(v) else None for v in bkv], y=blb, orientation='h', marker_color=C_KOMP, customdata=[[f,n] for f,n in zip(blb_full,bn_komp)], hovertemplate=f'<b>%{{customdata[0]}}</b><br>Kepuasan Komp: %{{x:.2f}}<br>N: %{{customdata[1]}}<extra></extra>'))
        fbc.update_layout(barmode='group', xaxis_range=[3,6.8], height=max(900,len(blb)*36), yaxis=dict(automargin=True, tickfont=dict(size=11)), margin=dict(t=46,b=14,l=220,r=20))
        st.plotly_chart(elo(fbc, f"Kepentingan vs Kepuasan XYZ vs {target_komp}"), use_container_width=True)
    else:
        st.info("Radar chart menampilkan 10 atribut dengan gap terbesar. Gunakan 'Bar Comparison' untuk melihat semua 24 atribut.")
        gap_abs=[abs(x-k) if not np.isnan(k) else 0 for x,k in zip(bxv,bkv)]
        top10_idx=sorted(range(len(gap_abs)),key=lambda i:gap_abs[i],reverse=True)[:10]
        blb_r10=[blb[i] for i in top10_idx]; bxv_r10=[bxv[i] for i in top10_idx]; bkv_r10=[v if not np.isnan(v) else 0 for v in [bkv[i] for i in top10_idx]]
        fr=go.Figure()
        fr.add_trace(go.Scatterpolar(r=bxv_r10,theta=blb_r10,fill='toself',name='Bank XYZ', line_color=C_XYZ,fillcolor='rgba(16, 185, 129, 0.15)', hovertemplate='<b>%{theta}</b><br>XYZ: %{r:.2f}<extra></extra>'))
        fr.add_trace(go.Scatterpolar(r=bkv_r10,theta=blb_r10,fill='toself',name=target_komp, line_color=C_KOMP,fillcolor='rgba(244, 63, 94, 0.10)', hovertemplate=f'<b>%{{theta}}</b><br>{target_komp}: %{{r:.2f}}<extra></extra>'))
        fr.update_layout(polar=dict(bgcolor=BG, radialaxis=dict(visible=True,range=[3,6.5],gridcolor=GRID, tickfont=dict(color=text_muted,size=11)), angularaxis=dict(tickfont=dict(color=text_main,size=12), direction='clockwise')), legend=dict(orientation="h",y=-0.15), height=600,margin=dict(t=40,b=90,l=90,r=90))
        st.plotly_chart(elo(fr, "Radar 10 Atribut Gap Terbesar (XYZ vs Kompetitor)"), use_container_width=True)

    sh("IPA Matrix 24 Atribut Brand")
    bdf=pd.DataFrame({'Label':blb,'Label_Full':blb_full,'Kepentingan':biv,'Kepuasan':bxv,'N':bn_xyz})
    mb,sb2_v=np.nanmean(biv),np.nanmean(bxv)
    BUFB=0.05*(np.nanmax(bxv)-np.nanmin(bxv)) if np.nanmax(bxv)>np.nanmin(bxv) else 0
    def bq(r):
        if abs(r['Kepentingan']-mb)<=BUFB and abs(r['Kepuasan']-sb2_v)<=BUFB: return "Netral"
        if r['Kepentingan']>=mb and r['Kepuasan']<sb2_v: return "Perbaiki"
        elif r['Kepentingan']>=mb: return "Pertahankan"
        elif r['Kepuasan']<sb2_v: return "Rendah"
        else: return "Berlebihan"
    bdf['Kuadran']=bdf.apply(bq,axis=1)
    qcb={"Perbaiki":C_KOMP,"Pertahankan":"#10B981","Rendah":C_IMP,"Berlebihan":"#F59E0B","Netral":"#9CA3AF"}
    
    fib=px.scatter(bdf,x='Kepuasan',y='Kepentingan',color='Kuadran',color_discrete_map=qcb)
    fib.update_traces(marker=dict(size=12), customdata=bdf[['Label_Full','N']].values, hovertemplate='<b>%{customdata[0]}</b><br>Kepentingan: %{y:.2f}<br>Kepuasan: %{x:.2f}<br>N: %{customdata[1]}<extra></extra>')
    fib.add_vline(x=sb2_v,line_dash="dash",line_color=border_col)
    fib.add_hline(y=mb,line_dash="dash",line_color=border_col)
    fib.update_layout(height=600)
    st.plotly_chart(elo(fib,f"IPA Matrix Brand (N={len(df):,})",legend_below=True),use_container_width=True)
    
    qcols_b=st.columns(5)
    for col_q,qn,qcv in zip(qcols_b, ["Perbaiki","Pertahankan","Rendah","Berlebihan","Netral"], [C_KOMP,"#10B981",C_IMP,"#F59E0B","#9CA3AF"]):
        its=bdf[bdf['Kuadran']==qn]['Label_Full'].tolist()
        if not its: continue
        with col_q:
            items_html="".join([f"<div style='color:{text_main}!important;font-size:13px;padding:4px 0;border-bottom:1px solid {border_col}'>{it}</div>" for it in its])
            st.markdown(f"<div style='background:{qcv}15;border:1px solid {qcv}40;border-radius:10px;padding:12px 14px;min-height:80px'><div style='color:{qcv}!important;font-weight:800;font-size:13px;margin-bottom:10px'>{qn}</div>{items_html}</div>", unsafe_allow_html=True)

    sh("Gap Kompetitif per Atribut Brand")
    gb=pd.DataFrame({'Atribut':blb,'Atribut_Full':blb_full,'XYZ':bxv,'Komp':bkv, 'Gap':[x-k for x,k in zip(bxv,bkv)], 'N_XYZ':bn_xyz,'N_Komp':bn_komp}).sort_values('Gap')
    gb['Gap']=pd.to_numeric(gb['Gap'],errors='coerce')
    gb['Gap']=gb['Gap'].clip(-1.5,1.5)
    gb=gb.dropna(subset=['Gap'])
    gb['W']=np.where(gb['Gap']<0,C_KOMP,C_XYZ)
    fgb=px.bar(gb,x='Gap',y='Atribut',orientation='h',text='Gap')
    fgb.update_traces(marker_color=gb['W'],texttemplate='%{x:.2f}',textposition='outside', customdata=gb[['Atribut_Full','XYZ','Komp','N_XYZ','N_Komp']].values, hovertemplate='<b>%{customdata[0]}</b><br>XYZ: %{customdata[1]:.2f} (N=%{customdata[3]:.0f})<br>Komp: %{customdata[2]:.2f} (N=%{customdata[4]:.0f})<br>Gap: %{x:.2f}<extra></extra>')
    fgb.update_layout(height=max(600,len(gb)*26),yaxis=dict(tickfont=dict(size=11),automargin=True))
    st.plotly_chart(elo(fgb,f"Gap Brand XYZ vs {target_komp} (+ = XYZ unggul)"),use_container_width=True)
    tg=gb.nlargest(1,'Gap').iloc[0]; bg2=gb.nsmallest(1,'Gap').iloc[0]
    ib(f"Keunggulan terbesar XYZ: **{tg['Atribut_Full']}** (+{tg['Gap']:.2f}). Perlu ditingkatkan: **{bg2['Atribut_Full']}** ({bg2['Gap']:.2f}).")

    sh("Share of Wallet")
    bo=df['A1AX'].dropna().str.split(';').explode().str.strip()
    bo=bo[bo!=''].value_counts().head(10).reset_index(); bo.columns=['Bank','N']
    fsw=px.bar(bo,x='N',y='Bank',orientation='h',color_discrete_sequence=[C_XYZ],text='N')
    fsw.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>%{x} responden menggunakan bank ini<extra></extra>')
    fsw.update_layout(yaxis=dict(automargin=True))
    st.plotly_chart(elo(fsw,f"Bank Lain yang Aktif Digunakan (Total N={len(df):,})",420),use_container_width=True)

    sw1,sw2=st.columns(2)
    with sw1:
        sp=df['A1B'].value_counts().reset_index(); sp.columns=['Bank','N']
        fsp=px.pie(sp,values='N',names='Bank',hole=0.55,color_discrete_sequence=px.colors.qualitative.Set2)
        fsp.update_traces(textposition='inside',textinfo='percent')
        fsp.update_layout(margin=dict(t=20, b=120, l=20, r=20), legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(elo(fsp,f"Bank Utama Simpan Dana (N={len(df):,})",450),use_container_width=True)
    with sw2:
        tr=df['A1C'].value_counts().reset_index(); tr.columns=['Bank','N']
        ftr2=px.pie(tr,values='N',names='Bank',hole=0.55,color_discrete_sequence=px.colors.qualitative.Pastel)
        ftr2.update_traces(textposition='inside',textinfo='percent')
        ftr2.update_layout(margin=dict(t=20, b=120, l=20, r=20), legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(elo(ftr2,f"Bank Utama Bertransaksi (N={len(df):,})",450),use_container_width=True)
    xs=(df['A1B']=='Bank XYZ').mean()*100; xt=(df['A1C']=='Bank XYZ').mean()*100
    rv=bo.iloc[0]['Bank'] if len(bo)>0 else "N/A"
    ib(f"XYZ rekening utama simpan: **{xs:.0f}%**, utama transaksi: **{xt:.0f}%** dari {len(df):,} responden. Kompetitor terbesar: **{rv}**.")

# ═══════════════════════════════════════════════════════
# TAB 4 — TOUCHPOINT & IPA
# ═══════════════════════════════════════════════════════
with t4:
    sh("Analisis Touchpoint")
    tc1,tc2,tc3=st.columns(3)
    with tc1: sel_tp=st.selectbox("Touchpoint:",key="sb_tp4",options=list(DM.keys()))
    with tc2: tpv=st.radio("Tampilan:",["IPA Scatter","Bar"],horizontal=True,key="rd_tp4")
    with tc3: tpc=st.selectbox("Filter Cabang:",key="sb_tpc4", options=["Semua"]+sorted(df['CABANG'].unique().tolist()), help="💡 Ketik nama cabang untuk mencari.")

    dft4=df if tpc=="Semua" else df[df['CABANG']==tpc]
    dftk4=df_hk if tpc=="Semua" else df_hk[df_hk['CABANG']==tpc]
    n_dft4=len(dft4)
    dt4=DM[sel_tp]
    it4=[c for c in dt4["imp"] if c in dft4.columns]
    xt4=[c for c in dt4["xyz"] if c in dft4.columns]
    kt4=[c for c in dt4["komp"] if c in dftk4.columns]
    mt4=min(len(it4),len(xt4))

    if mt4>0:
        lb4_full=[slbl(c,col_map,55) for c in it4[:mt4]]
        lb4=[slbl(c,col_map,40) for c in it4[:mt4]]
        iv4=[dft4[c].mean() for c in it4[:mt4]]
        xv4=[dft4[c].mean() for c in xt4[:mt4]]
        kv4=[dftk4[c].mean() if c in dftk4.columns else np.nan for c in kt4[:mt4]] if kt4 else []
        n4=[dft4[c].notna().sum() for c in it4[:mt4]]

        if tpv=="IPA Scatter":
            idf4=pd.DataFrame({'Item':lb4,'Item_Full':lb4_full,'Kepentingan':iv4,'Kepuasan':xv4,'N':n4})
            mi4,ms4=np.nanmean(iv4),np.nanmean(xv4)
            BUF4=0.05*(np.nanmax(xv4)-np.nanmin(xv4)) if np.nanmax(xv4)>np.nanmin(xv4) else 0
            def tq4(r):
                if abs(r['Kepentingan']-mi4)<=BUF4 and abs(r['Kepuasan']-ms4)<=BUF4: return "Netral"
                if r['Kepentingan']>=mi4 and r['Kepuasan']<ms4: return "Perbaiki"
                elif r['Kepentingan']>=mi4: return "Pertahankan"
                elif r['Kepuasan']<ms4: return "Rendah"
                else: return "Berlebihan"
            idf4['Kuadran']=idf4.apply(tq4,axis=1)
            qtc={"Perbaiki":C_KOMP,"Pertahankan":"#10B981","Rendah":C_IMP,"Berlebihan":"#F59E0B","Netral":"#9CA3AF"}
            
            fi4=px.scatter(idf4,x='Kepuasan',y='Kepentingan',color='Kuadran',color_discrete_map=qtc)
            if kv4:
                fi4.add_trace(go.Scatter(x=kv4,y=iv4,mode='markers',name=target_komp,text=lb4_full, marker=dict(size=10,symbol='x',color=C_KOMP,line=dict(width=2)), hovertemplate='<b>%{text}</b><br>Kepuasan Komp: %{x:.2f}<extra></extra>'))
            fi4.update_traces(marker=dict(size=12), customdata=idf4[['Item_Full','N']].values, hovertemplate='<b>%{customdata[0]}</b><br>Kepentingan: %{y:.2f}<br>Kepuasan: %{x:.2f}<br>N: %{customdata[1]}<extra></extra>')
            fi4.add_vline(x=ms4,line_dash="dash",line_color=border_col)
            fi4.add_hline(y=mi4,line_dash="dash",line_color=border_col)
            fi4.update_layout(height=560)
            st.plotly_chart(elo(fi4,f"IPA — {sel_tp} (N={n_dft4:,})",legend_below=True),use_container_width=True)

            sh("Prioritas per Kuadran")
            qcols4=st.columns(5)
            for col_q,qn,qcv in zip(qcols4, ["Perbaiki","Pertahankan","Rendah","Berlebihan","Netral"], [C_KOMP,"#10B981",C_IMP,"#F59E0B","#9CA3AF"]):
                its=idf4[idf4['Kuadran']==qn]['Item_Full'].tolist()
                if not its: continue
                with col_q:
                    items_html="".join([f"<div style='color:{text_main}!important;font-size:13px;padding:4px 0;border-bottom:1px solid {border_col}'>{it}</div>" for it in its])
                    st.markdown(f"<div style='background:{qcv}15;border:1px solid {qcv}40;border-radius:10px;padding:12px 14px;min-height:80px'><div style='color:{qcv}!important;font-weight:800;font-size:13px;margin-bottom:10px'>{qn}</div>{items_html}</div>", unsafe_allow_html=True)
        else:
            fb4=go.Figure()
            fb4.add_trace(go.Bar(name='Kepentingan',x=lb4,y=iv4,marker_color=C_IMP,opacity=0.72, customdata=n4, hovertemplate='<b>%{x}</b><br>Kepentingan: %{y:.2f}<br>N: %{customdata}<extra></extra>'))
            fb4.add_trace(go.Bar(name='Kepuasan XYZ',x=lb4,y=xv4,marker_color=C_XYZ, customdata=n4, hovertemplate='<b>%{x}</b><br>Kepuasan XYZ: %{y:.2f}<br>N: %{customdata}<extra></extra>'))
            if kv4:
                nk4=[dftk4[c].notna().sum() if c in dftk4.columns else 0 for c in kt4[:mt4]]
                fb4.add_trace(go.Bar(name=f'Kepuasan {target_komp}',x=lb4,y=kv4,marker_color=C_KOMP,opacity=0.85, customdata=nk4, hovertemplate=f'<b>%{{x}}</b><br>Kepuasan Komp: %{{y:.2f}}<br>N: %{{customdata}}<extra></extra>'))
            fb4.update_layout(barmode='group',yaxis_range=[3,6.8],xaxis_tickangle=-30,height=520, xaxis=dict(automargin=True,tickfont=dict(size=11)))
            st.plotly_chart(elo(fb4,f"Kepentingan vs Kepuasan — {sel_tp} (N={n_dft4:,})"),use_container_width=True)

        if kv4:
            sh("Gap Kompetitif per Item")
            gt4=pd.DataFrame({'Item':lb4,'Item_Full':lb4_full, 'Gap':[x-k for x,k in zip(xv4,kv4)], 'XYZ':xv4,'Komp':kv4,'N':n4}).sort_values('Gap')
            gt4['W']=np.where(gt4['Gap']<0,C_KOMP,C_XYZ)
            fg4=px.bar(gt4,x='Gap',y='Item',orientation='h',text='Gap')
            fg4.update_traces(marker_color=gt4['W'],texttemplate='%{x:.2f}',textposition='outside', customdata=gt4[['Item_Full','XYZ','Komp','N']].values, hovertemplate='<b>%{customdata[0]}</b><br>XYZ: %{customdata[1]:.2f} (N=%{customdata[3]:.0f})<br>Komp: %{customdata[2]:.2f} (N=%{customdata[4]:.0f})<br>Gap: %{x:.2f}<extra></extra>')
            fg4.update_layout(height=max(460,mt4*28),yaxis=dict(automargin=True,tickfont=dict(size=11)))
            st.plotly_chart(elo(fg4,f"Gap XYZ vs {target_komp} — {sel_tp}"),use_container_width=True)

    sh("Skor Layanan per Jenis Transaksi")
    d1m={'Teller':df[df['D1_TYPE']=='TELLER']['OVR_TELLER_XYZ'].mean(), 'Customer Service':df[df['D1_TYPE']=='CS']['OVR_CS_XYZ'].mean(), 'Keduanya':df[df['D1_TYPE']=='BOTH'][['OVR_TELLER_XYZ','OVR_CS_XYZ']].mean(axis=1).mean()}
    d1n={'Teller':len(df[df['D1_TYPE']=='TELLER']), 'Customer Service':len(df[df['D1_TYPE']=='CS']), 'Keduanya':len(df[df['D1_TYPE']=='BOTH'])}
    d1df=pd.DataFrame({'Jenis':list(d1m.keys()),'Skor':list(d1m.values()),'N':list(d1n.values())})
    fd1=px.bar(d1df,x='Jenis',y='Skor',color='Jenis',text='Skor', color_discrete_map={'Teller':C_XYZ,'Customer Service':C_KOMP,'Keduanya':'#FBBF24'})
    fd1.update_traces(texttemplate='%{y:.2f}',textposition='outside', customdata=d1df['N'].values, hovertemplate='<b>%{x}</b><br>Skor: %{y:.3f}<br>N responden: %{customdata}<extra></extra>')
    fd1.update_yaxes(range=[4.5,6.5])
    st.plotly_chart(elo(fd1,f"Skor Rata-rata per Jenis Transaksi (Total N={len(df):,}, N per jenis di hover)",400),use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 5 — EMOSI & LOYALITAS
# ═══════════════════════════════════════════════════════
with t5:
    epc=[c for c in["T_I1A_2","T_I1A_5","T_I1A_8","T_I1A_11","T_I1A_14","T_I1A_17","T_I1A_20","T_I1A_23","T_I1A_26"] if c in df.columns]
    enc=[c for c in["T_I1A_29","T_I1A_32","T_I1A_35","T_I1A_38","T_I1A_41","T_I1A_44","T_I1A_47"] if c in df.columns]
    epkc=[c for c in["T_I1A_3","T_I1A_6","T_I1A_9","T_I1A_12","T_I1A_15","T_I1A_18","T_I1A_21","T_I1A_24","T_I1A_27"] if c in df_hk.columns]
    enkc=[c for c in["T_I1A_30","T_I1A_33","T_I1A_36","T_I1A_39","T_I1A_42","T_I1A_45","T_I1A_48"] if c in df_hk.columns]
    elp=["Bahagia","Percaya","Dihargai","Diperhatikan","Aman","Fokus","Dimanjakan","Tertarik","Penuh Semangat"]
    eln=["Tidak Puas","Frustasi","Kecewa","Tertekan","Tidak Bahagia","Diabaikan","Tergesa-gesa"]
    epv=[df[c].mean() for c in epc]; env=[df[c].mean() for c in enc]
    epkv=[df_hk[c].mean() if c in df_hk.columns else np.nan for c in epkc]
    enkv=[df_hk[c].mean() if c in df_hk.columns else np.nan for c in enkc]
    epa=np.nanmean(epv); ena=np.nanmean(env); epka=np.nanmean(epkv); enka=np.nanmean(enkv)
    n_ep=[df[c].notna().sum() for c in epc]; n_en=[df[c].notna().sum() for c in enc]

    color_epa = "green" if epa >= epka else ("red" if epa < epka else "black")
    color_ena = "green" if ena <= enka else ("red" if ena > enka else "black")

    ek=st.columns(4)
    ek[0].markdown(card("Emosi Positif XYZ",f"{epa:.2f}", color_epa, f"N={len(df):,} · Avg 9 dimensi",delta=epa-epka,dlbl="vs Komp"),unsafe_allow_html=True)
    ek[1].markdown(card("Emosi Negatif XYZ",f"{ena:.2f}", color_ena, f"N={len(df):,} · Makin rendah makin baik"),unsafe_allow_html=True)
    ek[2].markdown(card("Emosi Positif Komp",f"{epka:.2f}","black", f"N={len(df_hk):,} · Avg 9 dimensi"),unsafe_allow_html=True)
    ek[3].markdown(card("Emosi Negatif Komp",f"{enka:.2f}","black", f"N={len(df_hk):,} · Makin rendah makin baik"),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ib(f"XYZ unggul di emosi positif ({epa:.2f} vs {epka:.2f}) dan lebih rendah di emosi negatif ({ena:.2f} vs {enka:.2f}) dari {len(df):,} responden.")

    ec1,ec2=st.columns(2)
    with ec1:
        sh("Emosi Positif XYZ vs Kompetitor")
        fep=go.Figure()
        fep.add_trace(go.Bar(name='XYZ',x=elp[:len(epv)],y=epv,marker_color=C_XYZ, text=np.round(epv,2),textposition='outside', customdata=n_ep, hovertemplate='<b>%{x}</b><br>XYZ: %{y:.3f}<br>N: %{customdata}<extra></extra>'))
        fep.add_trace(go.Bar(name=target_komp,x=elp[:len(epkv)],y=epkv,marker_color=C_KOMP, text=np.round(epkv,2),textposition='outside', hovertemplate=f'<b>%{{x}}</b><br>{target_komp}: %{{y:.3f}}<extra></extra>'))
        fep.update_layout(barmode='group',yaxis_range=[3,6.8],xaxis_tickangle=-20,height=420)
        st.plotly_chart(elo(fep,f"N={len(df):,}"),use_container_width=True)
    with ec2:
        sh("Emosi Negatif (Makin Rendah = Makin Baik)")
        fen=go.Figure()
        fen.add_trace(go.Bar(name='XYZ',x=eln[:len(env)],y=env,marker_color=C_XYZ, text=np.round(env,2),textposition='outside', customdata=n_en, hovertemplate='<b>%{x}</b><br>XYZ: %{y:.3f}<br>N: %{customdata}<extra></extra>'))
        fen.add_trace(go.Bar(name=target_komp,x=eln[:len(enkv)],y=enkv,marker_color=C_KOMP, text=np.round(enkv,2),textposition='outside', hovertemplate=f'<b>%{{x}}</b><br>{target_komp}: %{{y:.3f}}<extra></extra>'))
        fen.update_layout(barmode='group',yaxis_range=[1,4],xaxis_tickangle=-20,height=420)
        st.plotly_chart(elo(fen,f"N={len(df):,}"),use_container_width=True)

    sh("Brand Equity — 15 Atribut")
    hxc=[c for c in["T_H1A_2","T_H1A_5","T_H1A_8","T_H1A_11","T_H1A_14","T_H1A_17","T_H1A_20","T_H1A_23","T_H1A_26","T_H1A_29","T_H1A_32","T_H1A_35","T_H1A_38","T_H1A_41","T_H1A_44"] if c in df.columns]
    hkc=[c for c in["T_H1A_3","T_H1A_6","T_H1A_9","T_H1A_12","T_H1A_15","T_H1A_18","T_H1A_21","T_H1A_24","T_H1A_27","T_H1A_30","T_H1A_33","T_H1A_36","T_H1A_39","T_H1A_42","T_H1A_45"] if c in df_hk.columns]
    hlb=["Tetap Gunakan","Kemudahan Transaksi","Digunakan Banyak","Keuntungan Finansial","Produk Lengkap","Promo Gaya Hidup","Kecepatan","Rasa Aman","Kenyamanan","Merasa Dihargai","Bangga","Modern","Bank Turun-Temurun","Cukup Satu Bank","Bergengsi"]
    hxv=[df[c].mean() for c in hxc]; hkv=[df_hk[c].mean() if c in df_hk.columns else np.nan for c in hkc]
    hn_xyz=[df[c].notna().sum() for c in hxc]
    lhb=hlb[:min(len(hxv),len(hlb))]
    fhe=go.Figure()
    fhe.add_trace(go.Bar(name='Bank XYZ',x=lhb,y=hxv[:len(lhb)],marker_color=C_XYZ, text=np.round(hxv[:len(lhb)],2),textposition='outside', customdata=hn_xyz[:len(lhb)], hovertemplate='<b>%{x}</b><br>XYZ: %{y:.3f}<br>N: %{customdata}<extra></extra>'))
    fhe.add_trace(go.Bar(name=target_komp,x=lhb,y=hkv[:len(lhb)],marker_color=C_KOMP, text=np.round(hkv[:len(lhb)],2),textposition='outside', hovertemplate=f'<b>%{{x}}</b><br>{target_komp}: %{{y:.3f}}<extra></extra>'))
    fhe.update_layout(barmode='group',yaxis_range=[3,6.8],xaxis_tickangle=-28,height=480)
    st.plotly_chart(elo(fhe,f"Brand Equity XYZ vs Kompetitor (N={len(df):,})"),use_container_width=True)

    sh("Korelasi Emosi vs Outcome")
    epas=df[[c for c in epc if c in df.columns]].mean(axis=1)
    enas=df[[c for c in enc if c in df.columns]].mean(axis=1)
    cce_df=pd.DataFrame({'Emosi Pos':epas,'Emosi Neg':enas,'NPS':df['G1A'],'Kepuasan':df['E1A'],'Loyalitas':df['F1A']})
    r_mat5,p_mat5,n_mat5=corr_with_stats(cce_df)
    e_c1,e_c2,e_c3=st.columns(3)
    with e_c1:
        fec=px.imshow(r_mat5,text_auto=".2f",color_continuous_scale='Greens',aspect='auto',zmin=-1,zmax=1)
        customdata_cce=np.dstack([p_mat5.values, n_mat5.values])
        fec.update_traces(customdata=customdata_cce, hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<br>' + ('p-value: %{customdata[0]:.4f}<br>' if SCIPY_OK else '') + 'N (pairwise): %{customdata[1]:.0f}<extra></extra>')
        st.plotly_chart(elo(fec,"Korelasi Pearson (hover: p-value & N)",h=340),use_container_width=True)

    with e_c2:
        fs1, r1, n1 = scatter_ols_manual(pd.DataFrame({'Emosi Pos':epas,'G1A':df['G1A'],'G1A_CAT':df['G1A_CAT']}), 'Emosi Pos','G1A','Emosi Pos','NPS Score')
        st.plotly_chart(elo(fs1,f"Emosi Positif vs NPS (N={n1:,})",340,legend_below=True),use_container_width=True)
        if r1: st.caption(f"r={r1[0]:.2f}, p={r1[1]:.4f}. Titik digeser tipis (jitter) agar mudah dibaca — NPS aslinya skala diskrit 4-10, bukan kontinu, jadi garis OLS ini gambaran arah hubungan, bukan prediksi presisi.")
    with e_c3:
        fs2, r2, n2 = scatter_ols_manual(pd.DataFrame({'Emosi Pos':epas,'F1A':df['F1A'],'G1A_CAT':df['G1A_CAT']}), 'Emosi Pos','F1A','Emosi Pos','Loyalitas Nasabah')
        st.plotly_chart(elo(fs2,f"Emosi Positif vs Loyalitas (N={n2:,})",340,legend_below=True),use_container_width=True)
        if r2: st.caption(f"r={r2[0]:.2f}, p={r2[1]:.4f}. Titik digeser tipis (jitter) agar mudah dibaca — Loyalitas aslinya hanya berskala 4-6 (3 nilai), jadi garis OLS ini gambaran arah hubungan, bukan prediksi presisi.")

# ═══════════════════════════════════════════════════════
# TAB 6 — DIGITALISASI
# ═══════════════════════════════════════════════════════
with t6:
    dc=[c for c in["T_J1_1","T_J1_2","T_J1_3","T_J1_4","T_J1_5"] if c in df.columns]; dlb=["Digitalisasi","Digital Signage","Smart Table","Tablet Survey","Akses Cabang"]
    dv=[df[c].mean() for c in dc]; dn=[df[c].notna().sum() for c in dc]; g_dv=[df_raw[c].mean() if c in df_raw.columns else np.nan for c in dc]
    sh("KPI Digitalisasi vs Global")
    st.caption("Ambang warna (hijau ≥5.5, kuning ≥4.5, merah <4.5 dari skala 1–6) adalah patokan internal, bukan baku industri — gunakan sebagai panduan kasar, bukan standar mutlak.")
    dkk=st.columns(len(dc))
    for i,(cw,lb,vl,gvl,nn) in enumerate(zip(dkk,dlb,dv,g_dv,dn)):
        col2="green" if vl>=5.5 else ("amber" if vl>=4.5 else "red")
        cw.markdown(card(lb,f"{vl:.2f}",col2,f"N={nn:,} · Skala 1–6", delta=vl-gvl if not np.isnan(gvl) else None,dlbl="vs Global"),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    dg1,dg2=st.columns(2)
    with dg1:
        fdg=px.bar(x=dv,y=dlb[:len(dv)],orientation='h',color=dv,color_continuous_scale='Greens',text=np.round(dv,2))
        fdg.update_traces(textposition='outside', customdata=dn, hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<br>N responden: %{customdata}<extra></extra>')
        fdg.update_xaxes(range=[3,6.8])
        st.plotly_chart(elo(fdg,"Skor Persepsi Digitalisasi", 400),use_container_width=True)
    with dg2:
        if "T_J1_1" in df.columns:
            dp=df.groupby('PROV').agg(Skor=('T_J1_1','mean'),N=('SERIAL','count')).reset_index().sort_values('Skor')
            fdp=px.bar(dp,x='Skor',y='PROV',orientation='h',color='Skor',color_continuous_scale='Greens',text='Skor')
            fdp.update_traces(texttemplate='%{x:.2f}',textposition='outside', customdata=dp['N'].values, hovertemplate='<b>%{y}</b><br>Skor Digitalisasi: %{x:.3f}<br>N responden: %{customdata}<extra></extra>')
            fdp.update_xaxes(range=[3,6.8])
            st.plotly_chart(elo(fdp,"Digitalisasi per Provinsi", 400),use_container_width=True)

    sh("Sarana Elektronik Layanan")
    slc=[f"T_SL2_{i}" for i in range(1,17) if f"T_SL2_{i}" in df.columns]
    slb=[slbl(c,col_map,38) for c in slc]; slv=[df[c].mean() for c in slc]; sln=[df[c].notna().sum() for c in slc]
    sl1,sl2=st.columns(2)
    with sl1:
        fsl=px.bar(x=slv,y=slb[:len(slv)],orientation='h',color=slv,color_continuous_scale='Greens',text=np.round(slv,2))
        fsl.update_traces(textposition='outside', customdata=sln, hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<br>N responden: %{customdata}<extra></extra>')
        fsl.update_xaxes(range=[3,6.8])
        fsl.update_layout(yaxis=dict(automargin=True,tickfont=dict(size=11)),height=max(420,len(slv)*28))
        st.plotly_chart(elo(fsl,"Ketersediaan & Fungsi Sarana"),use_container_width=True)
    with sl2:
        if "T_J1_1" in df.columns:
            fds, r_ds, n_ds = scatter_ols_manual(df[['T_J1_1','E1A','G1A_CAT']].rename(columns={'T_J1_1':'Digitalisasi'}), 'Digitalisasi','E1A','Persepsi Digitalisasi','Kepuasan Nasabah')
            st.plotly_chart(elo(fds,f"Digitalisasi vs Kepuasan (N={n_ds:,})", 420),use_container_width=True)
            if r_ds: st.caption(f"r={r_ds[0]:.2f}, p={r_ds[1]:.4f}. Titik digeser tipis (jitter) agar mudah dibaca — Kepuasan berskala diskrit, garis OLS ini gambaran arah hubungan.")

    sh("E-Form & Awareness")
    ef1,ef2=st.columns(2)
    with ef1:
        if 'D2' in df.columns:
            ef=df['D2'].dropna().value_counts().reset_index(); ef.columns=['Status','N']
            fee=px.pie(ef,values='N',names='Status',hole=0.55,color_discrete_sequence=['#8B5CF6','#10B981','#F43F5E'])
            fee.update_traces(textposition='inside',textinfo='percent+label', hovertemplate='<b>%{label}</b><br>%{value} responden<br>%{percent}<extra></extra>')
            st.plotly_chart(elo(fee,f"Penggunaan E-Form (N={df['D2'].notna().sum():,})", 400),use_container_width=True)
    with ef2:
        if 'D4' in df.columns:
            ea=df['D4'].dropna().value_counts().reset_index(); ea.columns=['Status','N']
            fea=px.bar(ea,x='N',y='Status',orientation='h',color_discrete_sequence=[C_XYZ],text='N')
            fea.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
            st.plotly_chart(elo(fea,f"Awareness E-Form (N={df['D4'].notna().sum():,})", 400),use_container_width=True)

    j2m={"T_J2_1":"Digitalisasi Layanan","T_J2_2":"Digital Signage","T_J2_3":"Smart Table","T_J2_4":"Tablet Survey","T_J2_5":"Akses Cabang"}
    j2l=st.selectbox("Topik Saran Perbaikan:",key="sb_j26",options=list(j2m.values()))
    j2c=[k for k,v in j2m.items() if v==j2l][0]
    if j2c in df.columns:
        sr=df[j2c].dropna().value_counts().reset_index(); sr.columns=['Saran','N']
        sr=sr[~sr['Saran'].str.strip().isin(['','None','Tidak Ada','Tidak Ada /  Tidak Tahu'])]
        if len(sr)>0: st.dataframe(sr,use_container_width=True,hide_index=True,height=250)

# ═══════════════════════════════════════════════════════
# TAB 7 — CLUSTERING
# ═══════════════════════════════════════════════════════
with t7:
    sh("Segmentasi Nasabah via K-Means Clustering")
    if not SK_OK:
        st.warning("scikit-learn tidak terinstal. Tambahkan ke requirements.txt.")
    else:
        cl1,cl2,cl3=st.columns(3)
        with cl1: n_cl=st.slider("Jumlah Cluster:",2,6,3,key="sl_ncl")
        with cl2:
            cl_feat_opts={
                'NPS Score':'G1A', 'Kepuasan Nasabah':'E1A', 'Loyalitas Nasabah':'F1A',
                'Layanan Teller':'OVR_TELLER_XYZ', 'Layanan Customer Service':'OVR_CS_XYZ',
                'Layanan ATM':'OVR_ATM_XYZ', 'Layanan Sekuriti':'OVR_SEKURITI_XYZ',
                'Layanan Kantor Cabang':'OVR_KC_XYZ', 'Usia':'S2_1'
            }
            sel_feats=st.multiselect("Fitur Clustering:",key="ms_clf", options=list(cl_feat_opts.keys()), default=['NPS Score','Kepuasan Nasabah','Loyalitas Nasabah', 'Layanan Teller','Layanan Customer Service'])
        with cl3: cl_color=st.selectbox("Warna marker by:",key="sb_clc", options=["Cluster","Kategori NPS","Provinsi","Gender"])

        df_cl=df.copy()
        epc_cl=[c for c in["T_I1A_2","T_I1A_5","T_I1A_8","T_I1A_11","T_I1A_14","T_I1A_17","T_I1A_20","T_I1A_23","T_I1A_26"] if c in df_cl.columns]
        if epc_cl:
            df_cl['Emosi_Pos']=df_cl[epc_cl].mean(axis=1)
            cl_feat_opts['Emosi Positif']='Emosi_Pos'

        feat_cols=[cl_feat_opts[f] for f in sel_feats if f in cl_feat_opts and cl_feat_opts[f] in df_cl.columns]
        df_cl_valid=df_cl[feat_cols].copy()
        for _col in feat_cols: df_cl_valid[_col]=pd.to_numeric(df_cl_valid[_col],errors='coerce')
        df_cl_valid=df_cl_valid.dropna()

        if len(df_cl_valid)<n_cl*5:
            st.warning("Data tidak cukup untuk clustering dengan pilihan ini.")
        else:
            sc=StandardScaler()
            X=sc.fit_transform(df_cl_valid.values.astype(float))
            km=KMeans(n_clusters=n_cl,random_state=42,n_init=10)
            df_cl.loc[df_cl_valid.index,'Cluster']=km.fit_predict(X).astype(str)
            df_cl['Cluster']=df_cl['Cluster'].fillna('N/A')

            with st.expander("Elbow Chart — Pilih Jumlah Cluster Optimal"):
                inertias=[]
                for k in range(2,min(10,len(df_cl_valid)//5+1)):
                    km_e=KMeans(n_clusters=k,random_state=42,n_init=10); km_e.fit(X); inertias.append(km_e.inertia_)
                fel=px.line(x=list(range(2,min(10,len(df_cl_valid)//5+1))), y=inertias,markers=True, labels={'x':'Jumlah Cluster','y':'Inertia (WCSS)'}, color_discrete_sequence=[C_XYZ])
                fel.update_traces(hovertemplate='K=%{x} cluster<br>Inertia: %{y:.0f}<extra></extra>')
                st.plotly_chart(elo(fel, "Pilih K di titik siku — makin kecil inertia makin baik",300), use_container_width=True)

            pca=PCA(n_components=2,random_state=42)
            X_pca=pca.fit_transform(X)
            var1,var2=pca.explained_variance_ratio_
            df_pca=df_cl.loc[df_cl_valid.index].copy()
            df_pca['PC1']=X_pca[:,0]; df_pca['PC2']=X_pca[:,1]

            color_map_cl={"Cluster":"Cluster","Kategori NPS":"G1A_CAT", "Provinsi":"PROV","Gender":"S1"}
            color_col=color_map_cl[cl_color]
            color_d=(NPS_C if cl_color=="Kategori NPS" else {u:c for u,c in zip(df_pca[color_col].dropna().unique(), px.colors.qualitative.Plotly)})

            cd_exist=[c for c in ['CABANG','PROV','G1A','E1A','F1A','G1A_CAT','Cluster'] if c in df_pca.columns]
            fpc=px.scatter(df_pca,x='PC1',y='PC2', color=color_col,color_discrete_map=color_d, symbol='Cluster',opacity=0.7, labels={'PC1':f'Komponen Utama 1 ({var1*100:.1f}% variansi)', 'PC2':f'Komponen Utama 2 ({var2*100:.1f}% variansi)', color_col:cl_color})
            fpc.update_traces(marker=dict(size=8), customdata=df_pca[cd_exist].values, hovertemplate=('<b>%{customdata[0]}</b><br>Provinsi: %{customdata[1]}<br>NPS Score: %{customdata[2]:.1f}<br>Kepuasan: %{customdata[3]:.2f}<br>Loyalitas: %{customdata[4]:.2f}<br>Kategori NPS: %{customdata[5]}<br>Cluster: %{customdata[6]}<extra></extra>'))
            fpc.update_layout(height=500)
            st.plotly_chart(elo(fpc, f"Visualisasi Cluster 2D — Variansi Terjelaskan: {(var1+var2)*100:.1f}% (N={len(df_pca):,})",legend_below=True), use_container_width=True)

            sh("Profil Rata-rata per Cluster")
            prof_cols=[c for c in feat_cols if c in df_pca.columns]
            extra_c=[c for c in['G1A','E1A','F1A'] if c not in feat_cols and c in df_pca.columns]
            prof=df_pca.groupby('Cluster')[prof_cols+extra_c].mean().round(2)
            rename_map={c:nama_kolom(c) for c in prof.columns}; prof=prof.rename(columns=rename_map)
            prof['Jumlah Responden']=df_pca.groupby('Cluster').size()
            nps_dom=df_pca.groupby('Cluster')['G1A_CAT'].agg(lambda x: x.value_counts().index[0])
            prof['Kategori NPS Dominan']=nps_dom
            st.dataframe(prof,use_container_width=True)
            ib("Tabel menunjukkan rata-rata tiap metrik per cluster beserta jumlah responden. Cluster dengan sampel sedikit perlu diinterpretasi hati-hati.")

            if len(feat_cols)>=3:
                sh("Radar Profil per Cluster")
                fig_cl_r=go.Figure()
                cl_colors_r=px.colors.qualitative.Plotly
                theta_labels=[sel_feats[i] if i<len(sel_feats) else f for i,f in enumerate(feat_cols)]
                for ci,cl_name in enumerate(sorted(df_pca['Cluster'].dropna().unique())):
                    cl_data=(df_pca[df_pca['Cluster']==cl_name][feat_cols].mean().values.tolist())
                    n_cl_resp=len(df_pca[df_pca['Cluster']==cl_name])
                    fig_cl_r.add_trace(go.Scatterpolar(r=cl_data,theta=theta_labels,fill='toself', name=f'Cluster {cl_name} (N={n_cl_resp})', line_color=cl_colors_r[ci%len(cl_colors_r)], hovertemplate='<b>%{theta}</b><br>Rata-rata: %{r:.2f}<extra></extra>'))
                fig_cl_r.update_layout(polar=dict(bgcolor=BG, radialaxis=dict(visible=True,gridcolor=GRID, tickfont=dict(color=text_muted,size=10)), angularaxis=dict(tickfont=dict(color=text_main,size=12))), height=520,margin=dict(t=40,b=60,l=60,r=60))
                st.plotly_chart(elo(fig_cl_r,"Radar Profil per Cluster",legend_below=True), use_container_width=True)

            sh("Distribusi Kategori NPS per Cluster")
            nps_cl=df_pca.groupby(['Cluster','G1A_CAT']).size().reset_index(name='Jumlah')
            nps_cl_pct=(df_pca.groupby('Cluster')['G1A_CAT'].value_counts(normalize=True).mul(100).round(1).rename('Persen').reset_index())
            nps_cl=nps_cl.merge(nps_cl_pct,on=['Cluster','G1A_CAT'])
            fnc=px.bar(nps_cl,x='Cluster',y='Jumlah',color='G1A_CAT', color_discrete_map=NPS_C,barmode='stack',text='Jumlah', labels={'Jumlah':'Jumlah Responden','G1A_CAT':'Kategori NPS'})
            fnc.update_traces(textposition='inside', customdata=nps_cl['Persen'].values, hovertemplate=('Cluster %{x}<br>Kategori: %{fullData.name}<br>Jumlah: %{y} responden<br>Persentase: %{customdata:.1f}%<extra></extra>'))
            st.plotly_chart(elo(fnc,"Komposisi Kategori NPS per Cluster", 460), use_container_width=True)

            if 'G1A' in df_pca.columns:
                cl_prof=df_pca.groupby('Cluster')[['G1A','E1A','F1A']].mean()
                n_per_cl=df_pca.groupby('Cluster').size()
                best_cl=cl_prof['G1A'].idxmax(); worst_cl=cl_prof['G1A'].idxmin()
                ib(f"Cluster **{best_cl}** NPS Score tertinggi ({cl_prof.loc[best_cl,'G1A']:.1f}, N={n_per_cl[best_cl]}) — segmen promoter utama. Cluster **{worst_cl}** NPS Score terendah ({cl_prof.loc[worst_cl,'G1A']:.1f}, N={n_per_cl[worst_cl]}) — prioritas program retention.")

# ═══════════════════════════════════════════════════════
# TAB 8 — PROFIL & SEGMENTASI
# ═══════════════════════════════════════════════════════
with t8:
    sh("Profil Demografis")
    d1,d2=st.columns([1,2])
    with d1:
        fg8=px.pie(df,names='S1',hole=0.6,color_discrete_sequence=[C_XYZ,'#34D399','#A78BFA'])
        fg8.update_traces(textposition='inside',textinfo='percent+label', hovertemplate='<b>%{label}</b><br>%{value} responden<br>%{percent}<extra></extra>')
        st.plotly_chart(elo(fg8,f"Gender (N={len(df):,})",360),use_container_width=True)
    with d2:
        ac=df['S2_2'].value_counts().reset_index(); ac.columns=['Usia','N']
        fa=px.bar(ac,x='Usia',y='N',color_discrete_sequence=[C_XYZ],text='N')
        fa.update_traces(textposition='outside', hovertemplate='<b>%{x}</b><br>%{y} responden<extra></extra>')
        fa.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(elo(fa,f"Distribusi Usia (N={len(df):,})",360),use_container_width=True)

    d3,d4=st.columns(2)
    with d3:
        ec8=df['P3'].value_counts().reset_index(); ec8.columns=['Pendidikan','N']
        fe8=px.bar(ec8,x='N',y='Pendidikan',orientation='h',color_discrete_sequence=[C_XYZ],text='N')
        fe8.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fe8.update_layout(yaxis=dict(automargin=True),height=max(360,len(ec8)*36))
        st.plotly_chart(elo(fe8,f"Tingkat Pendidikan (N={len(df):,})"),use_container_width=True)
    with d4:
        jc8=df['P4'].value_counts().reset_index(); jc8.columns=['Pekerjaan','N']
        fj8=px.bar(jc8,x='N',y='Pekerjaan',orientation='h',color_discrete_sequence=['#A78BFA'],text='N')
        fj8.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fj8.update_layout(yaxis=dict(automargin=True),height=max(360,len(jc8)*36))
        st.plotly_chart(elo(fj8,f"Pekerjaan (N={len(df):,})"),use_container_width=True)

    ss1,ss2=st.columns(2)
    with ss1:
        sc8=df['P5'].value_counts().reset_index(); sc8.columns=['SES','N']
        fs8=px.bar(sc8,x='N',y='SES',orientation='h',color_discrete_sequence=[C_XYZ],text='N')
        fs8.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fs8.update_layout(yaxis=dict(automargin=True))
        st.plotly_chart(elo(fs8,f"Tingkat Pengeluaran / SES (N={len(df):,})", 380),use_container_width=True)
    with ss2:
        ic8=df['P6'].value_counts().reset_index(); ic8.columns=['Penghasilan','N']
        fi8=px.bar(ic8,x='N',y='Penghasilan',orientation='h',color_discrete_sequence=['#10B981'],text='N')
        fi8.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fi8.update_layout(yaxis=dict(automargin=True))
        st.plotly_chart(elo(fi8,f"Distribusi Penghasilan (N={len(df):,})", 380),use_container_width=True)

    sh("Peta Geografis Responden")
    geo=df.groupby(['PROV','KABKOTA','CABANG']).agg(N=('SERIAL','count'),NPS=('G1A','mean'), Kepuasan=('E1A','mean'),Loyalitas=('F1A','mean')).reset_index()
    cm8=st.selectbox("Warna berdasarkan:",key="sb_cm8",options=['NPS','Kepuasan','Loyalitas','N'])
    ftr8=px.treemap(geo,path=[px.Constant("Nasional"),'PROV','KABKOTA','CABANG'], values='N',color=cm8,color_continuous_scale='Greens',hover_data=['NPS','Kepuasan','Loyalitas'])
    ftr8.update_layout(height=560)
    st.plotly_chart(elo(ftr8,f"Treemap Geografis (Warna = {cm8}, Total N={len(df):,})"),use_container_width=True)

    sh("Segmentasi Interaktif")
    sg1,sg2,sg3=st.columns(3)
    with sg1:
        seg=st.selectbox("Segmen by:",key="sb_sg8",options=['S1 → Gender','S2_2 → Usia','S4 → Lama Nasabah','S7 → Frekuensi Transaksi','P3 → Pendidikan','P4 → Pekerjaan','P1 → Status Pernikahan','P5 → SES / Pengeluaran'])
    with sg2:
        met=st.selectbox("Metrik:",key="sb_mt8",options=['G1A → NPS Score','E1A → Kepuasan Nasabah','F1A → Loyalitas Nasabah','OVR_TELLER_XYZ → Layanan Teller','OVR_CS_XYZ → Layanan Customer Service','OVR_ATM_XYZ → Layanan ATM','OVR_KC_XYZ → Layanan Kantor Cabang'])
    with sg3:
        sct=st.radio("Chart:",["Bar","Box"],horizontal=True,key="rd_sct8")
    sk8=seg.split(' → ')[0].strip(); mk8=met.split(' → ')[0].strip()
    met_label=met.split(' → ')[1].strip(); seg_label=seg.split(' → ')[1].strip()
    MIN_N_SEGMEN=10
    if sk8 in df.columns and mk8 in df.columns:
        seg_counts=df[sk8].value_counts()
        valid_segs=seg_counts[seg_counts>=MIN_N_SEGMEN].index
        n_hidden=(seg_counts<MIN_N_SEGMEN).sum()
        df_seg8=df[df[sk8].isin(valid_segs)]
        if sct=="Bar":
            sa8=df_seg8.groupby(sk8)[mk8].agg(['mean','std','count']).reset_index()
            sa8.columns=['Segmen','Mean','Std','N']; sa8=sa8.sort_values('Mean')
            fs8b=px.bar(sa8,x='Mean',y='Segmen',orientation='h',color='Mean', color_continuous_scale='Greens',text='Mean',error_x='Std')
            fs8b.update_traces(texttemplate='%{x:.2f}',textposition='outside', customdata=sa8[['Std','N']].values, hovertemplate='<b>%{y}</b><br>Rata-rata: %{x:.3f}<br>Std Deviasi: %{customdata[0]:.3f}<br>N responden: %{customdata[1]:.0f}<extra></extra>')
            fs8b.update_layout(yaxis=dict(automargin=True))
        else:
            fs8b=px.box(df_seg8,x=sk8,y=mk8,color=sk8,points='outliers')
            fs8b.update_layout(xaxis_tickangle=-20, xaxis_title=seg_label, yaxis_title=met_label, showlegend=False)
            fs8b.update_traces(hovertemplate='<b>%{x}</b><br>Nilai: %{y:.2f}<extra></extra>')
        st.plotly_chart(elo(fs8b,f"{met_label} per {seg_label} (N={len(df_seg8):,})",460),use_container_width=True)
        if n_hidden>0:
            st.caption(f"ℹ️ {n_hidden} kategori disembunyikan karena N < {MIN_N_SEGMEN} (sampel terlalu kecil untuk dibandingkan secara andal).")

    sh("Frekuensi Transaksi vs Outcome")
    fr8=df.groupby('S7').agg(NPS=('G1A','mean'),Kepuasan=('E1A','mean'), Loyalitas=('F1A','mean'),N=('SERIAL','count')).reset_index()
    ffr=go.Figure()
    for cn,cc,nl in [('NPS',C_XYZ,'NPS Score'),('Kepuasan','#34D399','Kepuasan Nasabah'), ('Loyalitas','#FBBF24','Loyalitas Nasabah')]:
        ffr.add_trace(go.Bar(name=nl,x=fr8['S7'],y=fr8[cn],marker_color=cc, text=fr8[cn].round(2),textposition='outside',customdata=fr8['N'], hovertemplate=f'<b>%{{x}}</b><br>{nl}: %{{y:.2f}}<br>N responden: %{{customdata}}<extra></extra>'))
    ffr.update_layout(barmode='group',xaxis_tickangle=-15,height=420)
    st.plotly_chart(elo(ffr,f"Frekuensi Transaksi vs Outcome (N={len(df):,})"),use_container_width=True)

    sh("Tujuan Buka Rekening vs Loyalitas")
    tj=df['A2'].dropna().str.split(';').explode().str.strip()
    tjdf=tj.to_frame('Tujuan').join(df['F1A'],how='left')
    tjagg=tjdf.groupby('Tujuan').agg(Loyalitas=('F1A','mean'),N=('F1A','count')).reset_index()
    tjagg=tjagg[tjagg['N']>=10].sort_values('Loyalitas')
    ftj=px.bar(tjagg,x='Loyalitas',y='Tujuan',orientation='h',color='N', color_continuous_scale='Greens',text='Loyalitas', labels={'N':'Jumlah Responden'})
    ftj.update_traces(texttemplate='%{x:.2f}',textposition='outside', customdata=tjagg['N'].values, hovertemplate='<b>%{y}</b><br>Loyalitas Rata-rata: %{x:.3f}<br>N responden: %{customdata}<extra></extra>')
    ftj.update_xaxes(range=[4.5,6.6]); ftj.update_layout(yaxis=dict(automargin=True))
    st.plotly_chart(elo(ftj,f"Loyalitas per Tujuan Buka Rekening (N={len(df):,})", 460),use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 9 — VOICE OF CUSTOMER
# ═══════════════════════════════════════════════════════
with t9:
    nv9,pp9,pv9,dv9=calc_nps(df['G1A_CAT']); n9=len(df)
    vk=st.columns(4)
    vk[0].markdown(card("NPS Score",f"{nv9:.0f}", "green" if nv9>=g_nps else "red", f"N={n9:,} · {df['G1A_CAT'].eq('Promoter').sum()} promoter", delta=nv9-g_nps),unsafe_allow_html=True)
    vk[1].markdown(card("Promoter %",f"{pp9:.0f}%","green", f"{int(n9*pp9/100):,} dari {n9:,} responden"),unsafe_allow_html=True)
    vk[2].markdown(card("Passive %",f"{pv9:.0f}%","amber", f"{int(n9*pv9/100):,} dari {n9:,} responden"),unsafe_allow_html=True)
    vk[3].markdown(card("Detractor %",f"{dv9:.0f}%","red", f"{int(n9*dv9/100):,} dari {n9:,} responden"),unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    vh1,vh2=st.columns(2)
    with vh1:
        fnh=px.histogram(df,x='G1A',nbins=11,color='G1A_CAT',color_discrete_map=NPS_C, labels={'G1A':'NPS Score XYZ','count':'Jumlah Responden'})
        fnh.update_layout(bargap=0.08, height=400)
        fnh.update_traces(hovertemplate='NPS Score: %{x}<br>Jumlah responden: %{y}<extra></extra>')
        st.plotly_chart(elo(fnh,f"Distribusi NPS Score XYZ (N={n9:,})",legend_below=True),use_container_width=True)
    with vh2:
        if len(df_hk)>0:
            fnhk=px.histogram(df_hk,x='G1C',nbins=11,color='G1C_CAT',color_discrete_map=NPS_C, labels={'G1C':f'NPS Score {target_komp}','count':'Jumlah Responden'})
            fnhk.update_layout(bargap=0.08, height=400)
            fnhk.update_traces(hovertemplate='NPS Score: %{x}<br>Jumlah responden: %{y}<extra></extra>')
            st.plotly_chart(elo(fnhk,f"Distribusi NPS Score {target_komp} (N={len(df_hk):,})",legend_below=True),use_container_width=True)

    sh("Analisis Teks & Wordcloud")
    wf1,wf2,wf3=st.columns(3)
    with wf1: fc9=st.selectbox("Filter Kategori NPS:",key="sb_fc9",options=["Semua","Promoter","Passive","Detractor"])
    with wf2: ws9=st.selectbox("Sumber Komentar:",key="sb_ws9",options=["G1B — Alasan NPS XYZ","G1D — Alasan NPS Kompetitor", "E1AA — Alasan Kepuasan XYZ","E1BB — Alasan Kepuasan Komp"])
    with wf3: mwl=st.slider("Min. panjang kata:",3,7,4,key="sl_mwl9")

    cmap9={"G1B — Alasan NPS XYZ":("G1B","G1A_CAT",df), "G1D — Alasan NPS Kompetitor":("G1D","G1C_CAT",df_hk), "E1AA — Alasan Kepuasan XYZ":("E1AA","G1A_CAT",df), "E1BB — Alasan Kepuasan Komp":("E1BB","G1C_CAT",df_hk)}
    tc9,ca9,ds9=cmap9[ws9]
    dfw9=ds9 if fc9=="Semua" else ds9[ds9[ca9]==fc9] if ca9 in ds9.columns else ds9
    n_wc=len(dfw9)

    def mk_wc(series,cm='Greens',mw=80):
        parsed=clean_cmt(series); text=" ".join(parsed.astype(str).tolist()).lower()
        words=re.findall(rf'\b[a-zA-Z]{{{mwl},}}\b',text)
        wcw=[w for w in words if w not in STOP]
        if len(wcw)<5: return None,[]
        wco=WordCloud(width=720,height=340,background_color=bg_panel,colormap=cm,max_words=mw, stopwords=STOP,prefer_horizontal=0.8).generate(" ".join(wcw)) if WC_OK else None
        return wco,Counter(wcw).most_common(10)

    wco9,tw9=mk_wc(dfw9[tc9] if tc9 in dfw9.columns else pd.Series(dtype=str))
    wcc1,wcc2=st.columns([1.6,1])
    with wcc1:
        if wco9 and WC_OK:
            fig_wc9,ax9=plt.subplots(figsize=(7.2,3.4),facecolor=bg_panel)
            ax9.imshow(wco9,interpolation='bilinear'); ax9.axis('off')
            fig_wc9.tight_layout(pad=0); st.pyplot(fig_wc9); plt.close()
        else: st.info("Tidak ada teks yang cukup untuk wordcloud.")
    with wcc2:
        if tw9:
            kdf9=pd.DataFrame(tw9,columns=['Kata','Frekuensi'])
            fkw=px.bar(kdf9.sort_values('Frekuensi'),x='Frekuensi',y='Kata',orientation='h', color='Frekuensi',color_continuous_scale='Greens',text='Frekuensi')
            fkw.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>Muncul %{x}x dalam komentar<extra></extra>')
            fkw.update_layout(height=380,showlegend=False)
            st.plotly_chart(elo(fkw,f"Top 10 Kata Kunci (N={n_wc:,})"),use_container_width=True)

    sh("Tema Dominan: Promoter vs Detractor")
    n_prom=len(df[df['G1A_CAT']=='Promoter']); n_detr=len(df[df['G1A_CAT']=='Detractor'])
    pa1,pa2=st.columns(2)
    def top_theme9(series,n=8):
        parsed=clean_cmt(series); text=" ".join(parsed.astype(str).tolist()).lower()
        words=re.findall(r'\b[a-zA-Z]{4,}\b',text)
        return Counter([w for w in words if w not in STOP]).most_common(n)

    with pa1:
        st.markdown(f"<span style='color:#10B981;font-size:15px;font-weight:800'>Alasan Promoter (N={n_prom:,})</span>",unsafe_allow_html=True)
        pw9=top_theme9(df[df['G1A_CAT']=='Promoter']['G1B'] if 'G1B' in df.columns else pd.Series(dtype=str))
        if pw9:
            pdf9=pd.DataFrame(pw9,columns=['Tema','Frekuensi'])
            fp9=px.bar(pdf9,x='Frekuensi',y='Tema',orientation='h', color_discrete_sequence=['#10B981'],text='Frekuensi')
            fp9.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>Muncul %{x}x dalam komentar Promoter<extra></extra>')
            fp9.update_layout(height=360)
            st.plotly_chart(elo(fp9),use_container_width=True)
    with pa2:
        st.markdown(f"<span style='color:#EF4444;font-size:15px;font-weight:800'>Alasan Detractor (N={n_detr:,})</span>",unsafe_allow_html=True)
        dw9=top_theme9(df[df['G1A_CAT']=='Detractor']['G1B'] if 'G1B' in df.columns else pd.Series(dtype=str))
        if dw9:
            ddf9=pd.DataFrame(dw9,columns=['Tema','Frekuensi'])
            fd9=px.bar(ddf9,x='Frekuensi',y='Tema',orientation='h', color_discrete_sequence=[C_KOMP],text='Frekuensi')
            fd9.update_traces(textposition='outside', hovertemplate='<b>%{y}</b><br>Muncul %{x}x dalam komentar Detractor<extra></extra>')
            fd9.update_layout(height=360)
            st.plotly_chart(elo(fd9),use_container_width=True)
    if tw9: ib(f"Kata paling dominan ({fc9}, N={n_wc:,}): **'{tw9[0][0]}'** ({tw9[0][1]}x). Cerminan tema utama persepsi nasabah terhadap Bank XYZ.")

    sh("Eksplorasi Keyword")
    ve1,ve2,ve3,ve4=st.columns(4)
    with ve1: vnf=st.multiselect("Kategori NPS:",key="ms_vnf9",options=['Promoter','Passive','Detractor'],default=['Promoter','Passive','Detractor'])
    with ve2: vpf=st.multiselect("Provinsi:",key="ms_vpf9",options=sorted(df['PROV'].dropna().unique()))
    with ve3: vsr=st.text_input("Cari kata kunci:",key="ti_vsr9")
    with ve4: vms=st.slider("Min. NPS Score:",0,10,0,key="sl_vms9")

    vdf=df[df['G1A_CAT'].isin(vnf)] if vnf else df
    if vpf: vdf=vdf[vdf['PROV'].isin(vpf)]
    vdf=vdf[vdf['G1A']>=vms]
    if vsr:
        mask=vdf['G1B'].fillna('').str.contains(vsr,case=False)
        if 'E1AA' in vdf.columns: mask=mask|vdf['E1AA'].fillna('').str.contains(vsr,case=False)
        vdf=vdf[mask]
    vdfd=vdf.copy()
    if 'G1B' in vdfd.columns: vdfd['Alasan NPS']=clean_cmt(vdfd['G1B'])
    if 'E1AA' in vdfd.columns: vdfd['Alasan Kepuasan']=clean_cmt(vdfd['E1AA'])
    st.info(f"Menampilkan **{len(vdfd):,}** responden dari total {len(df):,}")
    dc9={'CABANG':'Cabang','PROV':'Provinsi','S1':'Gender','S2_2':'Usia','S4':'Lama Nasabah', 'G1A':'NPS Score','G1A_CAT':'Kategori NPS','E1A':'Kepuasan', 'Alasan NPS':'Alasan NPS','Alasan Kepuasan':'Alasan Kepuasan'}
    ex9={k:v for k,v in dc9.items() if k in vdfd.columns}
    st.dataframe(vdfd[list(ex9.keys())].rename(columns=ex9).sort_values('NPS Score'), use_container_width=True,height=460,hide_index=True)

# ═══════════════════════════════════════════════════════
# TAB 10 — AI ASSISTANT
# ═══════════════════════════════════════════════════════
with t10:
    sh("AI CX Assistant — Powered by Groq & Llama 3.1")
    st.markdown(f"""
    <div style='background:{hover_bg}; border:1px solid {accent_color}; border-radius:14px; padding:18px 22px; margin-bottom:20px'>
    <div style='color:{accent_color}!important; font-size:15px; font-weight:800; margin-bottom:10px'>
    Cara Menggunakan AI Assistant</div>
    <div style='color:{text_main}!important; font-size:14px; line-height:1.8'>
    AI menjawab berdasarkan data aktual sesuai filter aktif saat ini, termasuk breakdown per provinsi, demografi, dan waktu tunggu.<br>
    Contoh: <i>"Apa kelemahan utama XYZ?"</i> · <i>"Provinsi mana NPS-nya paling rendah?"</i> · 
    <i>"Mengapa nasabah jadi Detractor?"</i><br>
    <span style='font-size:12px;color:{text_muted}'>Catatan: AI hanya menjawab dari ringkasan data yang tersedia — untuk cabang/segmen yang sangat spesifik dan tidak tercakup di ringkasan, AI akan menyampaikan bila datanya tidak tersedia.</span>
    </div></div>""", unsafe_allow_html=True)

    ctx=build_ctx(df,df_hk)
    if "chat_history" not in st.session_state: st.session_state.chat_history=[]

    st.markdown(f"<div style='color:{text_muted}!important; font-size:13px; font-weight:700; text-transform:uppercase; letter-spacing:.8px; margin-bottom:12px'>Pertanyaan Cepat</div>", unsafe_allow_html=True)
    qq=["Apa kelemahan utama XYZ vs kompetitor?","Cabang mana yang perlu paling diperhatikan?", "Provinsi mana dengan NPS terendah dan kenapa?","Apakah ada perbedaan kepuasan antar segmen usia?"]
    qc=st.columns(4)
    for i,(cq,qt) in enumerate(zip(qc,qq)):
        if cq.button(qt,key=f"qq_{i}"):
            st.session_state.chat_history.append({"role":"user","content":qt})
            with st.spinner("AI sedang menganalisis data..."):
                rep=call_ai(st.session_state.chat_history,ctx)
            st.session_state.chat_history.append({"role":"assistant","content":rep})
            st.rerun()

    st.markdown("<br>",unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("<div class='chat-wrap'>",unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['role']=='user':
                st.markdown(f"<div class='chat-user'><p>{msg['content']}</p></div><div class='chat-cf'></div>",unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'><p>{msg['content']}</p></div><div class='chat-cf'></div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

    ai1,ai2=st.columns([4,1])
    with ai1: user_in=st.text_input("Tanyakan sesuatu tentang data:", placeholder="Contoh: Bagaimana perbandingan emosi nasabah XYZ vs kompetitor?", key="ti_ai10",label_visibility="collapsed")
    with ai2: send=st.button("Kirim",key="btn_ai10",use_container_width=True)

    if send and user_in.strip():
        st.session_state.chat_history.append({"role":"user","content":user_in})
        with st.spinner("AI sedang menganalisis..."):
            rep=call_ai(st.session_state.chat_history,ctx)
        st.session_state.chat_history.append({"role":"assistant","content":rep})
        st.rerun()

    if st.session_state.chat_history:
        _,c_rst=st.columns([4,1])
        with c_rst:
            if st.button("Reset Chat",key="btn_rst10",use_container_width=True):
                st.session_state.chat_history=[]; st.rerun()

    with st.expander("Lihat Konteks Data yang Dikirim ke AI"):
        st.code(ctx,language='text')
