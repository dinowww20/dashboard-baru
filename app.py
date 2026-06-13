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

# ═══════════════════════════════════════════════════════
# CONFIG & STYLE
# ═══════════════════════════════════════════════════════
st.set_page_config(page_title="Bank XYZ — CX Intelligence",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#07101F!important}
.main .block-container{padding-top:.6rem;max-width:100%}
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0C1829 0%,#07101F 100%)!important;
  border-right:1px solid #162038!important}
[data-testid="stSidebar"] *{color:#B8CCDF!important}
[data-testid="stSidebar"] .stSelectbox>label,
[data-testid="stSidebar"] .stMultiSelect>label,
[data-testid="stSidebar"] .stSlider>label{
  color:#3D5470!important;font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.8px}
*,p,span,div,label,h1,h2,h3,h4{color:#D8E6F3!important}
.stTabs [data-baseweb="tab-list"]{
  background:#0C1829!important;border-radius:14px;
  padding:5px;gap:3px;border:1px solid #162038}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;border:none!important;
  border-radius:10px;padding:9px 15px;
  color:#3D5470!important;font-weight:600;font-size:11px}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#163690,#2563EB)!important;
  color:#FFF!important;box-shadow:0 2px 14px rgba(37,99,235,.4)}
.metric-card{
  background:linear-gradient(150deg,#0C1829,#07101F);
  border:1px solid #162038;padding:18px 14px;border-radius:18px;
  text-align:center;transition:all .3s;position:relative;overflow:hidden}
.metric-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#163690,#3B82F6,#60A5FA)}
.metric-card:hover{transform:translateY(-4px);border-color:#2563EB;
  box-shadow:0 10px 35px rgba(37,99,235,.2)}
.metric-title{color:#2D4060!important;font-size:9px;font-weight:800;
  text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px}
.metric-value{font-size:28px;font-weight:900;line-height:1}
.metric-value.blue  {color:#60A5FA!important}
.metric-value.green {color:#34D399!important}
.metric-value.red   {color:#F87171!important}
.metric-value.amber {color:#FBBF24!important}
.metric-value.purple{color:#A78BFA!important}
.metric-value.white {color:#EEF4FF!important}
.metric-sub{color:#2D4060!important;font-size:10px;margin-top:5px}
.metric-delta-up  {color:#34D399!important;font-size:11px;font-weight:700;margin-top:7px}
.metric-delta-down{color:#F87171!important;font-size:11px;font-weight:700;margin-top:7px}
.metric-delta-neu {color:#4E6A88!important;font-size:11px;font-weight:700;margin-top:7px}
.sec-hdr{font-size:13px;font-weight:800;color:#EEF4FF!important;
  border-left:3px solid #2563EB;padding-left:10px;margin:18px 0 10px}
.insight-box{
  background:linear-gradient(135deg,#0C2048,#0C1829);
  border:1px solid #163690;border-radius:12px;padding:12px 16px;margin:8px 0}
.insight-box p{color:#93C5FD!important;font-size:12px;margin:0;line-height:1.6}
.chat-wrap{background:#0C1829;border:1px solid #162038;border-radius:14px;
  padding:16px;margin:8px 0;max-height:440px;overflow-y:auto}
.chat-user{background:linear-gradient(135deg,#163690,#2563EB);
  border-radius:14px 14px 4px 14px;padding:10px 14px;margin:6px 0 6px auto;
  max-width:76%;display:inline-block;float:right;clear:both}
.chat-user p{color:#DBEAFE!important;font-size:12px;margin:0;line-height:1.6}
.chat-ai{background:linear-gradient(135deg,#0C1E3C,#0C2449);
  border:1px solid #163690;border-radius:14px 14px 14px 4px;
  padding:10px 14px;margin:6px 0;max-width:82%;display:inline-block;
  float:left;clear:both}
.chat-ai p{color:#BAE6FD!important;font-size:12px;margin:0;line-height:1.6}
.chat-cf{clear:both}
.stButton button{
  background:linear-gradient(135deg,#163690,#2563EB)!important;
  color:#FFF!important;border:none!important;border-radius:10px;
  font-weight:700;font-size:12px;transition:all .2s}
.stButton button:hover{
  background:linear-gradient(135deg,#2563EB,#3B82F6)!important;
  box-shadow:0 4px 18px rgba(37,99,235,.45);transform:translateY(-1px)}
hr{border-color:#162038!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#07101F}
::-webkit-scrollbar-thumb{background:#162038;border-radius:4px}
.stTextInput input,.stTextArea textarea{
  background:#0C1829!important;color:#D8E6F3!important;
  border:1px solid #162038!important;border-radius:10px}
</style>""", unsafe_allow_html=True)

NPS_C  = {'Promoter':'#10B981','Passive':'#F59E0B','Detractor':'#EF4444'}
C_XYZ  = '#3B82F6'
C_KOMP = '#F87171'
C_IMP  = '#243650'
BG     = '#07101F'
SURF   = '#0C1829'
GRID   = '#0F1E30'

# ── helpers ──────────────────────────────────────────
def card(title, value, color="white", sub="", delta=None, dlbl="vs Global"):
    if delta is not None:
        if delta > 0.005:
            dh = f"<div class='metric-delta-up'>▲ {abs(delta):.2f} {dlbl}</div>"
        elif delta < -0.005:
            dh = f"<div class='metric-delta-down'>▼ {abs(delta):.2f} {dlbl}</div>"
        else:
            dh = f"<div class='metric-delta-neu'>─ Sama dgn {dlbl}</div>"
    else:
        dh = ""
    return (f"<div class='metric-card'>"
            f"<div class='metric-title'>{title}</div>"
            f"<div class='metric-value {color}'>{value}</div>"
            f"{'<div class=metric-sub>'+sub+'</div>' if sub else ''}"
            f"{dh}</div>")

def ib(txt):
    st.markdown(f"<div class='insight-box'><p>💡 {txt}</p></div>",
                unsafe_allow_html=True)

def sh(txt):
    st.markdown(f"<div class='sec-hdr'>{txt}</div>", unsafe_allow_html=True)

def elo(fig, title="", h=None):
    kw = dict(
        title=dict(text=title, font=dict(size=13, color='#D8E6F3', weight='bold')),
        template="plotly_dark", plot_bgcolor=BG, paper_bgcolor=BG,
        margin=dict(t=50, b=14, l=10, r=14),
        font=dict(color='#6A8AAA', size=11),
        hoverlabel=dict(bgcolor='#0C1E3C', font_color='#D8E6F3',
                        font_size=12, bordercolor='#2563EB'),
        legend=dict(font=dict(color='#6A8AAA', size=11),
                    bgcolor='rgba(12,24,41,.9)', bordercolor='#162038',
                    orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    if h: kw['height'] = h
    fig.update_layout(**kw)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     automargin=True, tickfont=dict(color='#3D5470'))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     automargin=True, tickfont=dict(color='#3D5470'))
    return fig

def slbl(col, col_map, n=36):
    s = col_map.get(col, col)
    s = re.sub(r'\s*-\s*(XYZ|kompetitor)\s*$','',s,flags=re.I)
    s = re.sub(r'\([^)]*\)','',s).strip()
    s = re.sub(r'\s+',' ',s)
    return s[:n]+'…' if len(s)>n else s

def parse_cmt(raw):
    if pd.isna(raw) or str(raw).strip() in ('','None','nan'): return None
    parts = [p.strip() for p in str(raw).split(';')]
    skip = {'NET','SUBNET','POSITIVE COMMENTS','NEGATIVE COMMENTS',
            'LAIN-LAIN','OTHERS','NEGATIVE COMMENTS (NET)','POSITIVE COMMENTS (NET)'}
    for p in reversed(parts):
        if len(p)>8 and p.upper() not in skip: return p
    return parts[-1] if parts else None

def clean_cmt(s): return s.apply(parse_cmt).dropna()

STOP = {
    'yang','untuk','dengan','pada','dari','sebagai','tidak','karena','sangat',
    'lebih','sudah','saya','bank','bisa','dan','di','ke','ini','itu','ada',
    'juga','net','subnet','positive','negative','comments','dalam','oleh',
    'akan','telah','dapat','kami','anda','nya','atau','jadi','baru','lagi',
    'saat','masih','serta','namun','jika','agar','bagi','atas','antara',
    'setiap','para','mereka','kita','xyz','nasabah','layanan','cabang',
    'rekening','lainnya','none','baik','bagus','cukup','sekali','paling',
    'makin','belum','kurang'
}

def calc_nps(s):
    t = s.notna().sum()
    if t==0: return 0.,0.,0.,0.
    pr=(s=='Promoter').sum(); ps=(s=='Passive').sum(); dt=(s=='Detractor').sum()
    return round((pr-dt)/t*100,1),round(pr/t*100,1),round(ps/t*100,1),round(dt/t*100,1)

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
    st.error(f"❌ Gagal memuat data: {e}"); st.stop()

g_nps,_,_,_ = calc_nps(df_raw['G1A_CAT'])
g_sat = df_raw['E1A'].mean()
g_loy = df_raw['F1A'].mean()

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='padding:12px 0 10px'>"
        "<h3 style='color:#60A5FA!important;font-weight:900;font-size:16px;margin:0'>"
        "🏦 CX Intelligence</h3>"
        "<p style='color:#243650!important;font-size:10px;margin:2px 0 0'>"
        "Bank XYZ Analytics Platform v5.0</p></div>",
        unsafe_allow_html=True)

    periods = sorted(df_raw['Periode'].unique())
    sel_per = st.select_slider("📅 Periode", key="sl_per",
        options=periods, value=(periods[0], periods[-1]))
    per_range = periods[periods.index(sel_per[0]):periods.index(sel_per[1])+1]

    st.markdown("---")
    st.markdown("**🎯 Benchmark**")
    komp_opts   = sorted(df_raw['KOMP'].dropna().unique())
    target_komp = st.selectbox("Kompetitor:", key="sb_komp",
        options=["Semua (Rata-rata)"]+komp_opts)

    st.markdown("---")
    st.markdown("**📍 Lokasi**")
    sel_prov = st.multiselect("Provinsi", key="ms_prov",
        options=sorted(df_raw['PROV'].dropna().unique()))
    kpool = df_raw[df_raw['PROV'].isin(sel_prov)] if sel_prov else df_raw
    sel_kota = st.multiselect("Kab/Kota", key="ms_kota",
        options=sorted(kpool['KABKOTA'].dropna().unique()))
    cpool = kpool[kpool['KABKOTA'].isin(sel_kota)] if sel_kota else kpool
    sel_cab = st.multiselect("Cabang", key="ms_cab",
        options=sorted(cpool['CABANG'].dropna().unique()))

    st.markdown("---")
    with st.expander("👤 Profil Responden"):
        sel_gender    = st.multiselect("Gender",key="ms_gd",options=sorted(df_raw['S1'].dropna().unique()))
        sel_usia      = st.multiselect("Usia",key="ms_us",options=sorted(df_raw['S2_2'].dropna().unique()))
        sel_tenure    = st.multiselect("Lama Nasabah",key="ms_tn",options=sorted(df_raw['S4'].dropna().unique()))
        sel_frek      = st.multiselect("Frekuensi",key="ms_fk",options=sorted(df_raw['S7'].dropna().unique()))
        sel_panel     = st.multiselect("Panel",key="ms_pl",options=sorted(df_raw['PANEL'].dropna().unique()))
        sel_pekerjaan = st.multiselect("Pekerjaan",key="ms_pk",options=sorted(df_raw['P4'].dropna().unique()))
        sel_pendidikan= st.multiselect("Pendidikan",key="ms_pd",options=sorted(df_raw['P3'].dropna().unique()))
        sel_status    = st.multiselect("Status Nikah",key="ms_st",options=sorted(df_raw['P1'].dropna().unique()))

    with st.expander("🏦 Perilaku Perbankan"):
        sel_bsimpan = st.multiselect("Bank Utama Simpan",key="ms_bs",options=sorted(df_raw['A1B'].dropna().unique()))
        sel_btrans  = st.multiselect("Bank Utama Transaksi",key="ms_bt",options=sorted(df_raw['A1C'].dropna().unique()))
        sel_npscat  = st.multiselect("Kategori NPS",key="ms_nc",options=['Promoter','Passive','Detractor'])

    with st.expander("🎯 Filter Skor"):
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

df_komp = df.copy() if target_komp=="Semua (Rata-rata)" else df[df['KOMP']==target_komp]
df_hk   = df_komp[df_komp['KOMP'].notna()]

with st.sidebar:
    st.success(f"📊 **{len(df):,}** Responden aktif")
    if len(df_hk): st.info(f"🏦 **{len(df_hk):,}** Dgn Kompetitor")
    if len(df)<30: st.warning("⚠️ Sampel < 30")

if df.empty: st.warning("⚠️ Data kosong — sesuaikan filter."); st.stop()

# dimensi map
def get_dm(dfr):
    return {
      "Kantor Cabang":{
        "imp":[f"T_KC1_{i}" for i in range(1,36) if f"T_KC1_{i}" in dfr.columns],
        "xyz":[c for c in dfr.columns if c.startswith("T_KC2_") and
               c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116",
                          "T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and
               int(c.split("_")[-1])%3==2],
        "komp":[c for c in dfr.columns if c.startswith("T_KC2_") and
                c not in ["T_KC2_107","T_KC2_110","T_KC2_113","T_KC2_116",
                           "T_KC2_108","T_KC2_111","T_KC2_114","T_KC2_117"] and
                int(c.split("_")[-1])%3==0]},
      "Sekuriti":{
        "imp":[f"T_SC1_{i}" for i in range(1,16) if f"T_SC1_{i}" in dfr.columns],
        "xyz":[c for c in dfr.columns if c.startswith("T_SC2_") and
               c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1])%3==2],
        "komp":[c for c in dfr.columns if c.startswith("T_SC2_") and
                c not in ["T_SC2_47","T_SC2_48"] and int(c.split("_")[-1])%3==0]},
      "Teller":{
        "imp":[f"T_TL2_{i}" for i in range(1,20) if f"T_TL2_{i}" in dfr.columns],
        "xyz":[c for c in dfr.columns if c.startswith("T_TL3_") and
               c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1])%3==2],
        "komp":[c for c in dfr.columns if c.startswith("T_TL3_") and
                c not in ["T_TL3_59","T_TL3_60"] and int(c.split("_")[-1])%3==0]},
      "Customer Service":{
        "imp":[f"T_CS2_{i}" for i in range(1,24) if f"T_CS2_{i}" in dfr.columns],
        "xyz":[c for c in dfr.columns if c.startswith("T_CS3_") and
               c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1])%3==2],
        "komp":[c for c in dfr.columns if c.startswith("T_CS3_") and
                c not in ["T_CS3_71","T_CS3_72"] and int(c.split("_")[-1])%3==0]},
      "Customer Advisor":{
        "imp":[f"T_CA1_{i}" for i in range(1,20) if f"T_CA1_{i}" in dfr.columns],
        "xyz":[c for c in dfr.columns if c.startswith("T_CA2_") and c!="T_CA2_20"],
        "komp":[]},
      "ATM":{
        "imp":[f"T_AT2_{i}" for i in range(1,19) if f"T_AT2_{i}" in dfr.columns],
        "xyz":[c for c in dfr.columns if c.startswith("T_AT3_") and
               c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1])%3==2],
        "komp":[c for c in dfr.columns if c.startswith("T_AT3_") and
                c not in ["T_AT3_56","T_AT3_57"] and int(c.split("_")[-1])%3==0]},
    }
DM = get_dm(df_raw)

# ═══════════════════════════════════════════════════════
# AI — GEMINI
# ═══════════════════════════════════════════════════════
def build_ctx(dff, dhk):
    ns,pp,_,pd2 = calc_nps(dff['G1A_CAT'])
    nk,pk,_,dk  = calc_nps(dhk['G1C_CAT']) if len(dhk)>0 else (0,0,0,0)
    oc = [c for c in dff.columns if c.startswith("OVR_") and "_XYZ" in c]
    om = {c.replace("OVR_","").replace("_XYZ",""):round(dff[c].mean(),2)
          for c in oc if c in dff.columns}
    tc = [c for c in dff.columns if c.startswith("OVR_") and "_XYZ" in c
          and c not in ["OVR_KC_OPERASIONAL_XYZ","OVR_KC_PARKIR_XYZ",
                         "OVR_KC_BANKINGHALL_XYZ","OVR_KC_TOILET_XYZ"]]
    cs = dff.groupby('CABANG')[tc].mean().mean(axis=1) if tc else pd.Series(dtype=float)
    t3 = cs.nlargest(3).index.tolist()  if len(cs)>0 else []
    b3 = cs.nsmallest(3).index.tolist() if len(cs)>0 else []
    pc = clean_cmt(dff[dff['G1A_CAT']=='Promoter']['G1B']).head(5).tolist() \
         if 'G1B' in dff.columns else []
    dc = clean_cmt(dff[dff['G1A_CAT']=='Detractor']['G1B']).head(5).tolist() \
         if 'G1B' in dff.columns else []
    return (f"DATA RINGKASAN BANK XYZ (filter aktif):\n"
            f"- Total responden: {len(dff):,}\n"
            f"- NPS XYZ: {ns} (Promoter {pp}%, Detractor {pd2}%)\n"
            f"- NPS Kompetitor: {nk} (Promoter {pk}%, Detractor {dk}%)\n"
            f"- Gap NPS: {round(ns-nk,1)} poin\n"
            f"- Kepuasan XYZ (1-6): {round(dff['E1A'].mean(),2)}\n"
            f"- Loyalitas XYZ (1-6): {round(dff['F1A'].mean(),2)}\n"
            f"- Skor dimensi: {json.dumps(om)}\n"
            f"- Top 3 cabang terbaik: {t3}\n"
            f"- Bottom 3 cabang: {b3}\n"
            f"- Sampel alasan Promoter: {pc}\n"
            f"- Sampel alasan Detractor: {dc}\n"
            f"- Kompetitor dibandingkan: {target_komp}")

def call_ai(msgs, ctx):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            return "⚠️ GROQ_API_KEY belum dikonfigurasi di Streamlit Secrets."
        from groq import Groq
        client = Groq(api_key=api_key)
        sys_p = (
            "Kamu adalah analis Customer Experience (CX) senior ahli perbankan Indonesia. "
            "Analisis data survei kepuasan nasabah Bank XYZ dan berikan insight actionable. "
            "Jawab dalam Bahasa Indonesia profesional. Sertakan angka spesifik dari data. "
            "Format: maksimal 4 paragraf pendek, langsung ke point.\n\n"
            f"Konteks data:\n{ctx}"
        )
        all_msgs = [{"role":"system","content":sys_p}] + msgs
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=all_msgs,
            max_tokens=800,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Gagal menghubungi AI: {str(e)}"

# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center;padding:10px 0 18px;
background:linear-gradient(180deg,#0C1829 0%,transparent 100%);
border-radius:16px;margin-bottom:6px'>
<div style='font-size:10px;color:#1E3A5F!important;letter-spacing:3px;
text-transform:uppercase;margin-bottom:6px'>ADVANCED ANALYTICS PLATFORM</div>
<h1 style='font-weight:900;letter-spacing:-2px;color:#EEF4FF!important;
font-size:26px;margin:0;line-height:1.1'>
🏦 BANK XYZ — CX INTELLIGENCE
</h1>
<div style='color:#1E3A5F!important;font-size:11px;margin-top:6px'>
Customer Experience Dashboard · Dark Edition v5.0
</div></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════
tabs = st.tabs(["🌟 Executive","🏢 Kinerja Layanan","🏆 Brand & Komp.",
    "🎯 Touchpoint","💡 Emosi & Loyalitas","📱 Digitalisasi",
    "🧩 Clustering","👥 Profil & Segmen","💬 Voice of Customer","🤖 AI Analyst"])
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

    sh("📌 Key Performance Indicators")
    k = st.columns(8)
    k[0].markdown(card("NPS Score XYZ",  f"{ns:.0f}","blue",  f"P:{pp:.0f}% D:{pd2:.0f}%",delta=ns-g_nps),   unsafe_allow_html=True)
    k[1].markdown(card("NPS Kompetitor", f"{nk:.0f}","red",   f"P:{pk:.0f}% D:{dk:.0f}%"),                     unsafe_allow_html=True)
    k[2].markdown(card("Gap NPS",f"+{gap:.0f}" if gap>=0 else f"{gap:.0f}","green" if gap>=0 else "red","XYZ−Komp"), unsafe_allow_html=True)
    k[3].markdown(card("Kepuasan XYZ",   f"{sat:.2f}","blue", "Skala 1–6",delta=sat-g_sat),                    unsafe_allow_html=True)
    k[4].markdown(card("Kepuasan Komp",  f"{sat_k:.2f}" if not np.isnan(sat_k) else "N/A","red","Skala 1–6"),  unsafe_allow_html=True)
    k[5].markdown(card("Loyalitas XYZ",  f"{loy:.2f}","green","Skala 1–6",delta=loy-g_loy),                   unsafe_allow_html=True)
    k[6].markdown(card("Promoter XYZ",   f"{pp:.0f}%","green",f"{int(df['G1A_CAT'].eq('Promoter').sum())} orang"), unsafe_allow_html=True)
    k[7].markdown(card("Total Resp.",    f"{len(df):,}","white",f"{len(df_hk):,} dgn komp"),                   unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if gap>50:   ib(f"XYZ unggul sangat signifikan — gap NPS {gap:.0f} poin. Promoter {pp:.0f}% vs kompetitor {pk:.0f}%.")
    elif gap>20: ib(f"XYZ unggul {gap:.0f} poin NPS. Pertahankan momentum dan tingkatkan dimensi dengan gap terkecil.")
    else:        ib(f"Gap NPS hanya {gap:.0f} poin — persaingan sangat ketat. Perlu diferensiasi lebih tajam di touchpoint utama.")

    st.markdown("<br>",unsafe_allow_html=True)
    r1,r2 = st.columns([1,2.2])
    with r1:
        sh("🎯 NPS Composition")
        for cat_s, lbl, nv in [
            (df['G1A_CAT'],"Bank XYZ",ns),
            (df_hk['G1C_CAT'] if len(df_hk)>0 else pd.Series(dtype=str),
             target_komp[:20], nk)]:
            if len(cat_s)>0:
                cp = cat_s.value_counts().reset_index(); cp.columns=['K','N']
                fd = px.pie(cp,values='N',names='K',hole=0.65,color='K',color_discrete_map=NPS_C)
                fd.update_traces(textposition='outside',textinfo='percent+label',
                    marker=dict(line=dict(color=BG,width=2)),
                    hovertemplate='<b>%{label}</b><br>Jumlah: %{value}<br>%{percent}<extra></extra>')
                fd.add_annotation(text=f"{lbl}<br><b>{nv:.0f}</b>",
                    x=0.5,y=0.5,showarrow=False,font=dict(size=12,color='#D8E6F3'))
                fd.update_layout(height=205,margin=dict(t=22,b=6,l=0,r=0))
                st.plotly_chart(elo(fd),use_container_width=True)

    with r2:
        sh("📊 Scorecard Dimensi XYZ vs Kompetitor")
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
            gp = xv-kv if not(np.isnan(xv) or np.isnan(kv)) else np.nan
            rows.append({"Dimensi":lb,"XYZ":round(xv,2),
                "Kompetitor":round(kv,2) if not np.isnan(kv) else None,
                "Gap":round(gp,2) if not np.isnan(gp) else None})
        sdf = pd.DataFrame(rows)
        fig_sc=go.Figure()
        fig_sc.add_trace(go.Bar(name='Bank XYZ',y=sdf['Dimensi'],x=sdf['XYZ'],
            orientation='h',marker_color=C_XYZ,
            text=sdf['XYZ'],texttemplate='%{x:.2f}',textposition='outside',
            customdata=sdf[['XYZ','Kompetitor','Gap']].values,
            hovertemplate='<b>%{y}</b><br>XYZ: %{customdata[0]:.2f}<br>Komp: %{customdata[1]}<br>Gap: %{customdata[2]}<extra></extra>'))
        fig_sc.add_trace(go.Bar(name='Kompetitor',y=sdf['Dimensi'],
            x=pd.to_numeric(sdf['Kompetitor'],errors='coerce'),
            orientation='h',marker_color=C_KOMP,opacity=0.85,
            hovertemplate='<b>%{y}</b><br>Kompetitor: %{x:.2f}<extra></extra>'))
        fig_sc.update_layout(barmode='group',xaxis_range=[4,6.8],height=430)
        st.plotly_chart(elo(fig_sc),use_container_width=True)

    r3,r4=st.columns(2)
    with r3:
        sh("🏆 Top & Bottom 5 Cabang")
        oc=[c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c
            and c not in ["OVR_KC_OPERASIONAL_XYZ","OVR_KC_PARKIR_XYZ",
                           "OVR_KC_BANKINGHALL_XYZ","OVR_KC_TOILET_XYZ"]]
        if oc:
            cs2=df.groupby('CABANG')[oc].mean().mean(axis=1).reset_index()
            cs2.columns=['CABANG','Skor']
            tb=pd.concat([cs2.nlargest(5,'Skor').assign(S='🌟 Top 5'),
                          cs2.nsmallest(5,'Skor').assign(S='⚠️ Bottom 5')])
            ftb=px.bar(tb,x='Skor',y='CABANG',color='S',orientation='h',text='Skor',
                color_discrete_map={'🌟 Top 5':C_XYZ,'⚠️ Bottom 5':C_KOMP})
            ftb.update_traces(texttemplate='%{x:.2f}',textposition='outside',
                hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<extra></extra>')
            ftb.update_xaxes(range=[4.5,6.6])
            st.plotly_chart(elo(ftb,h=380),use_container_width=True)

    with r4:
        sh("📊 Matriks Korelasi Antar Dimensi")
        cm_map={'NPS':'G1A','Kepuasan':'E1A','Loyalitas':'F1A',
                'Teller':'OVR_TELLER_XYZ','CS':'OVR_CS_XYZ',
                'ATM':'OVR_ATM_XYZ','Sekuriti':'OVR_SEKURITI_XYZ','KC':'OVR_KC_XYZ'}
        vm={k:v for k,v in cm_map.items() if v in df.columns}
        cdf_=df[list(vm.values())].copy(); cdf_.columns=list(vm.keys())
        fco=px.imshow(cdf_.corr(),text_auto=".2f",
            color_continuous_scale='RdBu',aspect='auto',zmin=-1,zmax=1)
        fco.update_traces(hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>')
        st.plotly_chart(elo(fco,h=380),use_container_width=True)

    sh("📈 Tren NPS, Kepuasan & Loyalitas per Periode")
    tren=df.groupby('Periode').agg(NPS=('G1A','mean'),Kepuasan=('E1A','mean'),
        Loyalitas=('F1A','mean'),N=('SERIAL','count')).reset_index()
    ftr=go.Figure()
    for cn,cc,nl in [('NPS',C_XYZ,'NPS'),('Kepuasan','#34D399','Kepuasan'),('Loyalitas','#FBBF24','Loyalitas')]:
        ftr.add_trace(go.Scatter(x=tren['Periode'],y=tren[cn],mode='lines+markers',
            name=nl,line=dict(color=cc,width=2.5),marker=dict(size=7),
            customdata=tren['N'],
            hovertemplate=f'<b>{nl}</b><br>Periode: %{{x}}<br>Nilai: %{{y:.2f}}<br>N: %{{customdata}}<extra></extra>'))
    st.plotly_chart(elo(ftr,"",310),use_container_width=True)

    sh("🗺️ NPS per Provinsi")
    pn=df.groupby('PROV').agg(NPS=('G1A','mean'),Kepuasan=('E1A','mean'),
        Loyalitas=('F1A','mean'),N=('SERIAL','count')).reset_index().sort_values('NPS')
    fpn=px.bar(pn,x='NPS',y='PROV',orientation='h',
        color='NPS',color_continuous_scale='Blues',text='NPS')
    fpn.update_traces(texttemplate='%{x:.1f}',textposition='outside',
        customdata=pn[['Kepuasan','Loyalitas','N']].values,
        hovertemplate='<b>%{y}</b><br>NPS: %{x:.2f}<br>Kepuasan: %{customdata[0]:.2f}<br>Loyalitas: %{customdata[1]:.2f}<br>N: %{customdata[2]}<extra></extra>')
    fpn.update_xaxes(range=[7,11])
    st.plotly_chart(elo(fpn,h=max(360,len(pn)*34)),use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — KINERJA LAYANAN
# ═══════════════════════════════════════════════════════
with t2:
    sh("🔥 Heatmap Kinerja Cabang")
    hm1,hm2=st.columns([2,1])
    with hm1:
        oa=[c for c in df.columns if c.startswith("OVR_") and "_XYZ" in c]
        ol={c:c.replace("OVR_","").replace("_XYZ","").replace("_"," ").title() for c in oa}
        sel_hm=st.multiselect("Dimensi heatmap:",key="ms_hm",options=list(ol.keys()),
            default=oa,format_func=lambda x:ol[x])
    with hm2:
        mr=st.slider("Min. responden per cabang:",3,30,8,key="sl_mr")
        shm=st.selectbox("Urutkan:",key="sb_shm",options=["Rata-rata"]+[ol[c] for c in sel_hm if c in ol])

    if sel_hm:
        hf=df.groupby('CABANG').filter(lambda x:len(x)>=mr)
        hd=hf.groupby('CABANG')[sel_hm].mean().round(2)
        hd.columns=[ol[c] for c in hd.columns]
        hd=hd.loc[(hd.mean(axis=1).sort_values(ascending=False).index if shm=="Rata-rata"
                   else hd.sort_values(shm,ascending=False).index)]
        fhm=px.imshow(hd,text_auto=".2f",color_continuous_scale='RdYlGn',aspect='auto',zmin=4,zmax=6)
        fhm.update_traces(hovertemplate='Cabang: <b>%{y}</b><br>Dimensi: <b>%{x}</b><br>Skor: <b>%{z:.2f}</b><extra></extra>')
        fhm.update_layout(height=max(400,len(hd)*22))
        st.plotly_chart(elo(fhm,"Heatmap Skor OVR (Merah=Rendah → Hijau=Tinggi)"),use_container_width=True)
        if len(hd)>0:
            wb=hd.mean(axis=1).idxmax(); ww=hd.mean(axis=1).idxmin()
            ib(f"Performa terbaik: **{wb}** ({hd.loc[wb].mean():.2f}). Perlu perhatian: **{ww}** ({hd.loc[ww].mean():.2f}).")

    st.markdown("---")
    sh("🔍 Drill-Down Item Level per Dimensi")
    dc1,dc2,dc3=st.columns(3)
    with dc1: sel_dim=st.selectbox("Dimensi:",key="sb_dim2",options=list(DM.keys()))
    with dc2: dm2=st.radio("Tampilan:",["Bar Chart","Scatter IPA"],horizontal=True,key="rd_dm2")
    with dc3:
        sel_c2=st.selectbox("Filter Cabang:",key="sb_c2",
            options=["Semua"]+sorted(df['CABANG'].dropna().unique().tolist()))

    dfd=df if sel_c2=="Semua" else df[df['CABANG']==sel_c2]
    dfdk=df_hk if sel_c2=="Semua" else df_hk[df_hk['CABANG']==sel_c2]
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

        if dm2=="Bar Chart":
            # Bar importance vs sat — full width
            fdd=go.Figure()
            fdd.add_trace(go.Bar(name='Importance',x=iv2,y=lb2,orientation='h',
                marker_color=C_IMP,opacity=0.75,
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>'))
            fdd.add_trace(go.Bar(name='Satisfaction XYZ',x=xv2,y=lb2,orientation='h',
                marker_color=C_XYZ,
                hovertemplate='<b>%{y}</b><br>Sat. XYZ: %{x:.2f}<extra></extra>'))
            if kv2:
                fdd.add_trace(go.Bar(name=f'Sat. {target_komp}',x=kv2,y=lb2,orientation='h',
                    marker_color=C_KOMP,opacity=0.85,
                    hovertemplate=f'<b>%{{y}}</b><br>Sat.Komp: %{{x:.2f}}<extra></extra>'))
            fdd.update_layout(barmode='group',xaxis_range=[3,6.8],
                height=max(420,mn2*30),yaxis=dict(automargin=True,tickfont=dict(size=10)))
            st.plotly_chart(elo(fdd,f"Importance vs Satisfaction — {sel_dim}"),use_container_width=True)

            # Gap chart di bawah, full width
            sh("Gap: Satisfaction − Importance")
            gd2=pd.DataFrame({'Item':lb2,'Gap':[x-i for x,i in zip(xv2,iv2)]}).sort_values('Gap')
            gd2['W']=np.where(gd2['Gap']<0,C_KOMP,C_XYZ)
            fg2=px.bar(gd2,x='Gap',y='Item',orientation='h',text='Gap')
            fg2.update_traces(marker_color=gd2['W'],texttemplate='%{x:.2f}',textposition='outside',
                hovertemplate='<b>%{y}</b><br>Gap: %{x:.3f}<extra></extra>')
            fg2.update_layout(height=max(380,mn2*28),yaxis=dict(automargin=True,tickfont=dict(size=10)))
            st.plotly_chart(elo(fg2),use_container_width=True)
            worst=gd2.iloc[0]
            if worst['Gap']<0: ib(f"Item paling kritis: **{worst['Item']}** (gap {worst['Gap']:.2f}) — high importance, low satisfaction.")

        else:  # IPA Scatter
            idf2=pd.DataFrame({'Item':lb2,'Importance':iv2,'Satisfaction':xv2})
            mi2,ms2=np.nanmean(iv2),np.nanmean(xv2)
            def q2(r):
                if r['Importance']>=mi2 and r['Satisfaction']<ms2: return "⚠️ Perbaiki"
                elif r['Importance']>=mi2: return "🌟 Pertahankan"
                elif r['Satisfaction']<ms2: return "💤 Rendah"
                else: return "✅ Berlebihan"
            idf2['Kuadran']=idf2.apply(q2,axis=1)
            qc2={"⚠️ Perbaiki":C_KOMP,"🌟 Pertahankan":"#34D399","💤 Rendah":C_IMP,"✅ Berlebihan":"#FBBF24"}
            fi2=px.scatter(idf2,x='Satisfaction',y='Importance',text='Item',color='Kuadran',color_discrete_map=qc2)
            if kv2:
                fi2.add_trace(go.Scatter(x=kv2,y=iv2,mode='markers',name=target_komp,text=lb2,
                    marker=dict(size=10,symbol='x',color=C_KOMP,line=dict(width=2)),
                    hovertemplate='<b>%{text}</b><br>Sat.Komp: %{x:.2f}<extra></extra>'))
            fi2.update_traces(marker=dict(size=10),textposition='top center',textfont=dict(size=9),
                hovertemplate='<b>%{text}</b><br>Imp: %{y:.2f}<br>Sat: %{x:.2f}<extra></extra>',
                selector=dict(mode='markers+text'))
            fi2.add_vline(x=ms2,line_dash="dash",line_color="#162038")
            fi2.add_hline(y=mi2,line_dash="dash",line_color="#162038")
            fi2.update_layout(height=520)
            st.plotly_chart(elo(fi2,f"IPA Matrix — {sel_dim}"),use_container_width=True)

            # Prioritas kuadran dalam 4 kolom
            sh("📋 Prioritas per Kuadran")
            qcols=st.columns(4)
            for col_q,qn,qcv in zip(qcols,
                ["⚠️ Perbaiki","🌟 Pertahankan","💤 Rendah","✅ Berlebihan"],
                [C_KOMP,"#34D399",C_IMP,"#FBBF24"]):
                its=idf2[idf2['Kuadran']==qn]['Item'].tolist()
                with col_q:
                    items_html="".join([
                        f"<div style='color:#6A8AAA!important;font-size:11px;"
                        f"padding:3px 0;border-bottom:1px solid #162038'>{it}</div>"
                        for it in its])
                    st.markdown(
                        f"<div style='background:{qcv}15;border:1px solid {qcv}40;"
                        f"border-radius:10px;padding:10px 12px;min-height:60px'>"
                        f"<div style='color:{qcv}!important;font-weight:800;font-size:11px;"
                        f"margin-bottom:8px'>{qn}</div>{items_html}</div>",
                        unsafe_allow_html=True)

    st.markdown("---")
    sh("⏱️ Analisis Waktu Tunggu")
    wt1,wt2,wt3=st.columns(3)
    with wt1:
        dft2=df[df['PANEL']=='Teller'].dropna(subset=['TL5','TL6'])
        if len(dft2)>0:
            wd=pd.DataFrame({'Metrik':['Aktual','Toleransi'],'Menit':[dft2['TL5'].mean(),dft2['TL6'].mean()]})
            fw=px.bar(wd,x='Metrik',y='Menit',color='Metrik',text='Menit',
                color_discrete_map={'Aktual':C_KOMP,'Toleransi':C_XYZ})
            fw.update_traces(texttemplate='%{y:.1f} mnt',textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{y:.2f} menit<extra></extra>')
            fw.update_yaxes(range=[0,dft2['TL6'].mean()*1.7])
            st.plotly_chart(elo(fw,"⏳ Teller: Aktual vs Toleransi"),use_container_width=True)
            ib(f"Teller: **{dft2['TL5'].mean():.1f} mnt** aktual (toleransi {dft2['TL6'].mean():.1f} mnt).")
    with wt2:
        dfc2=df[df['PANEL']=='CS'].dropna(subset=['CS5','CS6'])
        if len(dfc2)>0:
            wd2=pd.DataFrame({'Metrik':['Aktual','Toleransi'],'Menit':[dfc2['CS5'].mean(),dfc2['CS6'].mean()]})
            fw2=px.bar(wd2,x='Metrik',y='Menit',color='Metrik',text='Menit',
                color_discrete_map={'Aktual':C_KOMP,'Toleransi':C_XYZ})
            fw2.update_traces(texttemplate='%{y:.1f} mnt',textposition='outside',
                hovertemplate='<b>%{x}</b><br>%{y:.2f} menit<extra></extra>')
            fw2.update_yaxes(range=[0,dfc2['CS6'].mean()*1.7])
            st.plotly_chart(elo(fw2,"⏳ CS: Aktual vs Toleransi"),use_container_width=True)
    with wt3:
        jt=df['TL1'].dropna().value_counts().reset_index(); jt.columns=['Jam','Teller']
        jc=df['CS1'].dropna().value_counts().reset_index(); jc.columns=['Jam','CS']
        jd=pd.merge(jt,jc,on='Jam',how='outer').fillna(0)
        fj=go.Figure()
        fj.add_trace(go.Bar(name='Teller',x=jd['Jam'],y=jd['Teller'],marker_color=C_XYZ,
            hovertemplate='<b>%{x}</b><br>Teller: %{y:.0f} resp.<extra></extra>'))
        fj.add_trace(go.Bar(name='CS',x=jd['Jam'],y=jd['CS'],marker_color=C_KOMP,
            hovertemplate='<b>%{x}</b><br>CS: %{y:.0f} resp.<extra></extra>'))
        fj.update_layout(barmode='group',xaxis_tickangle=-25,height=300)
        st.plotly_chart(elo(fj,"⏰ Distribusi Jam Sibuk"),use_container_width=True)
        if len(jt)>0: ib(f"Jam paling sibuk Teller: **{jt.iloc[0]['Jam']}**.")

    sh("⏱️ Waktu Tunggu per Cabang")
    wtp=st.radio("Panel:",["Teller","CS"],horizontal=True,key="rd_wtp2")
    if wtp=="Teller":
        wcd=df[df['PANEL']=='Teller'].groupby('CABANG').agg(Aktual=('TL5','mean'),Toleransi=('TL6','mean'),N=('SERIAL','count')).reset_index()
    else:
        wcd=df[df['PANEL']=='CS'].groupby('CABANG').agg(Aktual=('CS5','mean'),Toleransi=('CS6','mean'),N=('SERIAL','count')).reset_index()
    wcd=wcd[wcd['N']>=5].sort_values('Aktual',ascending=False)
    if len(wcd)>0:
        fwc=go.Figure()
        fwc.add_trace(go.Bar(name='Aktual',x=wcd['CABANG'],y=wcd['Aktual'],marker_color=C_KOMP,
            customdata=wcd['N'],hovertemplate='<b>%{x}</b><br>Aktual: %{y:.1f} mnt<br>N: %{customdata}<extra></extra>'))
        fwc.add_trace(go.Bar(name='Toleransi',x=wcd['CABANG'],y=wcd['Toleransi'],marker_color=C_XYZ,opacity=0.7,
            hovertemplate='<b>%{x}</b><br>Toleransi: %{y:.1f} mnt<extra></extra>'))
        fwc.update_layout(barmode='overlay',xaxis_tickangle=-30,height=380)
        st.plotly_chart(elo(fwc,f"Waktu Tunggu {wtp} per Cabang"),use_container_width=True)
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
    blb=[slbl(c,col_map,30) for c in bic]
    biv=[df[c].mean() for c in bic]
    bxv=[df[c].mean() for c in bxc[:len(bic)]]
    bkv=[df_hk[c].mean() if c in df_hk.columns else np.nan for c in bkc[:len(bic)]]

    sh("📊 Perbandingan 24 Atribut Brand — XYZ vs Kompetitor")
    brand_view=st.radio("Tampilan:",["Bar Comparison","Radar Chart"],horizontal=True,key="rd_brand3")
    st.markdown("<br>",unsafe_allow_html=True)

    if brand_view=="Bar Comparison":
        # Full width, tinggi proporsional agar label tidak terpotong
        fbc=go.Figure()
        fbc.add_trace(go.Bar(name='Bank XYZ',y=blb,x=bxv,orientation='h',marker_color=C_XYZ,
            customdata=blb_full,
            hovertemplate='<b>%{customdata}</b><br>XYZ: %{x:.2f}<extra></extra>'))
        fbc.add_trace(go.Bar(name=target_komp,y=blb,
            x=[v if not np.isnan(v) else None for v in bkv],
            orientation='h',marker_color=C_KOMP,opacity=0.85,
            customdata=blb_full,
            hovertemplate=f'<b>%{{customdata}}</b><br>{target_komp}: %{{x:.2f}}<extra></extra>'))
        fbc.update_layout(barmode='group',xaxis_range=[4,6.8],
            height=max(720,len(blb)*30),
            yaxis=dict(automargin=True,tickfont=dict(size=10)),
            margin=dict(t=46,b=14,l=230,r=20))
        st.plotly_chart(elo(fbc),use_container_width=True)
    else:
        blb_r=[l[:13]+'…' if len(l)>13 else l for l in blb]
        fr=go.Figure()
        fr.add_trace(go.Scatterpolar(r=bxv,theta=blb_r,fill='toself',name='Bank XYZ',
            line_color=C_XYZ,fillcolor='rgba(59,130,246,0.10)',
            hovertemplate='<b>%{theta}</b><br>XYZ: %{r:.2f}<extra></extra>'))
        fr.add_trace(go.Scatterpolar(r=bkv,theta=blb_r,fill='toself',name=target_komp,
            line_color=C_KOMP,fillcolor='rgba(248,113,113,0.07)',
            hovertemplate=f'<b>%{{theta}}</b><br>{target_komp}: %{{r:.2f}}<extra></extra>'))
        fr.update_layout(
            polar=dict(bgcolor=SURF,
                radialaxis=dict(visible=True,range=[3,6],gridcolor=GRID,tickfont=dict(color='#3D5470',size=8)),
                angularaxis=dict(tickfont=dict(color='#6A8AAA',size=8))),
            legend=dict(orientation="h",y=-0.1),height=580,margin=dict(t=40,b=80,l=80,r=80))
        st.info("💡 Hover ke titik untuk nama atribut lengkap. Gunakan 'Bar Comparison' untuk label yang lebih jelas.")
        st.plotly_chart(elo(fr),use_container_width=True)

    sh("🎯 IPA Matrix 24 Atribut Brand")
    bdf=pd.DataFrame({'Label':blb,'Label_Full':blb_full,'Importance':biv,'Satisfaction':bxv})
    mb,sb2_v=np.nanmean(biv),np.nanmean(bxv)
    def bq(r):
        if r['Importance']>=mb and r['Satisfaction']<sb2_v: return "⚠️ Perbaiki"
        elif r['Importance']>=mb: return "🌟 Pertahankan"
        elif r['Satisfaction']<sb2_v: return "💤 Rendah"
        else: return "✅ Berlebihan"
    bdf['Kuadran']=bdf.apply(bq,axis=1)
    qcb={"⚠️ Perbaiki":C_KOMP,"🌟 Pertahankan":"#34D399","💤 Rendah":C_IMP,"✅ Berlebihan":"#FBBF24"}
    fib=px.scatter(bdf,x='Satisfaction',y='Importance',text='Label',color='Kuadran',color_discrete_map=qcb)
    fib.update_traces(marker=dict(size=10),textposition='top center',textfont=dict(size=8),
        customdata=bdf['Label_Full'].values,
        hovertemplate='<b>%{customdata}</b><br>Imp: %{y:.2f}<br>Sat: %{x:.2f}<extra></extra>',
        selector=dict(mode='markers+text'))
    fib.add_vline(x=sb2_v,line_dash="dash",line_color="#162038")
    fib.add_hline(y=mb,line_dash="dash",line_color="#162038")
    fib.update_layout(height=540)
    st.plotly_chart(elo(fib),use_container_width=True)

    sh("📊 Gap Kompetitif per Atribut Brand")
    gb=pd.DataFrame({'Atribut':blb,'Atribut_Full':blb_full,'XYZ':bxv,'Komp':bkv,
        'Gap':[x-k for x,k in zip(bxv,bkv)]}).sort_values('Gap')
    gb['Gap']=gb['Gap'].clip(-2.5,2.5)  # clip anomali
    gb['W']=np.where(gb['Gap']<0,C_KOMP,C_XYZ)
    fgb=px.bar(gb,x='Gap',y='Atribut',orientation='h',text='Gap')
    fgb.update_traces(marker_color=gb['W'],texttemplate='%{x:.2f}',textposition='outside',
        customdata=gb[['Atribut_Full','XYZ','Komp']].values,
        hovertemplate='<b>%{customdata[0]}</b><br>XYZ: %{customdata[1]:.2f}<br>Komp: %{customdata[2]:.2f}<br>Gap: %{x:.2f}<extra></extra>')
    fgb.update_layout(height=max(560,len(gb)*24),yaxis=dict(tickfont=dict(size=9),automargin=True))
    st.plotly_chart(elo(fgb,f"Gap Brand XYZ vs {target_komp} (+ = XYZ unggul)"),use_container_width=True)
    tg=gb.nlargest(1,'Gap').iloc[0]; bg2=gb.nsmallest(1,'Gap').iloc[0]
    ib(f"Keunggulan terbesar XYZ: **{tg['Atribut_Full']}** (+{tg['Gap']:.2f}). Perlu ditingkatkan: **{bg2['Atribut_Full']}** ({bg2['Gap']:.2f}).")

    # Share of Wallet — bar full width dulu, lalu 2 pie berdampingan
    sh("🏦 Share of Wallet")
    bo=df['A1AX'].dropna().str.split(';').explode().str.strip()
    bo=bo[bo!=''].value_counts().head(10).reset_index(); bo.columns=['Bank','N']
    fsw=px.bar(bo,x='N',y='Bank',orientation='h',color_discrete_sequence=[C_XYZ],text='N')
    fsw.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
    fsw.update_layout(yaxis=dict(automargin=True))
    st.plotly_chart(elo(fsw,"Bank Lain yang Aktif Digunakan",380),use_container_width=True)

    sw1,sw2=st.columns(2)
    with sw1:
        sp=df['A1B'].value_counts().reset_index(); sp.columns=['Bank','N']
        fsp=px.pie(sp,values='N',names='Bank',hole=0.55,color_discrete_sequence=px.colors.qualitative.Set2)
        fsp.update_traces(textposition='outside',textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>%{value} resp.<br>%{percent}<extra></extra>')
        st.plotly_chart(elo(fsp,"Bank Utama Simpan Dana",380),use_container_width=True)
    with sw2:
        tr=df['A1C'].value_counts().reset_index(); tr.columns=['Bank','N']
        ftr2=px.pie(tr,values='N',names='Bank',hole=0.55,color_discrete_sequence=px.colors.qualitative.Pastel)
        ftr2.update_traces(textposition='outside',textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>%{value} resp.<br>%{percent}<extra></extra>')
        st.plotly_chart(elo(ftr2,"Bank Utama Bertransaksi",380),use_container_width=True)
    xs=(df['A1B']=='Bank XYZ').mean()*100; xt=(df['A1C']=='Bank XYZ').mean()*100
    rv=bo.iloc[0]['Bank'] if len(bo)>0 else "N/A"
    ib(f"XYZ rekening utama simpan: **{xs:.0f}%**, utama transaksi: **{xt:.0f}%**. Kompetitor terbesar: **{rv}**.")

# ═══════════════════════════════════════════════════════
# TAB 4 — TOUCHPOINT & IPA
# ═══════════════════════════════════════════════════════
with t4:
    sh("🎯 Analisis Touchpoint Interaktif")
    tc1,tc2,tc3=st.columns(3)
    with tc1: sel_tp=st.selectbox("Touchpoint:",key="sb_tp4",options=list(DM.keys()))
    with tc2: tpv=st.radio("Tampilan:",["IPA Scatter","Bar"],horizontal=True,key="rd_tp4")
    with tc3:
        tpc=st.selectbox("Filter Cabang:",key="sb_tpc4",
            options=["Semua"]+sorted(df['CABANG'].unique().tolist()))

    dft4=df if tpc=="Semua" else df[df['CABANG']==tpc]
    dftk4=df_hk if tpc=="Semua" else df_hk[df_hk['CABANG']==tpc]
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

        if tpv=="IPA Scatter":
            idf4=pd.DataFrame({'Item':lb4,'Item_Full':lb4_full,'Importance':iv4,'Satisfaction':xv4})
            mi4,ms4=np.nanmean(iv4),np.nanmean(xv4)
            def tq4(r):
                if r['Importance']>=mi4 and r['Satisfaction']<ms4: return "⚠️ Perbaiki"
                elif r['Importance']>=mi4: return "🌟 Pertahankan"
                elif r['Satisfaction']<ms4: return "💤 Rendah"
                else: return "✅ Berlebihan"
            idf4['Kuadran']=idf4.apply(tq4,axis=1)
            qtc={"⚠️ Perbaiki":C_KOMP,"🌟 Pertahankan":"#34D399","💤 Rendah":C_IMP,"✅ Berlebihan":"#FBBF24"}
            fi4=px.scatter(idf4,x='Satisfaction',y='Importance',text='Item',color='Kuadran',color_discrete_map=qtc)
            if kv4:
                fi4.add_trace(go.Scatter(x=kv4,y=iv4,mode='markers',name=target_komp,text=lb4_full,
                    marker=dict(size=10,symbol='x',color=C_KOMP,line=dict(width=2)),
                    hovertemplate='<b>%{text}</b><br>Sat.Komp: %{x:.2f}<extra></extra>'))
            fi4.update_traces(marker=dict(size=10),textposition='top center',textfont=dict(size=9),
                customdata=idf4['Item_Full'].values,
                hovertemplate='<b>%{customdata}</b><br>Imp: %{y:.2f}<br>Sat: %{x:.2f}<extra></extra>',
                selector=dict(mode='markers+text'))
            fi4.add_vline(x=ms4,line_dash="dash",line_color="#162038")
            fi4.add_hline(y=mi4,line_dash="dash",line_color="#162038")
            fi4.update_layout(height=520)
            st.plotly_chart(elo(fi4,f"IPA — {sel_tp}"),use_container_width=True)

            # Prioritas 4 kolom di bawah scatter
            sh("📋 Prioritas per Kuadran")
            qcols4=st.columns(4)
            for col_q,qn,qcv in zip(qcols4,
                ["⚠️ Perbaiki","🌟 Pertahankan","💤 Rendah","✅ Berlebihan"],
                [C_KOMP,"#34D399",C_IMP,"#FBBF24"]):
                its=idf4[idf4['Kuadran']==qn]['Item_Full'].tolist()
                with col_q:
                    items_html="".join([
                        f"<div style='color:#6A8AAA!important;font-size:11px;"
                        f"padding:3px 0;border-bottom:1px solid #162038'>{it}</div>"
                        for it in its])
                    st.markdown(
                        f"<div style='background:{qcv}15;border:1px solid {qcv}40;"
                        f"border-radius:10px;padding:10px 12px;min-height:60px'>"
                        f"<div style='color:{qcv}!important;font-weight:800;font-size:11px;"
                        f"margin-bottom:8px'>{qn}</div>{items_html}</div>",
                        unsafe_allow_html=True)
        else:
            fb4=go.Figure()
            fb4.add_trace(go.Bar(name='Importance',x=lb4,y=iv4,marker_color=C_IMP,opacity=0.72,
                hovertemplate='<b>%{x}</b><br>Importance: %{y:.2f}<extra></extra>'))
            fb4.add_trace(go.Bar(name='Sat. XYZ',x=lb4,y=xv4,marker_color=C_XYZ,
                hovertemplate='<b>%{x}</b><br>Sat. XYZ: %{y:.2f}<extra></extra>'))
            if kv4:
                fb4.add_trace(go.Bar(name=f'Sat. {target_komp}',x=lb4,y=kv4,marker_color=C_KOMP,opacity=0.85,
                    hovertemplate=f'<b>%{{x}}</b><br>Sat.{target_komp}: %{{y:.2f}}<extra></extra>'))
            fb4.update_layout(barmode='group',yaxis_range=[3,6.8],xaxis_tickangle=-30,height=450,
                xaxis=dict(automargin=True,tickfont=dict(size=9)))
            st.plotly_chart(elo(fb4,f"Comparison — {sel_tp}"),use_container_width=True)

        if kv4:
            sh("📊 Gap Kompetitif per Item")
            gt4=pd.DataFrame({'Item':lb4,'Item_Full':lb4_full,'Gap':[x-k for x,k in zip(xv4,kv4)],
                'XYZ':xv4,'Komp':kv4}).sort_values('Gap')
            gt4['W']=np.where(gt4['Gap']<0,C_KOMP,C_XYZ)
            fg4=px.bar(gt4,x='Gap',y='Item',orientation='h',text='Gap')
            fg4.update_traces(marker_color=gt4['W'],texttemplate='%{x:.2f}',textposition='outside',
                customdata=gt4[['Item_Full','XYZ','Komp']].values,
                hovertemplate='<b>%{customdata[0]}</b><br>XYZ: %{customdata[1]:.2f}<br>Komp: %{customdata[2]:.2f}<br>Gap: %{x:.2f}<extra></extra>')
            fg4.update_layout(height=max(420,mt4*26),yaxis=dict(automargin=True,tickfont=dict(size=9)))
            st.plotly_chart(elo(fg4,f"Gap XYZ vs {target_komp} — {sel_tp}"),use_container_width=True)

    sh("🔄 Skor Layanan per Jenis Transaksi")
    d1m={'TELLER':df[df['D1_TYPE']=='TELLER']['OVR_TELLER_XYZ'].mean(),
         'CS':df[df['D1_TYPE']=='CS']['OVR_CS_XYZ'].mean(),
         'KEDUANYA':df[df['D1_TYPE']=='BOTH'][['OVR_TELLER_XYZ','OVR_CS_XYZ']].mean(axis=1).mean()}
    d1df=pd.DataFrame({'Jenis':list(d1m.keys()),'Skor':list(d1m.values())})
    fd1=px.bar(d1df,x='Jenis',y='Skor',color='Jenis',text='Skor',
        color_discrete_map={'TELLER':C_XYZ,'CS':C_KOMP,'KEDUANYA':'#FBBF24'})
    fd1.update_traces(texttemplate='%{y:.2f}',textposition='outside',
        hovertemplate='<b>%{x}</b><br>Skor: %{y:.3f}<extra></extra>')
    fd1.update_yaxes(range=[4.5,6.5])
    st.plotly_chart(elo(fd1,""),use_container_width=True)

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
    epa=np.nanmean(epv); ena=np.nanmean(env)
    epka=np.nanmean(epkv); enka=np.nanmean(enkv)

    ek=st.columns(4)
    ek[0].markdown(card("Emosi Positif XYZ",f"{epa:.2f}","blue","Avg 9 dimensi",delta=epa-epka,dlbl="vs Komp"),unsafe_allow_html=True)
    ek[1].markdown(card("Emosi Negatif XYZ",f"{ena:.2f}","amber","↓ makin rendah makin baik"),unsafe_allow_html=True)
    ek[2].markdown(card("Emosi Positif Komp",f"{epka:.2f}","red","Avg 9 dimensi"),unsafe_allow_html=True)
    ek[3].markdown(card("Emosi Negatif Komp",f"{enka:.2f}","amber","↓ makin rendah makin baik"),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ib(f"XYZ unggul di emosi positif ({epa:.2f} vs {epka:.2f}) dan lebih rendah di emosi negatif ({ena:.2f} vs {enka:.2f}) — pengalaman nasabah XYZ secara keseluruhan lebih positif.")

    ec1,ec2=st.columns(2)
    with ec1:
        sh("😊 Emosi Positif XYZ vs Kompetitor")
        fep=go.Figure()
        fep.add_trace(go.Bar(name='XYZ',x=elp[:len(epv)],y=epv,marker_color=C_XYZ,
            text=np.round(epv,2),textposition='outside',
            hovertemplate='<b>%{x}</b><br>XYZ: %{y:.3f}<extra></extra>'))
        fep.add_trace(go.Bar(name=target_komp,x=elp[:len(epkv)],y=epkv,marker_color=C_KOMP,
            text=np.round(epkv,2),textposition='outside',
            hovertemplate=f'<b>%{{x}}</b><br>{target_komp}: %{{y:.3f}}<extra></extra>'))
        fep.update_layout(barmode='group',yaxis_range=[3,6.8],xaxis_tickangle=-20,height=370)
        st.plotly_chart(elo(fep),use_container_width=True)
    with ec2:
        sh("😠 Emosi Negatif (↓ Lebih Rendah = Lebih Baik)")
        fen=go.Figure()
        fen.add_trace(go.Bar(name='XYZ',x=eln[:len(env)],y=env,marker_color=C_XYZ,
            text=np.round(env,2),textposition='outside',
            hovertemplate='<b>%{x}</b><br>XYZ: %{y:.3f}<extra></extra>'))
        fen.add_trace(go.Bar(name=target_komp,x=eln[:len(enkv)],y=enkv,marker_color=C_KOMP,
            text=np.round(enkv,2),textposition='outside',
            hovertemplate=f'<b>%{{x}}</b><br>{target_komp}: %{{y:.3f}}<extra></extra>'))
        fen.update_layout(barmode='group',yaxis_range=[1,4],xaxis_tickangle=-20,height=370)
        st.plotly_chart(elo(fen),use_container_width=True)

    sh("💎 Brand Equity — 15 Atribut")
    hxc=[c for c in["T_H1A_2","T_H1A_5","T_H1A_8","T_H1A_11","T_H1A_14","T_H1A_17","T_H1A_20","T_H1A_23","T_H1A_26","T_H1A_29","T_H1A_32","T_H1A_35","T_H1A_38","T_H1A_41","T_H1A_44"] if c in df.columns]
    hkc=[c for c in["T_H1A_3","T_H1A_6","T_H1A_9","T_H1A_12","T_H1A_15","T_H1A_18","T_H1A_21","T_H1A_24","T_H1A_27","T_H1A_30","T_H1A_33","T_H1A_36","T_H1A_39","T_H1A_42","T_H1A_45"] if c in df_hk.columns]
    hlb=["Tetap Gunakan","Kemudahan Transaksi","Digunakan Banyak","Keuntungan Finansial","Produk Lengkap","Promo Gaya Hidup","Kecepatan","Rasa Aman","Kenyamanan","Merasa Dihargai","Bangga","Modern","Bank Turun-Temurun","Cukup Satu Bank","Bergengsi"]
    hxv=[df[c].mean() for c in hxc]; hkv=[df_hk[c].mean() if c in df_hk.columns else np.nan for c in hkc]
    lhb=hlb[:min(len(hxv),len(hlb))]
    fhe=go.Figure()
    fhe.add_trace(go.Bar(name='Bank XYZ',x=lhb,y=hxv[:len(lhb)],marker_color=C_XYZ,
        text=np.round(hxv[:len(lhb)],2),textposition='outside',
        hovertemplate='<b>%{x}</b><br>XYZ: %{y:.3f}<extra></extra>'))
    fhe.add_trace(go.Bar(name=target_komp,x=lhb,y=hkv[:len(lhb)],marker_color=C_KOMP,
        text=np.round(hkv[:len(lhb)],2),textposition='outside',
        hovertemplate=f'<b>%{{x}}</b><br>{target_komp}: %{{y:.3f}}<extra></extra>'))
    fhe.update_layout(barmode='group',yaxis_range=[3,6.8],xaxis_tickangle=-28,height=400)
    st.plotly_chart(elo(fhe,"Brand Equity XYZ vs Kompetitor"),use_container_width=True)

    sh("📈 Korelasi Emosi vs Outcome")
    epas=df[[c for c in epc if c in df.columns]].mean(axis=1)
    enas=df[[c for c in enc if c in df.columns]].mean(axis=1)
    cce=pd.DataFrame({'Emosi Pos':epas,'Emosi Neg':enas,'NPS':df['G1A'],'Kepuasan':df['E1A'],'Loyalitas':df['F1A']}).corr()
    e_c1,e_c2,e_c3=st.columns(3)
    with e_c1:
        fec=px.imshow(cce,text_auto=".2f",color_continuous_scale='RdBu',aspect='auto',zmin=-1,zmax=1)
        fec.update_traces(hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.3f}<extra></extra>')
        st.plotly_chart(elo(fec,h=300),use_container_width=True)
    with e_c2:
        s1df=df[['G1A','G1A_CAT']].copy(); s1df['Emosi Pos']=epas
        fs1=px.scatter(s1df,x='Emosi Pos',y='G1A',color='G1A_CAT',color_discrete_map=NPS_C,
            trendline='ols',opacity=0.5,labels={'G1A':'NPS'})
        fs1.update_traces(hovertemplate='Emosi Pos: %{x:.2f}<br>NPS: %{y:.1f}<extra></extra>',selector=dict(mode='markers'))
        st.plotly_chart(elo(fs1,"Emosi Positif vs NPS",300),use_container_width=True)
    with e_c3:
        s2df=df[['F1A','G1A_CAT']].copy(); s2df['Emosi Pos']=epas
        fs2=px.scatter(s2df,x='Emosi Pos',y='F1A',color='G1A_CAT',color_discrete_map=NPS_C,
            trendline='ols',opacity=0.5,labels={'F1A':'Loyalitas'})
        fs2.update_traces(hovertemplate='Emosi Pos: %{x:.2f}<br>Loyalitas: %{y:.2f}<extra></extra>',selector=dict(mode='markers'))
        st.plotly_chart(elo(fs2,"Emosi Positif vs Loyalitas",300),use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 6 — DIGITALISASI
# ═══════════════════════════════════════════════════════
with t6:
    dc=[c for c in["T_J1_1","T_J1_2","T_J1_3","T_J1_4","T_J1_5"] if c in df.columns]
    dlb=["Digitalisasi","Digital Signage","Smart Table","Tablet Survey","Akses Cabang"]
    dv=[df[c].mean() for c in dc]
    g_dv=[df_raw[c].mean() if c in df_raw.columns else np.nan for c in dc]
    sh("📱 KPI Digitalisasi vs Global")
    dkk=st.columns(len(dc))
    for i,(cw,lb,vl,gvl) in enumerate(zip(dkk,dlb,dv,g_dv)):
        col2="green" if vl>=5.5 else ("amber" if vl>=4.5 else "red")
        cw.markdown(card(lb,f"{vl:.2f}",col2,"Skala 1–6",
            delta=vl-gvl if not np.isnan(gvl) else None,dlbl="vs Global"),unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    dg1,dg2=st.columns(2)
    with dg1:
        fdg=px.bar(x=dv,y=dlb[:len(dv)],orientation='h',color=dv,color_continuous_scale='Blues',text=np.round(dv,2))
        fdg.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<extra></extra>')
        fdg.update_xaxes(range=[3,6.8])
        st.plotly_chart(elo(fdg,"Skor Persepsi Digitalisasi"),use_container_width=True)
    with dg2:
        if "T_J1_1" in df.columns:
            dp=df.groupby('PROV')["T_J1_1"].mean().reset_index(); dp.columns=['Provinsi','Skor']; dp=dp.sort_values('Skor')
            fdp=px.bar(dp,x='Skor',y='Provinsi',orientation='h',color='Skor',color_continuous_scale='Blues',text='Skor')
            fdp.update_traces(texttemplate='%{x:.2f}',textposition='outside',hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<extra></extra>')
            fdp.update_xaxes(range=[3,6.8])
            st.plotly_chart(elo(fdp,"Digitalisasi per Provinsi"),use_container_width=True)

    sh("🖥️ Sarana Elektronik Layanan")
    slc=[f"T_SL2_{i}" for i in range(1,17) if f"T_SL2_{i}" in df.columns]
    slb=[slbl(c,col_map,38) for c in slc]; slv=[df[c].mean() for c in slc]
    sl1,sl2=st.columns(2)
    with sl1:
        fsl=px.bar(x=slv,y=slb[:len(slv)],orientation='h',color=slv,color_continuous_scale='Teal',text=np.round(slv,2))
        fsl.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>Skor: %{x:.3f}<extra></extra>')
        fsl.update_xaxes(range=[3,6.8])
        fsl.update_layout(yaxis=dict(automargin=True,tickfont=dict(size=9)),height=max(380,len(slv)*28))
        st.plotly_chart(elo(fsl,"Ketersediaan & Fungsi Sarana"),use_container_width=True)
    with sl2:
        if "T_J1_1" in df.columns:
            scd=df[['T_J1_1','E1A','G1A_CAT']].dropna()
            fds=px.scatter(scd,x='T_J1_1',y='E1A',color='G1A_CAT',color_discrete_map=NPS_C,
                trendline='ols',opacity=0.5,labels={'T_J1_1':'Persepsi Digitalisasi','E1A':'Kepuasan'})
            fds.update_traces(hovertemplate='Digital: %{x:.2f}<br>Kepuasan: %{y:.2f}<extra></extra>',selector=dict(mode='markers'))
            st.plotly_chart(elo(fds,"Digitalisasi vs Kepuasan"),use_container_width=True)

    sh("📋 E-Form & Awareness")
    ef1,ef2=st.columns(2)
    with ef1:
        if 'D2' in df.columns:
            ef=df['D2'].dropna().value_counts().reset_index(); ef.columns=['Status','N']
            fee=px.pie(ef,values='N',names='Status',hole=0.55,color_discrete_sequence=['#8B5CF6','#3B82F6','#F43F5E'])
            fee.update_traces(textposition='outside',textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>%{value} resp.<br>%{percent}<extra></extra>')
            st.plotly_chart(elo(fee,"Penggunaan E-Form"),use_container_width=True)
    with ef2:
        if 'D4' in df.columns:
            ea=df['D4'].dropna().value_counts().reset_index(); ea.columns=['Status','N']
            fea=px.bar(ea,x='N',y='Status',orientation='h',color_discrete_sequence=['#0EA5E9'],text='N')
            fea.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
            st.plotly_chart(elo(fea,"Awareness E-Form"),use_container_width=True)

    j2m={"T_J2_1":"Digitalisasi Layanan","T_J2_2":"Digital Signage","T_J2_3":"Smart Table","T_J2_4":"Tablet Survey","T_J2_5":"Akses Cabang"}
    j2l=st.selectbox("📝 Topik Saran Perbaikan:",key="sb_j26",options=list(j2m.values()))
    j2c=[k for k,v in j2m.items() if v==j2l][0]
    if j2c in df.columns:
        sr=df[j2c].dropna().value_counts().reset_index(); sr.columns=['Saran','N']
        sr=sr[~sr['Saran'].str.strip().isin(['','None','Tidak Ada','Tidak Ada /  Tidak Tahu'])]
        if len(sr)>0: st.dataframe(sr,use_container_width=True,hide_index=True,height=200)

# ═══════════════════════════════════════════════════════
# TAB 7 — CLUSTERING
# ═══════════════════════════════════════════════════════
with t7:
    sh("🧩 Segmentasi Nasabah via K-Means Clustering")
    if not SK_OK:
        st.warning("scikit-learn tidak terinstal. Tambahkan ke requirements.txt.")
    else:
        cl1,cl2,cl3=st.columns(3)
        with cl1: n_cl=st.slider("Jumlah Cluster:",2,6,3,key="sl_ncl")
        with cl2:
            cl_feat_opts={'NPS':'G1A','Kepuasan':'E1A','Loyalitas':'F1A',
                'OVR Teller':'OVR_TELLER_XYZ','OVR CS':'OVR_CS_XYZ',
                'OVR ATM':'OVR_ATM_XYZ','OVR Sekuriti':'OVR_SEKURITI_XYZ',
                'OVR Kantor Cabang':'OVR_KC_XYZ','Usia Numerik':'S2_1'}
            sel_feats=st.multiselect("Fitur Clustering:",key="ms_clf",
                options=list(cl_feat_opts.keys()),
                default=['NPS','Kepuasan','Loyalitas','OVR Teller','OVR CS'])
        with cl3:
            cl_color=st.selectbox("Warna marker by:",key="sb_clc",
                options=["Cluster","NPS Category","Provinsi","Gender"])

        df_cl=df.copy()
        epc_cl=[c for c in["T_I1A_2","T_I1A_5","T_I1A_8","T_I1A_11","T_I1A_14","T_I1A_17","T_I1A_20","T_I1A_23","T_I1A_26"] if c in df_cl.columns]
        if epc_cl:
            df_cl['Emosi_Pos']=df_cl[epc_cl].mean(axis=1)
            cl_feat_opts['Emosi Positif']='Emosi_Pos'

        feat_cols=[cl_feat_opts[f] for f in sel_feats if f in cl_feat_opts and cl_feat_opts[f] in df_cl.columns]
        df_cl_valid=df_cl[feat_cols].dropna()

        if len(df_cl_valid)<n_cl*5:
            st.warning("Data tidak cukup untuk clustering dengan pilihan ini.")
        else:
            sc=StandardScaler()
            X=sc.fit_transform(df_cl_valid)
            km=KMeans(n_clusters=n_cl,random_state=42,n_init=10)
            df_cl.loc[df_cl_valid.index,'Cluster']=km.fit_predict(X).astype(str)
            df_cl['Cluster']=df_cl['Cluster'].fillna('N/A')

            with st.expander("📊 Elbow Chart — Pilih Jumlah Cluster Optimal"):
                inertias=[]
                for k in range(2,min(10,len(df_cl_valid)//5+1)):
                    km_e=KMeans(n_clusters=k,random_state=42,n_init=10); km_e.fit(X); inertias.append(km_e.inertia_)
                fel=px.line(x=list(range(2,min(10,len(df_cl_valid)//5+1))),y=inertias,markers=True,
                    labels={'x':'Jumlah Cluster','y':'Inertia'},color_discrete_sequence=[C_XYZ])
                fel.update_traces(hovertemplate='K=%{x}<br>Inertia: %{y:.0f}<extra></extra>')
                st.plotly_chart(elo(fel,"Pilih K di titik siku",300),use_container_width=True)

            pca=PCA(n_components=2,random_state=42); X_pca=pca.fit_transform(X)
            var1,var2=pca.explained_variance_ratio_
            df_pca=df_cl.loc[df_cl_valid.index].copy()
            df_pca['PC1']=X_pca[:,0]; df_pca['PC2']=X_pca[:,1]
            color_map={"Cluster":"Cluster","NPS Category":"G1A_CAT","Provinsi":"PROV","Gender":"S1"}
            color_col=color_map[cl_color]
            color_d=NPS_C if cl_color=="NPS Category" else {u:c for u,c in zip(df_pca[color_col].dropna().unique(),px.colors.qualitative.Plotly)}
            cd_exist=[c for c in['CABANG','PROV','G1A','E1A','F1A','G1A_CAT','Cluster'] if c in df_pca.columns]
            fpc=px.scatter(df_pca,x='PC1',y='PC2',color=color_col,color_discrete_map=color_d,
                symbol='Cluster',opacity=0.7,
                labels={'PC1':f'PC1 ({var1*100:.1f}% var)','PC2':f'PC2 ({var2*100:.1f}% var)'})
            fpc.update_traces(marker=dict(size=6),customdata=df_pca[cd_exist].values,
                hovertemplate='<b>%{customdata[0]}</b><br>Provinsi: %{customdata[1]}<br>NPS: %{customdata[2]:.1f}<br>Kepuasan: %{customdata[3]:.2f}<br>Loyalitas: %{customdata[4]:.2f}<br>Cluster: %{customdata[6]}<extra></extra>')
            fpc.update_layout(height=500)
            st.plotly_chart(elo(fpc,f"PCA 2D Cluster Plot — Variance Explained: {(var1+var2)*100:.1f}%"),use_container_width=True)

            sh("📋 Profil Rata-rata per Cluster")
            prof_cols=[c for c in feat_cols if c in df_pca.columns]
            extra_c=[c for c in['G1A','E1A','F1A'] if c not in feat_cols and c in df_pca.columns]
            prof=df_pca.groupby('Cluster')[prof_cols+extra_c].mean().round(2)
            prof['N Responden']=df_pca.groupby('Cluster').size()
            st.dataframe(prof,use_container_width=True)

            if len(feat_cols)>=3:
                sh("🕸️ Radar Profil per Cluster")
                fig_cl_r=go.Figure()
                cl_colors_r=px.colors.qualitative.Plotly
                for ci,cl_name in enumerate(sorted(df_pca['Cluster'].dropna().unique())):
                    cl_data=df_pca[df_pca['Cluster']==cl_name][feat_cols].mean().values.tolist()
                    fig_cl_r.add_trace(go.Scatterpolar(r=cl_data,theta=sel_feats[:len(feat_cols)],
                        fill='toself',name=f'Cluster {cl_name}',
                        line_color=cl_colors_r[ci%len(cl_colors_r)],
                        hovertemplate='<b>%{theta}</b><br>Nilai: %{r:.2f}<extra></extra>'))
                fig_cl_r.update_layout(
                    polar=dict(bgcolor=SURF,
                        radialaxis=dict(visible=True,gridcolor=GRID,tickfont=dict(color='#3D5470',size=8)),
                        angularaxis=dict(tickfont=dict(color='#6A8AAA',size=10))),
                    height=440,margin=dict(t=40,b=60,l=60,r=60))
                st.plotly_chart(elo(fig_cl_r,"Radar Profil per Cluster"),use_container_width=True)

            sh("📊 Distribusi NPS per Cluster")
            nps_cl=df_pca.groupby(['Cluster','G1A_CAT']).size().reset_index(name='N')
            fnc=px.bar(nps_cl,x='Cluster',y='N',color='G1A_CAT',color_discrete_map=NPS_C,
                barmode='stack',text='N',labels={'N':'Jumlah Responden'})
            fnc.update_traces(textposition='inside',
                hovertemplate='Cluster %{x}<br>Kategori: %{fullData.name}<br>N: %{y}<extra></extra>')
            st.plotly_chart(elo(fnc,"Komposisi Kategori NPS per Cluster"),use_container_width=True)

            if 'G1A' in df_pca.columns:
                cl_prof=df_pca.groupby('Cluster')[['G1A','E1A','F1A']].mean()
                best_cl=cl_prof['G1A'].idxmax(); worst_cl=cl_prof['G1A'].idxmin()
                ib(f"Cluster **{best_cl}** NPS tertinggi ({cl_prof.loc[best_cl,'G1A']:.1f}) — segmen promoter utama. "
                   f"Cluster **{worst_cl}** NPS terendah ({cl_prof.loc[worst_cl,'G1A']:.1f}) — prioritas program retention.")

# ═══════════════════════════════════════════════════════
# TAB 8 — PROFIL & SEGMENTASI
# ═══════════════════════════════════════════════════════
with t8:
    # Baris 1: Gender + Usia (lebar berbeda)
    sh("👥 Profil Demografis")
    d1,d2=st.columns([1,2])
    with d1:
        fg8=px.pie(df,names='S1',hole=0.6,color_discrete_sequence=[C_XYZ,'#60A5FA','#A78BFA'])
        fg8.update_traces(textposition='outside',textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>%{value} resp.<br>%{percent}<extra></extra>')
        st.plotly_chart(elo(fg8,"Gender",300),use_container_width=True)
    with d2:
        ac=df['S2_2'].value_counts().reset_index(); ac.columns=['Usia','N']
        fa=px.bar(ac,x='Usia',y='N',color_discrete_sequence=[C_XYZ],text='N')
        fa.update_traces(textposition='outside',hovertemplate='<b>%{x}</b><br>%{y} responden<extra></extra>')
        fa.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(elo(fa,"Distribusi Usia",300),use_container_width=True)

    # Baris 2: Pendidikan + Pekerjaan
    d3,d4=st.columns(2)
    with d3:
        ec8=df['P3'].value_counts().reset_index(); ec8.columns=['Pend','N']
        fe8=px.bar(ec8,x='N',y='Pend',orientation='h',color_discrete_sequence=[C_XYZ],text='N')
        fe8.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fe8.update_layout(yaxis=dict(automargin=True),height=max(300,len(ec8)*36))
        st.plotly_chart(elo(fe8,"Tingkat Pendidikan"),use_container_width=True)
    with d4:
        jc8=df['P4'].value_counts().reset_index(); jc8.columns=['Pekerjaan','N']
        fj8=px.bar(jc8,x='N',y='Pekerjaan',orientation='h',color_discrete_sequence=['#A78BFA'],text='N')
        fj8.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fj8.update_layout(yaxis=dict(automargin=True),height=max(300,len(jc8)*33))
        st.plotly_chart(elo(fj8,"Pekerjaan"),use_container_width=True)

    # Baris 3: SES + Penghasilan
    ss1,ss2=st.columns(2)
    with ss1:
        sc8=df['P5'].value_counts().reset_index(); sc8.columns=['SES','N']
        fs8=px.bar(sc8,x='N',y='SES',orientation='h',color_discrete_sequence=[C_XYZ],text='N')
        fs8.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fs8.update_layout(yaxis=dict(automargin=True))
        st.plotly_chart(elo(fs8,"Tingkat Pengeluaran (SES)"),use_container_width=True)
    with ss2:
        ic8=df['P6'].value_counts().reset_index(); ic8.columns=['Penghasilan','N']
        fi8=px.bar(ic8,x='N',y='Penghasilan',orientation='h',color_discrete_sequence=['#34D399'],text='N')
        fi8.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>%{x} responden<extra></extra>')
        fi8.update_layout(yaxis=dict(automargin=True))
        st.plotly_chart(elo(fi8,"Distribusi Penghasilan"),use_container_width=True)

    sh("🗺️ Peta Geografis Responden")
    geo=df.groupby(['PROV','KABKOTA','CABANG']).agg(N=('SERIAL','count'),NPS=('G1A','mean'),
        Kepuasan=('E1A','mean'),Loyalitas=('F1A','mean')).reset_index()
    cm8=st.selectbox("Warna berdasarkan:",key="sb_cm8",options=['NPS','Kepuasan','Loyalitas','N'])
    ftr8=px.treemap(geo,path=[px.Constant("Nasional"),'PROV','KABKOTA','CABANG'],
        values='N',color=cm8,color_continuous_scale='RdYlGn',hover_data=['NPS','Kepuasan','Loyalitas'])
    ftr8.update_layout(height=500)
    st.plotly_chart(elo(ftr8,f"Treemap Geografis (Warna = {cm8})"),use_container_width=True)

    sh("🔀 Segmentasi Interaktif")
    sg1,sg2,sg3=st.columns(3)
    with sg1:
        seg=st.selectbox("Segmen by:",key="sb_sg8",options=[
            'S1 → Gender','S2_2 → Usia','S4 → Tenure','S7 → Frekuensi',
            'P3 → Pendidikan','P4 → Pekerjaan','P1 → Status Nikah','P5 → SES'])
    with sg2:
        met=st.selectbox("Metrik:",key="sb_mt8",options=[
            'G1A → NPS','E1A → Kepuasan','F1A → Loyalitas',
            'OVR_TELLER_XYZ → Teller','OVR_CS_XYZ → CS',
            'OVR_ATM_XYZ → ATM','OVR_KC_XYZ → Kantor Cabang'])
    with sg3:
        sct=st.radio("Chart:",["Bar","Box"],horizontal=True,key="rd_sct8")
    sk8=seg.split(' → ')[0].strip(); mk8=met.split(' → ')[0].strip()
    if sk8 in df.columns and mk8 in df.columns:
        if sct=="Bar":
            sa8=df.groupby(sk8)[mk8].agg(['mean','std','count']).reset_index()
            sa8.columns=['Segmen','Mean','Std','N']; sa8=sa8.sort_values('Mean')
            fs8b=px.bar(sa8,x='Mean',y='Segmen',orientation='h',color='Mean',
                color_continuous_scale='Blues',text='Mean',error_x='Std')
            fs8b.update_traces(texttemplate='%{x:.2f}',textposition='outside',
                customdata=sa8[['Std','N']].values,
                hovertemplate='<b>%{y}</b><br>Mean: %{x:.3f}<br>Std: %{customdata[0]:.3f}<br>N: %{customdata[1]}<extra></extra>')
            fs8b.update_layout(yaxis=dict(automargin=True))
        else:
            fs8b=px.box(df,x=sk8,y=mk8,color=sk8,points='outliers')
            fs8b.update_layout(xaxis_tickangle=-20)
            fs8b.update_traces(hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>')
        st.plotly_chart(elo(fs8b,f"{met.split('→')[1].strip()} per {seg.split('→')[1].strip()}",420),use_container_width=True)

    sh("🔄 Frekuensi Transaksi vs Outcome")
    fr8=df.groupby('S7').agg(NPS=('G1A','mean'),Kepuasan=('E1A','mean'),Loyalitas=('F1A','mean'),N=('SERIAL','count')).reset_index()
    ffr=go.Figure()
    for cn,cc,nl in [('NPS',C_XYZ,'NPS'),('Kepuasan','#34D399','Kepuasan'),('Loyalitas','#FBBF24','Loyalitas')]:
        ffr.add_trace(go.Bar(name=nl,x=fr8['S7'],y=fr8[cn],marker_color=cc,
            text=fr8[cn].round(2),textposition='outside',customdata=fr8['N'],
            hovertemplate=f'<b>%{{x}}</b><br>{nl}: %{{y:.2f}}<br>N: %{{customdata}}<extra></extra>'))
    ffr.update_layout(barmode='group',xaxis_tickangle=-15,height=360)
    st.plotly_chart(elo(ffr,"Frekuensi Transaksi vs Outcome"),use_container_width=True)

    sh("🎯 Tujuan Buka Rekening vs Loyalitas")
    tj=df['A2'].dropna().str.split(';').explode().str.strip()
    tjdf=tj.to_frame('Tujuan').join(df['F1A'],how='left')
    tjagg=tjdf.groupby('Tujuan').agg(Loyalitas=('F1A','mean'),N=('F1A','count')).reset_index()
    tjagg=tjagg[tjagg['N']>=10].sort_values('Loyalitas')
    ftj=px.bar(tjagg,x='Loyalitas',y='Tujuan',orientation='h',color='N',color_continuous_scale='Blues',text='Loyalitas')
    ftj.update_traces(texttemplate='%{x:.2f}',textposition='outside',
        customdata=tjagg['N'].values,
        hovertemplate='<b>%{y}</b><br>Loyalitas: %{x:.3f}<br>N: %{customdata}<extra></extra>')
    ftj.update_xaxes(range=[4.5,6.6]); ftj.update_layout(yaxis=dict(automargin=True))
    st.plotly_chart(elo(ftj,"Loyalitas per Tujuan Buka Rekening"),use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 9 — VOICE OF CUSTOMER
# ═══════════════════════════════════════════════════════
with t9:
    nv9,pp9,pv9,dv9=calc_nps(df['G1A_CAT'])
    vk=st.columns(4)
    vk[0].markdown(card("NPS Score",f"{nv9:.0f}","blue",f"{df['G1A_CAT'].eq('Promoter').sum()} promoter",delta=nv9-g_nps),unsafe_allow_html=True)
    vk[1].markdown(card("Promoter %",f"{pp9:.0f}%","green","Pasti rekomendasikan"),unsafe_allow_html=True)
    vk[2].markdown(card("Passive %",f"{pv9:.0f}%","amber","Netral"),unsafe_allow_html=True)
    vk[3].markdown(card("Detractor %",f"{dv9:.0f}%","red","Tidak rekomendasikan"),unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    vh1,vh2=st.columns(2)
    with vh1:
        fnh=px.histogram(df,x='G1A',nbins=11,color='G1A_CAT',color_discrete_map=NPS_C,labels={'G1A':'Skor NPS XYZ'})
        fnh.update_layout(bargap=0.08)
        fnh.update_traces(hovertemplate='Skor: %{x}<br>Jumlah: %{y}<extra></extra>')
        st.plotly_chart(elo(fnh,"Distribusi Skor NPS XYZ"),use_container_width=True)
    with vh2:
        if len(df_hk)>0:
            fnhk=px.histogram(df_hk,x='G1C',nbins=11,color='G1C_CAT',color_discrete_map=NPS_C,labels={'G1C':f'NPS {target_komp}'})
            fnhk.update_layout(bargap=0.08)
            fnhk.update_traces(hovertemplate='Skor: %{x}<br>Jumlah: %{y}<extra></extra>')
            st.plotly_chart(elo(fnhk,f"Distribusi Skor NPS {target_komp}"),use_container_width=True)

    sh("☁️ Analisis Teks & Wordcloud")
    wf1,wf2,wf3=st.columns(3)
    with wf1: fc9=st.selectbox("Filter Kategori NPS:",key="sb_fc9",options=["Semua","Promoter","Passive","Detractor"])
    with wf2:
        ws9=st.selectbox("Sumber Komentar:",key="sb_ws9",options=["G1B — Alasan NPS XYZ","G1D — Alasan NPS Kompetitor","E1AA — Alasan Kepuasan XYZ","E1BB — Alasan Kepuasan Komp"])
    with wf3: mwl=st.slider("Min. panjang kata:",3,7,4,key="sl_mwl9")

    cmap9={"G1B — Alasan NPS XYZ":("G1B","G1A_CAT",df),"G1D — Alasan NPS Kompetitor":("G1D","G1C_CAT",df_hk),
           "E1AA — Alasan Kepuasan XYZ":("E1AA","G1A_CAT",df),"E1BB — Alasan Kepuasan Komp":("E1BB","G1C_CAT",df_hk)}
    tc9,ca9,ds9=cmap9[ws9]
    dfw9=ds9 if fc9=="Semua" else ds9[ds9[ca9]==fc9] if ca9 in ds9.columns else ds9

    def mk_wc(series,cm='Blues',mw=80):
        parsed=clean_cmt(series); text=" ".join(parsed.astype(str).tolist()).lower()
        words=re.findall(rf'\b[a-zA-Z]{{{mwl},}}\b',text)
        wcw=[w for w in words if w not in STOP]
        if len(wcw)<5: return None,[]
        wco=WordCloud(width=720,height=340,background_color=BG,colormap=cm,max_words=mw,
            stopwords=STOP,prefer_horizontal=0.8).generate(" ".join(wcw)) if WC_OK else None
        return wco,Counter(wcw).most_common(10)

    wco9,tw9=mk_wc(dfw9[tc9] if tc9 in dfw9.columns else pd.Series(dtype=str))
    wcc1,wcc2=st.columns([1.6,1])
    with wcc1:
        if wco9 and WC_OK:
            fig_wc9,ax9=plt.subplots(figsize=(7.2,3.4),facecolor=BG)
            ax9.imshow(wco9,interpolation='bilinear'); ax9.axis('off')
            fig_wc9.tight_layout(pad=0); st.pyplot(fig_wc9); plt.close()
        else: st.info("Tidak ada teks yang cukup untuk wordcloud.")
    with wcc2:
        if tw9:
            kdf9=pd.DataFrame(tw9,columns=['Kata','Frekuensi'])
            fkw=px.bar(kdf9.sort_values('Frekuensi'),x='Frekuensi',y='Kata',orientation='h',
                color='Frekuensi',color_continuous_scale='Blues',text='Frekuensi')
            fkw.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>')
            fkw.update_layout(height=340,showlegend=False)
            st.plotly_chart(elo(fkw,"Top 10 Kata Kunci"),use_container_width=True)

    sh("💡 Tema Dominan: Promoter vs Detractor")
    pa1,pa2=st.columns(2)
    def top_theme9(series,n=8):
        parsed=clean_cmt(series); text=" ".join(parsed.astype(str).tolist()).lower()
        words=re.findall(r'\b[a-zA-Z]{4,}\b',text)
        return Counter([w for w in words if w not in STOP]).most_common(n)

    with pa1:
        st.markdown("<span style='color:#34D399;font-size:13px;font-weight:800'>🌟 Alasan Promoter</span>",unsafe_allow_html=True)
        pw9=top_theme9(df[df['G1A_CAT']=='Promoter']['G1B'] if 'G1B' in df.columns else pd.Series(dtype=str))
        if pw9:
            pdf9=pd.DataFrame(pw9,columns=['Tema','Count'])
            fp9=px.bar(pdf9,x='Count',y='Tema',orientation='h',color_discrete_sequence=['#34D399'],text='Count')
            fp9.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>')
            fp9.update_layout(height=320)
            st.plotly_chart(elo(fp9),use_container_width=True)
    with pa2:
        st.markdown("<span style='color:#F87171;font-size:13px;font-weight:800'>⚠️ Alasan Detractor</span>",unsafe_allow_html=True)
        dw9=top_theme9(df[df['G1A_CAT']=='Detractor']['G1B'] if 'G1B' in df.columns else pd.Series(dtype=str))
        if dw9:
            ddf9=pd.DataFrame(dw9,columns=['Tema','Count'])
            fd9=px.bar(ddf9,x='Count',y='Tema',orientation='h',color_discrete_sequence=[C_KOMP],text='Count')
            fd9.update_traces(textposition='outside',hovertemplate='<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>')
            fd9.update_layout(height=320)
            st.plotly_chart(elo(fd9),use_container_width=True)
    if tw9: ib(f"Kata paling dominan ({fc9}): **'{tw9[0][0]}'** ({tw9[0][1]}x). Cerminan tema utama persepsi nasabah terhadap Bank XYZ.")

    sh("🔍 Verbatim Explorer")
    ve1,ve2,ve3,ve4=st.columns(4)
    with ve1: vnf=st.multiselect("Kategori NPS:",key="ms_vnf9",options=['Promoter','Passive','Detractor'],default=['Promoter','Passive','Detractor'])
    with ve2: vpf=st.multiselect("Provinsi:",key="ms_vpf9",options=sorted(df['PROV'].dropna().unique()))
    with ve3: vsr=st.text_input("🔎 Cari kata kunci:",key="ti_vsr9")
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
    st.info(f"Menampilkan **{len(vdfd):,}** responden")
    dc9={'CABANG':'Cabang','PROV':'Provinsi','S1':'Gender','S2_2':'Usia','S4':'Tenure',
         'G1A':'NPS','G1A_CAT':'Kategori','E1A':'Kepuasan','Alasan NPS':'Alasan NPS','Alasan Kepuasan':'Alasan Kepuasan'}
    ex9={k:v for k,v in dc9.items() if k in vdfd.columns}
    st.dataframe(vdfd[list(ex9.keys())].rename(columns=ex9).sort_values('NPS'),use_container_width=True,height=400,hide_index=True)

# ═══════════════════════════════════════════════════════
# TAB 10 — AI ANALYST (GEMINI)
# ═══════════════════════════════════════════════════════
with t10:
    sh("🤖 AI CX Analyst — Powered by Groq & Llama 3.1")
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0C2048,#0C1829);
    border:1px solid #163690;border-radius:14px;padding:16px 20px;margin-bottom:16px'>
    <div style='color:#60A5FA!important;font-size:13px;font-weight:800;margin-bottom:8px'>
    💡 Cara Menggunakan AI Analyst</div>
    <div style='color:#6A8AAA!important;font-size:12px;line-height:1.8'>
    AI menjawab berdasarkan data aktual sesuai filter aktif saat ini.<br>
    Contoh: <i>"Apa kelemahan utama XYZ?"</i> · <i>"Cabang mana yang paling perlu diperhatikan?"</i> · 
    <i>"Mengapa nasabah jadi Detractor?"</i>
    </div></div>""", unsafe_allow_html=True)

    ctx=build_ctx(df,df_hk)
    if "chat_history" not in st.session_state: st.session_state.chat_history=[]

    st.markdown("<div style='color:#2D4060!important;font-size:11px;font-weight:700;"
                "text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px'>"
                "⚡ Pertanyaan Cepat</div>",unsafe_allow_html=True)
    qq=["Apa kelemahan utama XYZ vs kompetitor?","Cabang mana yang perlu paling diperhatikan?",
        "Faktor apa yang paling mendorong Promoter?","Mengapa nasabah menjadi Detractor?"]
    qc=st.columns(4)
    for i,(cq,qt) in enumerate(zip(qc,qq)):
        if cq.button(qt,key=f"qq_{i}"):
            st.session_state.chat_history.append({"role":"user","content":qt})
            with st.spinner("🤖 AI sedang menganalisis data..."):
                rep=call_ai(st.session_state.chat_history,ctx)
            st.session_state.chat_history.append({"role":"assistant","content":rep})
            st.rerun()

    st.markdown("<br>",unsafe_allow_html=True)

    if st.session_state.chat_history:
        st.markdown("<div class='chat-wrap'>",unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg['role']=='user':
                st.markdown(f"<div class='chat-user'><p>👤 {msg['content']}</p></div><div class='chat-cf'></div>",unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'><p>🤖 {msg['content']}</p></div><div class='chat-cf'></div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

    ai1,ai2=st.columns([4,1])
    with ai1:
        user_in=st.text_input("Tanyakan sesuatu tentang data:",
            placeholder="Contoh: Bagaimana perbandingan emosi nasabah XYZ vs kompetitor?",
            key="ti_ai10",label_visibility="collapsed")
    with ai2:
        send=st.button("Kirim 🚀",key="btn_ai10",use_container_width=True)

    if send and user_in.strip():
        st.session_state.chat_history.append({"role":"user","content":user_in})
        with st.spinner("🤖 AI sedang menganalisis..."):
            rep=call_ai(st.session_state.chat_history,ctx)
        st.session_state.chat_history.append({"role":"assistant","content":rep})
        st.rerun()

    if st.session_state.chat_history:
        _,c_rst=st.columns([4,1])
        with c_rst:
            if st.button("🗑️ Reset Chat",key="btn_rst10",use_container_width=True):
                st.session_state.chat_history=[]; st.rerun()

    with st.expander("📊 Lihat Konteks Data yang Dikirim ke AI"):
        st.code(ctx,language='text')

    with st.expander("⚙️ Setup API Key Gemini"):
        st.markdown("""
**Langkah-langkah:**
1. Dapatkan API key gratis di [Google AI Studio](https://aistudio.google.com/apikey)
2. Di Streamlit Cloud → **Settings** → **Secrets**
3. Tambahkan:
```toml
GEMINI_API_KEY = "AIzaSy..."
```
4. Klik **Save** → app restart otomatis
5. Pastikan `google-generativeai` ada di `requirements.txt`
        """)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#0F1E30!important;font-size:10px;letter-spacing:.5px'>"
    "🚀 Bank XYZ CX Intelligence v5.0 · Dark Edition · K-Means Clustering · "
    "Gemini AI · Streamlit & Plotly</p>",unsafe_allow_html=True)
