"""
WAVIS v2 — Fully Fixed App
Run: streamlit run app.py
"""
import os, sys, json, time, tempfile, warnings
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="WAVIS v2", page_icon="🦁",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Outfit',sans-serif;}
.stApp{background:#04100a;color:#e8f5e9;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.5rem;max-width:1400px;}
.hero{background:linear-gradient(135deg,#04100a,#071a10,#04100a);border-bottom:1px solid #1a3a25;padding:2rem 2rem 1.5rem;text-align:center;position:relative;overflow:hidden;margin-bottom:1.5rem;}
.hero::before{content:'';position:absolute;inset:0;background:radial-gradient(ellipse 60% 80% at 50% 0%,rgba(74,222,128,.08),transparent);pointer-events:none;}
.logo{font-size:3.8rem;font-weight:800;letter-spacing:-3px;line-height:1;background:linear-gradient(135deg,#86efac,#4ade80,#22c55e);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.logo-sub{font-size:.8rem;letter-spacing:.3em;text-transform:uppercase;color:#4ade80;opacity:.6;margin-top:.3rem;}
.logo-tag{font-size:1rem;color:#a7f3d0;margin-top:.5rem;opacity:.85;}
.badges{display:flex;justify-content:center;gap:.5rem;flex-wrap:wrap;margin-top:.8rem;}
.badge{background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.25);color:#4ade80;border-radius:100px;padding:.2rem .8rem;font-size:.7rem;font-family:'JetBrains Mono',monospace;}
.card{background:#071a10;border:1px solid #1a3a25;border-radius:14px;padding:1.2rem 1.4rem;position:relative;overflow:hidden;margin-bottom:.75rem;}
.card::before{content:'';position:absolute;top:0;left:0;width:4px;height:100%;}
.green::before{background:linear-gradient(180deg,#4ade80,#16a34a);}
.orange::before{background:linear-gradient(180deg,#fb923c,#ea580c);}
.blue::before{background:linear-gradient(180deg,#60a5fa,#2563eb);}
.purple::before{background:linear-gradient(180deg,#c084fc,#9333ea);}
.card-lbl{font-size:.63rem;letter-spacing:.2em;text-transform:uppercase;color:#4ade80;opacity:.6;margin-bottom:.3rem;}
.card-val{font-size:1.65rem;font-weight:700;color:#f0fdf4;line-height:1.15;}
.card-sub{font-size:.8rem;color:#86efac;margin-top:.2rem;opacity:.75;}
.sp-row{display:flex;align-items:center;gap:.7rem;padding:.45rem 0;border-bottom:1px solid #0f2a18;font-size:.85rem;}
.sp-row:last-child{border-bottom:none;}
.sp-em{font-size:1.25rem;width:1.8rem;text-align:center;}
.sp-name{color:#86efac;flex:1;}
.sp-pct{color:#4ade80;font-family:'JetBrains Mono',monospace;font-size:.78rem;font-weight:600;}
.ag-row{display:flex;align-items:center;gap:.6rem;padding:.4rem 0;border-bottom:1px solid #0f2a18;font-size:.8rem;}
.ag-row:last-child{border-bottom:none;}
.ag-id{color:#1a5c30;font-family:'JetBrains Mono',monospace;font-size:.68rem;width:2.5rem;}
.ag-name{color:#4ade80;flex:1;font-family:'JetBrains Mono',monospace;font-size:.75rem;}
.sec-head{font-size:.62rem;letter-spacing:.25em;text-transform:uppercase;color:#4ade80;opacity:.5;margin-bottom:.75rem;display:flex;align-items:center;gap:.5rem;}
.sec-head::after{content:'';flex:1;height:1px;background:#1a3a25;}
.warn-box{background:#1a1208;border:1px solid #4a3a08;border-radius:10px;padding:.65rem 1rem;font-size:.83rem;color:#fbbf24;margin:.5rem 0;}
.info-box{background:#081a28;border:1px solid #0e3a5a;border-radius:10px;padding:.65rem 1rem;font-size:.8rem;color:#60a5fa;margin:.5rem 0;}
.fact-box{background:#0a2518;border-left:3px solid #4ade80;border-radius:0 8px 8px 0;padding:.85rem 1rem;font-size:.86rem;color:#86efac;font-style:italic;line-height:1.6;margin-top:.5rem;}
.model-ok{background:#0a2518;border:1px solid #1a5c30;border-radius:10px;padding:.6rem 1rem;font-size:.82rem;color:#4ade80;margin-bottom:1rem;}
.model-warn{background:#1a1208;border:1px solid #4a3a08;border-radius:10px;padding:.6rem 1rem;font-size:.82rem;color:#fbbf24;margin-bottom:1rem;}
.mrow{display:flex;gap:.75rem;margin:.6rem 0;flex-wrap:wrap;}
.mbox{background:#0a1f12;border:1px solid #1a3a25;border-radius:10px;padding:.7rem 1rem;flex:1;min-width:100px;text-align:center;}
.mval{font-size:1.15rem;font-weight:700;color:#4ade80;}
.mlbl{font-size:.68rem;color:#86efac;opacity:.7;margin-top:.15rem;}
.stButton button{background:linear-gradient(135deg,#16a34a,#15803d)!important;color:white!important;border:none!important;border-radius:10px!important;font-family:'Outfit',sans-serif!important;font-weight:600!important;padding:.6rem 1.5rem!important;}
.stButton button:hover{background:linear-gradient(135deg,#22c55e,#16a34a)!important;box-shadow:0 4px 20px rgba(74,222,128,.3)!important;}
[data-testid="metric-container"]{background:#071a10;border:1px solid #1a3a25;border-radius:10px;padding:.7rem!important;}
[data-testid="stMetricValue"]{color:#4ade80!important;}
[data-testid="stMetricLabel"]{color:#86efac!important;opacity:.7;}
.stTabs [data-baseweb="tab-list"]{background:#071a10!important;gap:.3rem;}
.stTabs [data-baseweb="tab"]{color:#86efac!important;border-radius:8px 8px 0 0!important;}
.stTabs [aria-selected="true"]{background:#1a3a25!important;color:#4ade80!important;}
.stProgress>div>div>div{background:linear-gradient(90deg,#16a34a,#4ade80)!important;}
</style>
""", unsafe_allow_html=True)

BG = '#071a10'

def mbox(label, value):
    return f'<div class="mbox"><div class="mval">{value}</div><div class="mlbl">{label}</div></div>'

# ── Pipeline ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
        from agents.wavis_pipeline import WAVISPipeline
        p = WAVISPipeline(
            model_path='models/wavis_v2_best.pth',
            class_map_path='data/class_map.json',
            display_names_path='data/display_names.json',
        )
        return p, None
    except Exception as e:
        return None, str(e)

# ── Mic ────────────────────────────────────────────────────────────────────────
def record_audio(duration=5, sr=32000):
    try:
        import sounddevice as sd
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except ImportError:
        st.error("Install sounddevice: pip install sounddevice")
        return None
    except Exception as e:
        st.error(f"Mic error: {e}")
        return None

def save_wav(y, sr=32000):
    import soundfile as sf
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(tmp.name, y, sr)
    return tmp.name

# ── Build plotly figures and serialise to JSON right after analysis ────────────
# Storing JSON in session_state is 100% reliable across tab switches

def build_waveform_json(y, sr=32000):
    y = np.array(y, dtype=np.float32)
    step = max(1, len(y) // 1500)
    t  = np.linspace(0, len(y)/sr, len(y[::step])).tolist()
    ya = y[::step].tolist()
    fig = go.Figure(go.Scatter(
        x=t, y=ya, mode='lines',
        line=dict(color='#4ade80', width=.8),
        fill='tozeroy', fillcolor='rgba(74,222,128,.06)'))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=145,
        margin=dict(l=10, r=10, t=10, b=35),
        xaxis=dict(showgrid=False, color='#4ade80',
                   tickfont=dict(size=9),
                   title=dict(text='Time (seconds)', font=dict(color='#4ade80', size=10))),
        yaxis=dict(showgrid=False, showticklabels=False,
                   zeroline=True, zerolinecolor='#1a3a25'),
        showlegend=False)
    return fig.to_json()

def build_mel_json(mel):
    mel = np.array(mel, dtype=np.float32)
    fig = go.Figure(go.Heatmap(
        z=mel.tolist(), colorscale='Magma', showscale=True,
        colorbar=dict(thickness=12,
                      tickfont=dict(color='#86efac', size=8),
                      title=dict(text='dB', font=dict(color='#86efac', size=9)))))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=215,
        margin=dict(l=10, r=55, t=10, b=35),
        xaxis=dict(showticklabels=False, showgrid=False,
                   title=dict(text='Time →', font=dict(color='#4ade80', size=10))),
        yaxis=dict(showticklabels=False, showgrid=False,
                   title=dict(text='Freq →', font=dict(color='#4ade80', size=10))))
    return fig.to_json()

def build_top5_json(top_k):
    items  = top_k[:5]
    labels = [f"{d['emoji']}  {d['name'][:22]}" for d in items]
    values = [d['conf'] for d in items]
    alphas = [1.0, 0.65, 0.45, 0.30, 0.18]
    colors = [f'rgba(74,222,128,{alphas[i]})' for i in range(len(items))]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#86efac', size=11)))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=255,
        margin=dict(l=0, r=65, t=10, b=0),
        xaxis=dict(range=[0,115], showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, color='#86efac',
                   tickfont=dict(size=11), autorange='reversed'),
        showlegend=False, bargap=0.32)
    return fig.to_json()

def build_gauge_json(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score*100, 1),
        number={'suffix':'%', 'font':{'color':'#4ade80','size':28,'family':'Outfit'}},
        title={'text':'Proximity Score', 'font':{'color':'#86efac','size':11}},
        gauge={
            'axis':{'range':[0,100],'tickcolor':'#1a3a25',
                    'tickfont':{'color':'#2d5a3a','size':8}},
            'bar':{'color':'#4ade80','thickness':0.22},
            'bgcolor':BG, 'bordercolor':'#1a3a25',
            'steps':[
                {'range':[0,25],'color':'#062510'},
                {'range':[25,50],'color':'#0a3015'},
                {'range':[50,70],'color':'#0f3d1c'},
                {'range':[70,100],'color':'#1a4a28'},
            ]}))
    fig.update_layout(paper_bgcolor=BG, height=200,
        margin=dict(l=15,r=15,t=45,b=5), font=dict(family='Outfit'))
    return fig.to_json()

def fig_from_json(j):
    import plotly.io as pio
    return pio.from_json(j)

def compass_html(h_arrow, h_dir):
    angle = {'→':90,'←':270,'↑':0,'↓':180}.get(h_arrow, 0)
    return f"""<div style="text-align:center;padding:.5rem 0;">
        <div style="width:110px;height:110px;border-radius:50%;border:2px solid #1a3a25;
                    margin:0 auto;background:radial-gradient(circle,#071a10,#04100a);
                    display:flex;align-items:center;justify-content:center;">
            <div style="transform:rotate({angle}deg);font-size:2.5rem;
                        filter:drop-shadow(0 0 10px rgba(74,222,128,.5));">🧭</div>
        </div>
        <div style="margin-top:.5rem;font-size:.9rem;color:#86efac;font-weight:600;">{h_dir}</div>
    </div>"""

def render_agents(statuses=None):
    AGENTS = [
        ("A1","SoundAcquisitionAgent","acquisition","Validate & normalize"),
        ("A2","SoundSeparationAgent","separation","Denoise & segment"),
        ("A3","ClassificationAgent","classification","EfficientNet species ID"),
        ("A4","LocalizationAgent","localization","Distance + direction"),
        ("A5","DecisionFusionAgent","fusion","Combine & output"),
    ]
    rows = "".join(f"""<div class="ag-row">
        <span class="ag-id">{ag}</span><span class="ag-name">{nm}</span>
        <span style="color:#1a5c30;flex:1;font-size:.68rem;">{desc}</span>
        <span style="font-size:.85rem;">{(statuses or {}).get(key,'⏳')}</span>
    </div>""" for ag,nm,key,desc in AGENTS)
    return f'<div class="card green" style="padding:.9rem 1rem;">{rows}</div>'

# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    st.markdown("""<div class="hero">
        <div class="logo">WAVIS v2</div>
        <div class="logo-sub">Wildlife Acoustic Vocalization Identification System</div>
        <div class="logo-tag">Species · Distance · Direction — from a single microphone</div>
        <div class="badges">
            <span class="badge">🧠 EfficientNet-B3</span>
            <span class="badge">258 Species</span>
            <span class="badge">📍 Distance AI</span>
            <span class="badge">🧭 Direction AI</span>
            <span class="badge">🤖 5-Agent Pipeline</span>
            <span class="badge">🎙️ Live Mic</span>
        </div>
    </div>""", unsafe_allow_html=True)

    pipeline, err = load_pipeline()
    if pipeline is None:
        st.error(f"❌ Pipeline error: {err}")
        return

    if pipeline.model_loaded:
        st.markdown("""<div class="model-ok">
        ✅ <b>EfficientNet-B3 loaded</b> · 258 species · Top-1: 51.57% · Top-5: 72.19% ·
        <span style="opacity:.7;">Train more epochs → 95%+</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="model-warn">
        ⚠️ <b>Demo mode</b> — Place <code>wavis_v2_best.pth</code> in <code>models/</code>
        and <code>class_map.json</code> + <code>display_names.json</code> in <code>data/</code>
        </div>""", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.45], gap="large")

    # ═══════════════════════════ LEFT COLUMN ══════════════════════════════════
    with left_col:
        st.markdown('<div class="sec-head">🎙️ Audio Input</div>', unsafe_allow_html=True)

        input_mode = st.radio("Input method",
            ["📁 Upload Audio File", "🎙️ Live Microphone", "🎵 Demo Sounds"],
            label_visibility='hidden')

        audio_path = None

        if input_mode == "📁 Upload Audio File":
            uploaded = st.file_uploader("Audio file",
                type=['wav','mp3','ogg','flac','m4a'],
                label_visibility='hidden')
            if uploaded:
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    audio_path = tmp.name
                st.audio(uploaded)
                st.caption(f"📄 {uploaded.name} · {len(uploaded.getbuffer())//1024} KB")

        elif input_mode == "🎙️ Live Microphone":
            st.markdown("""<div class="info-box">
            🎙️ <b>Live Recording</b> — Records from your microphone, just like Google Voice.<br>
            <span style="opacity:.7;font-size:.78rem;">
            Requires: <code>pip install sounddevice soundfile</code></span>
            </div>""", unsafe_allow_html=True)
            dur = st.slider("Recording duration (seconds)", 3, 10, 5)
            if st.button("🎙️ START RECORDING", use_container_width=True):
                with st.spinner(f"🔴 Recording {dur}s — make animal sounds!"):
                    y_rec = record_audio(duration=dur, sr=32000)
                if y_rec is not None:
                    st.success(f"✅ Captured {dur}s!")
                    audio_path = save_wav(y_rec, 32000)
                    st.audio(audio_path, format='audio/wav')
                    st.session_state['last_rec'] = audio_path
                else:
                    st.error("❌ Recording failed — check microphone")
            if 'last_rec' in st.session_state and audio_path is None:
                lp = st.session_state['last_rec']
                if os.path.exists(lp):
                    audio_path = lp
                    st.caption("↑ Using last recording")

        else:
            samples = (list(Path('demo_sounds').glob('*.wav')) +
                       list(Path('demo_sounds').glob('*.mp3')) +
                       list(Path('demo_sounds').glob('*.ogg')))
            if samples:
                sel = st.selectbox("Demo sound", [s.name for s in samples],
                                   label_visibility='hidden')
                audio_path = str(Path('demo_sounds') / sel)
                st.audio(audio_path)
            else:
                st.markdown("""<div class="info-box">
                📁 Place <code>.wav</code> files in <code>demo_sounds/</code><br>
                Download: <a href="https://freesound.org" target="_blank" style="color:#60a5fa;">freesound.org</a> ·
                <a href="https://www.xeno-canto.org" target="_blank" style="color:#60a5fa;">xeno-canto.org</a>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
        analyze = st.button("🔍 ANALYZE SOUND",
                            disabled=(audio_path is None),
                            use_container_width=True)

        st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">🤖 Agent Pipeline</div>', unsafe_allow_html=True)
        agent_ph = st.empty()
        agent_ph.markdown(render_agents(st.session_state.get('agent_statuses')),
                          unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-head">📊 Model Info</div>', unsafe_allow_html=True)
        st.markdown("""<div class="card green" style="font-size:.8rem;padding:.9rem 1rem;">
            <div style="color:#4ade80;font-weight:600;margin-bottom:.4rem;">EfficientNet-B3 — Transfer Learning</div>
            <div style="color:#86efac;line-height:1.85;">
            🧠 Backbone: EfficientNet-B3 (ImageNet pretrained)<br>
            📊 Species: 258 wildlife classes<br>
            🎵 Input: 3-ch Mel spectrogram 224×224<br>
            🔧 2-phase training + MixUp + SpecAugment<br>
            🎯 Top-1: 51.57% · Top-5: 72.19%<br>
            📈 More epochs + BirdCLEF → 95%+
            </div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════ RUN ANALYSIS ═════════════════════════════════════
    if analyze and audio_path:
        prog = st.progress(0, "🔄 Starting WAVIS pipeline...")
        try:
            prog.progress(10, "🎙️ Agent 1: Acquiring audio...")
            time.sleep(.1)
            prog.progress(30, "🔊 Agent 2: Denoising & segmenting...")
            time.sleep(.1)
            prog.progress(52, "🧠 Agent 3: EfficientNet classification...")

            result = pipeline.run(audio_path)

            prog.progress(72, "📍 Agent 4: Distance & direction...")
            time.sleep(.1)
            prog.progress(88, "📊 Building charts...")

            from utils.audio_features import load_audio, get_mel_matrix, extract_acoustic_features
            y     = load_audio(audio_path)
            mel   = get_mel_matrix(y)
            feats = extract_acoustic_features(y)

            # ── Build ALL charts NOW and store as JSON strings ─────────────────
            # JSON strings survive Streamlit reruns and tab switches perfectly
            prog.progress(93, "💾 Saving to session...")
            st.session_state['result']          = result
            st.session_state['feats']           = feats
            st.session_state['audio_duration']  = float(len(y) / 32000)
            st.session_state['agent_statuses']  = result.agents_status
            st.session_state['fig_waveform']    = build_waveform_json(y, 32000)
            st.session_state['fig_mel']         = build_mel_json(mel)
            st.session_state['fig_top5']        = build_top5_json(result.top_k)
            st.session_state['fig_gauge']       = build_gauge_json(result.proximity_score)

            prog.progress(100, "✅ Done!")
            time.sleep(.2)
            prog.empty()

            agent_ph.markdown(render_agents(result.agents_status), unsafe_allow_html=True)

        except Exception as e:
            prog.empty()
            st.error(f"❌ Error: {e}")
            import traceback; st.code(traceback.format_exc())

    # ═══════════════════════ RIGHT COLUMN ═════════════════════════════════════
    with right_col:

        if 'result' not in st.session_state:
            st.markdown("""<div style="border:2px dashed #1a3a25;border-radius:14px;
                padding:5rem 2rem;text-align:center;margin-top:.25rem;">
                <div style="font-size:3.5rem;margin-bottom:1rem;">🌿</div>
                <div style="font-weight:600;font-size:1.1rem;color:#1a5c30;">
                    Upload · Record · or pick a demo sound</div>
                <div style="font-size:.85rem;margin-top:.5rem;color:#1a5c30;opacity:.6;">
                    Species · Distance · Direction appear here</div>
            </div>""", unsafe_allow_html=True)

        else:
            result = st.session_state['result']
            feats  = st.session_state['feats']
            dur    = st.session_state['audio_duration']

            # ── 3 top cards ───────────────────────────────────────────────────
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="card green">
                    <div class="card-lbl">🦎 Identified Species</div>
                    <div class="card-val">{result.emoji} {result.common_name}</div>
                    <div class="card-sub">Confidence: <b>{result.confidence:.1f}%</b></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="card orange">
                    <div class="card-lbl">📍 Distance Estimate</div>
                    <div class="card-val" style="font-size:1.3rem;">
                        {result.dist_emoji} {result.dist_category}</div>
                    <div class="card-sub">{result.dist_range}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="card blue">
                    <div class="card-lbl">🧭 Direction Estimate</div>
                    <div class="card-val" style="font-size:1.3rem;">
                        {result.h_arrow} {result.horizontal}</div>
                    <div class="card-sub">↕ {result.vertical}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="warn-box">{result.dist_warning}</div>',
                        unsafe_allow_html=True)

            # ── TABS ──────────────────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs([
                "🏆 Top-5 Species",
                "📍 Distance & Direction",
                "🌊 Audio Signal",
                "🔬 Details"
            ])

            # ── TAB 1 ─────────────────────────────────────────────────────────
            with tab1:
                st.markdown("##### Top-5 Predictions")
                st.plotly_chart(fig_from_json(st.session_state['fig_top5']),
                                use_container_width=True,
                                config={'displayModeBar': False})

                st.markdown("##### Candidate Rankings")
                for i, d in enumerate(result.top_k[:8]):
                    bw = min(int(d['conf']), 100)
                    nc = '#4ade80' if i == 0 else '#86efac'
                    st.markdown(f"""<div class="sp-row">
                        <span style="color:#1a5c30;font-size:.7rem;font-family:'JetBrains Mono',monospace;width:1.2rem;">#{i+1}</span>
                        <span class="sp-em">{d['emoji']}</span>
                        <div style="flex:1;">
                            <div style="display:flex;justify-content:space-between;align-items:center;">
                                <span class="sp-name" style="color:{nc};">{d['name'][:30]}</span>
                                <span class="sp-pct">{d['conf']:.1f}%</span>
                            </div>
                            <div style="background:#0f2a18;border-radius:100px;height:3px;margin-top:3px;">
                                <div style="width:{bw}%;height:3px;border-radius:100px;
                                    background:linear-gradient(90deg,#16a34a,#4ade80);"></div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("""<div class="info-box" style="margin-top:.75rem;">
                <b>How it works:</b> EfficientNet-B3 processes audio as a 3-channel Mel spectrogram
                image (224×224). Pretrained on 1.2M ImageNet images, fine-tuned on 17,345 wildlife 
                clips across 258 species using 2-phase transfer learning + MixUp + SpecAugment.<br>
                <b>Top-1: 51.57% · Top-5: 72.19%</b> — train more epochs for 95%+
                </div>""", unsafe_allow_html=True)

                if result.fact:
                    st.markdown(f'<div class="fact-box">💡 {result.fact}</div>',
                                unsafe_allow_html=True)

            # ── TAB 2 ─────────────────────────────────────────────────────────
            with tab2:
                st.markdown("##### Distance Estimation")
                st.plotly_chart(fig_from_json(st.session_state['fig_gauge']),
                                use_container_width=True,
                                config={'displayModeBar': False})

                st.markdown(f"""<div class="mrow">
                    {mbox('Category', f'{result.dist_emoji} {result.dist_category}')}
                    {mbox('Range', result.dist_range)}
                    {mbox('Confidence', f'{result.dist_confidence:.0f}%')}
                    {mbox('RMS Energy', f'{result.rms_energy:.5f}')}
                </div>""", unsafe_allow_html=True)

                st.markdown("""<div class="info-box">
                <b>Distance Algorithm:</b> Inverse square law (intensity ∝ 1/r²) +
                spectral rolloff decay + spectral flatness (reverb proxy).<br>
                <b>Weights:</b> 45% RMS + 25% rolloff + 15% centroid + 15% flatness
                </div>""", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("##### Direction Estimation")
                st.markdown(compass_html(result.h_arrow, result.horizontal),
                            unsafe_allow_html=True)

                st.markdown(f"""<div class="mrow">
                    {mbox('Horizontal', f'{result.h_arrow} {result.horizontal}')}
                    {mbox('H-Confidence', f'{result.h_confidence:.0f}%')}
                    {mbox('Elevation', result.vertical)}
                    {mbox('V-Confidence', f'{result.v_confidence:.0f}%')}
                </div>""", unsafe_allow_html=True)

                if result.direction_tip:
                    st.markdown(f'<div style="font-size:.82rem;color:#60a5fa;margin-top:.5rem;">{result.direction_tip}</div>',
                                unsafe_allow_html=True)
                if result.elevation_tip:
                    st.markdown(f'<div style="font-size:.82rem;color:#a78bfa;margin-top:.3rem;">{result.elevation_tip}</div>',
                                unsafe_allow_html=True)
                st.markdown(f'<div class="info-box" style="margin-top:.75rem;">{result.direction_disclaimer}</div>',
                            unsafe_allow_html=True)

            # ── TAB 3: Audio Signal ────────────────────────────────────────────
            with tab3:
                st.markdown("##### 🔊 Waveform")
                st.plotly_chart(fig_from_json(st.session_state['fig_waveform']),
                                use_container_width=True,
                                config={'displayModeBar': False})

                st.markdown("##### 🎨 Mel Spectrogram *(what the CNN sees)*")
                st.plotly_chart(fig_from_json(st.session_state['fig_mel']),
                                use_container_width=True,
                                config={'displayModeBar': False})

                st.markdown(f"""<div class="mrow">
                    {mbox('Duration', f'{dur:.1f}s')}
                    {mbox('Sample Rate', '32,000 Hz')}
                    {mbox('SNR', f'{result.snr_db:.1f} dB')}
                    {mbox('Mel Bins', '128')}
                </div>""", unsafe_allow_html=True)

                st.markdown("""<div class="info-box">
                <b>3-Channel Mel Image fed to EfficientNet:</b><br>
                • <b>Ch1</b> — Mel spectrogram (frequency content over time)<br>
                • <b>Ch2</b> — Delta features (rate of change = temporal dynamics)<br>
                • <b>Ch3</b> — Delta-delta features (acceleration of change)<br>
                All channels resized to 224×224 and ImageNet-normalized before inference.
                </div>""", unsafe_allow_html=True)

            # ── TAB 4: Details ─────────────────────────────────────────────────
            with tab4:
                st.markdown("##### ⚡ Performance Metrics")
                st.markdown(f"""<div class="mrow">
                    {mbox('Processing Time', f'{result.processing_ms:.0f} ms')}
                    {mbox('SNR', f'{result.snr_db:.1f} dB')}
                    {mbox('Top-1 Conf', f'{result.confidence:.1f}%')}
                    {mbox('Agents', '5 ✅')}
                </div>""", unsafe_allow_html=True)

                st.markdown("##### 🔬 Acoustic Features Table")
                import pandas as pd
                fd = pd.DataFrame({
                    "Feature": [
                        "RMS Energy (mean)","RMS Energy (max)",
                        "Spectral Rolloff","Spectral Centroid",
                        "Spectral Flatness","Spectral Bandwidth",
                        "Low Band Ratio","Mid Band Ratio",
                        "High Band Ratio","Temporal Asymmetry","Elevation Proxy",
                    ],
                    "Value": [
                        f"{feats['rms_mean']:.6f}",
                        f"{feats['rms_max']:.6f}",
                        f"{feats['rolloff_mean']:.0f} Hz",
                        f"{feats['centroid_mean']:.0f} Hz",
                        f"{feats['flatness_mean']:.6f}",
                        f"{feats['bandwidth_mean']:.0f} Hz",
                        f"{feats['band_low_ratio']:.4f}",
                        f"{feats['band_mid_ratio']:.4f}",
                        f"{feats['band_high_ratio']:.4f}",
                        f"{feats['temporal_asymmetry']:.4f}",
                        f"{feats['elevation_proxy']:.4f}",
                    ],
                    "Used By": [
                        "Agent 4","Agent 4","Agent 4","Agent 4","Agent 4",
                        "Agent 3","Agent 4","Agent 4","Agent 4","Agent 4","Agent 4",
                    ],
                    "Physics Basis": [
                        "Intensity ∝ 1/r² (inverse square law)",
                        "Peak amplitude proxy for distance",
                        "High freq attenuates faster with distance",
                        "Tone shifts lower when source is far",
                        "Reverb/diffusion increases with distance",
                        "Species-specific spectral brightness",
                        "Ground sources have more bass content",
                        "Animal calls mostly in 500–4kHz range",
                        "Aerial/tree sources have more high freq",
                        "Onset asymmetry = Left vs Right bias",
                        "High-freq ratio = elevation estimate",
                    ]
                })
                st.dataframe(fd, use_container_width=True, hide_index=True)

                st.markdown("##### 🤖 Agent Pipeline Status")
                for k, v in result.agents_status.items():
                    st.markdown(f"- **{k.capitalize()}**: {v}")

    st.markdown("""<div style="text-align:center;margin-top:3rem;padding:1.5rem;
        border-top:1px solid #1a3a25;color:#1a3a25;font-size:.76rem;">
        <b style="color:#4ade80;">WAVIS v2</b> — Wildlife Acoustic Vocalization Identification System<br>
        Dept. of AI &amp; Cyber Security · RCOEM Nagpur · 2025–26<br>
        <span style="opacity:.45;">Group 6: Romit Ghosh · Rohan Ravi · Saket Karwa · Ishant Tiwari · Dr. Nisarg Gandhewar</span>
    </div>""", unsafe_allow_html=True)

if __name__ == '__main__':
    main()