import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import requests, time
from streamlit_lottie import st_lottie
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

st.set_page_config(
    page_title="DYNAMO | Motor Intelligence Platform",
    layout="wide", page_icon="🔴",
    initial_sidebar_state="expanded"
)

def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_gear    = load_lottie("https://assets9.lottiefiles.com/packages/lf20_ystsffqy.json")
lottie_warning = load_lottie("https://assets5.lottiefiles.com/packages/lf20_urgn8b4i.json")
lottie_success = load_lottie("https://assets1.lottiefiles.com/packages/lf20_jbrw3hcz.json")
lottie_scan    = load_lottie("https://assets3.lottiefiles.com/packages/lf20_fcfjwiyb.json")

CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,4,4,0.95)",
    font=dict(family="Courier Prime, monospace", color="#8a7070", size=10),
    margin=dict(l=4, r=4, t=46, b=4),
)

def ax(label="", col="#8a7070"):
    return dict(
        title=dict(text=label, font=dict(family="Courier Prime, monospace", size=9, color=col)),
        tickfont=dict(family="Courier Prime, monospace", size=9, color=col),
        gridcolor="rgba(220,38,38,0.06)",
        linecolor="rgba(220,38,38,0.12)",
        zerolinecolor="rgba(220,38,38,0.08)",
    )

# ══════════════════════════════════════════════════════════════════════
# MASTER CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Teko:wght@300;400;500;600;700&family=Courier+Prime:wght@400;700&display=swap');

:root {
  --black:   #080404;
  --iron0:   #0e0606;
  --iron1:   #160a0a;
  --iron2:   #1e0f0f;
  --iron3:   #281515;
  --iron4:   #341c1c;

  --red:     #dc2626;
  --red-hi:  #ef4444;
  --red-lo:  rgba(220,38,38,0.08);
  --red-md:  rgba(220,38,38,0.18);
  --ember:   #f97316;
  --emb-lo:  rgba(249,115,22,0.07);
  --chrome:  #d4c4c4;
  --chr-md:  #7a6060;
  --chr-lo:  #3a2828;
  --silver:  #f1e8e8;
  --gold:    #fbbf24;
  --gld-lo:  rgba(251,191,36,0.07);
  --cyan:    #06b6d4;
  --cyn-lo:  rgba(6,182,212,0.07);
  --green:   #22c55e;
  --grn-lo:  rgba(34,197,94,0.07);
  --grn-md:  rgba(34,197,94,0.15);

  --border:  rgba(220,38,38,0.14);
  --bhi:     rgba(220,38,38,0.32);
  --bdim:    rgba(220,38,38,0.06);

  --fh: 'Bebas Neue', sans-serif;
  --ft: 'Teko', sans-serif;
  --fd: 'Courier Prime', monospace;
}

/* ─ GLOBAL ─ */
.stApp {
  background: var(--black) !important;
  background-image:
    radial-gradient(ellipse 60% 30% at 0% 0%,   rgba(220,38,38,0.07)  0%, transparent 55%),
    radial-gradient(ellipse 40% 20% at 100% 100%, rgba(249,115,22,0.04) 0%, transparent 55%),
    radial-gradient(ellipse 30% 18% at 50% 50%,  rgba(220,38,38,0.02)  0%, transparent 60%) !important;
}
.block-container {
  background: transparent !important;
  padding: 0 1.8rem 5rem !important;
  max-width: 1580px !important;
  position: relative; z-index: 2;
  padding-top: 0.4rem !important;
}
header[data-testid="stHeader"] { background: transparent !important; }
::-webkit-scrollbar       { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--iron0); }
::-webkit-scrollbar-thumb { background: var(--red); border-radius: 2px; }

/* ══ ANIMATED BACKGROUND ══ */
.bg-scene {
  position: fixed; top:0; left:0; width:100vw; height:100vh;
  z-index: 0; pointer-events: none; overflow: hidden;
}
@keyframes rot_cw  { from{transform:rotate(0deg);}  to{transform:rotate(360deg);} }
@keyframes rot_ccw { from{transform:rotate(0deg);}  to{transform:rotate(-360deg);} }
@keyframes bg_pulse{ 0%,100%{opacity:0.08;} 50%{opacity:0.22;} }
@keyframes bg_sweep{ from{transform:translateX(-100vw);} to{transform:translateX(100vw);} }
@keyframes bg_vscn { from{transform:translateY(-100vh);} to{transform:translateY(100vh);} }
.bg-ring {
  position: absolute; border-radius: 50%; border: 1px solid rgba(220,38,38,0.09);
  animation: bg_pulse 5s ease-in-out infinite;
}
.bg-cog {
  position: absolute; border-radius: 50%;
  border-top: 2px dashed rgba(220,38,38,0.07);
  border-right: 2px dashed rgba(220,38,38,0.04);
  border-bottom: 2px dashed rgba(220,38,38,0.07);
  border-left: 2px dashed rgba(220,38,38,0.04);
}
.bg-grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(220,38,38,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(220,38,38,0.025) 1px, transparent 1px);
  background-size: 60px 60px;
}
.bg-sweep-h {
  position: absolute; top:0; height:100vh; width:1px;
  background: linear-gradient(180deg, transparent, rgba(220,38,38,0.18), transparent);
  animation: bg_sweep 16s linear infinite;
}
.bg-sweep-v {
  position: absolute; left:0; width:100vw; height:1px;
  background: linear-gradient(90deg, transparent, rgba(249,115,22,0.12), transparent);
  animation: bg_vscn 11s linear infinite 3s;
}
.bg-vignette {
  position: absolute; inset:0;
  background: radial-gradient(ellipse 80% 80% at 50% 50%, transparent 40%, rgba(8,4,4,0.7) 100%);
}

/* ══ TOPBAR ══ */
.topbar {
  background: linear-gradient(90deg, var(--iron0), var(--iron1), var(--iron0));
  border-bottom: 2px solid var(--red);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px; height: 66px; position: relative;
  overflow: hidden; flex-wrap: nowrap; gap: 14px;
  box-shadow: 0 0 40px rgba(220,38,38,0.18), 0 4px 30px rgba(0,0,0,0.6);
}
.topbar::before {
  content:''; position:absolute; inset:0;
  background: repeating-linear-gradient(90deg,
    transparent, transparent 79px,
    rgba(220,38,38,0.03) 79px, rgba(220,38,38,0.03) 80px);
  pointer-events:none;
}
.topbar::after {
  content:''; position:absolute; top:0; left:-80%; width:60%; height:100%;
  background: linear-gradient(90deg, transparent, rgba(220,38,38,0.06), transparent);
  animation: topShimmer 6s ease-in-out infinite 2s;
}
@keyframes topShimmer { 0%{left:-80%;} 100%{left:120%;} }

.tb-left { display:flex; align-items:center; gap:12px; flex-shrink:0; }
.tb-logo {
  width:48px; height:48px; border-radius:50%;
  background: conic-gradient(var(--red), var(--ember), var(--gold), var(--red));
  display:flex; align-items:center; justify-content:center;
  font-size:22px; flex-shrink:0;
  border: 2px solid rgba(220,38,38,0.5);
  box-shadow: 0 0 24px rgba(220,38,38,0.4), inset 0 0 12px rgba(0,0,0,0.5);
  animation: logoSpin 8s linear infinite;
}
@keyframes logoSpin { from{filter:hue-rotate(0deg);} to{filter:hue-rotate(360deg);} }
.tb-brand {
  font-family: var(--fh); font-size: 28px; letter-spacing: 8px;
  color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;
  text-shadow: 0 0 30px rgba(220,38,38,0.7), 0 0 60px rgba(220,38,38,0.3);
  line-height: 1;
}
.tb-sub {
  font-family: var(--fd); font-size: 8px; color: var(--chr-md);
  letter-spacing: 3px; text-transform: uppercase; margin-top: 3px; display:block;
}
.tb-ticker {
  font-family: var(--fd); font-size: 9px; color: rgba(220,38,38,0.5);
  overflow: hidden; flex: 1; min-width: 0; letter-spacing: 1px;
}
.tick-inner { display:inline-block; white-space:nowrap; animation: tickroll 26s linear infinite; }
@keyframes tickroll { from{transform:translateX(100%);} to{transform:translateX(-100%);} }
.tb-leds { display:flex; gap:16px; flex-shrink:0; }
.led-u {
  display:flex; align-items:center; gap:5px;
  font-family:var(--fd); font-size:8px; color:var(--chr-lo); letter-spacing:1.5px;
}
.ld { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.ld-r{ background:var(--red);   box-shadow:0 0 8px var(--red);   animation:blink 1.4s ease-in-out infinite; }
.ld-e{ background:var(--ember); box-shadow:0 0 8px var(--ember); animation:blink 2s ease-in-out infinite .4s; }
.ld-g{ background:var(--green); box-shadow:0 0 8px var(--green); animation:blink 2.8s ease-in-out infinite .8s; }
.ld-c{ background:var(--cyan);  box-shadow:0 0 8px var(--cyan);  animation:blink 3.5s ease-in-out infinite 1.2s; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.08;} }

/* ─ DATA STRIP ─ */
.dstrip {
  display:flex; overflow-x:auto; background:var(--iron0);
  border-bottom:1px solid var(--border);
}
.dstrip::-webkit-scrollbar{ display:none; }
.di {
  padding:7px 14px; border-right:1px solid var(--border);
  font-family:var(--fd); font-size:9px; color:var(--chr-md); white-space:nowrap; flex-shrink:0;
}
.di:last-child{ border-right:none; }
.dv  { color:var(--red-hi); }
.dv.w{ color:var(--ember); }
.dv.c{ color:var(--gold); animation:blink .9s ease-in-out infinite; }

/* ══ CRAZY MOTOR ANIMATION ══ */
/* 6 rings, 4 orbiting dots, spark effect, counter-rotating inner cross */
@keyframes cw1  { from{transform:rotate(0);}   to{transform:rotate(360deg);} }
@keyframes cw2  { from{transform:rotate(0);}   to{transform:rotate(360deg);} }
@keyframes cw3  { from{transform:rotate(0);}   to{transform:rotate(360deg);} }
@keyframes cw4  { from{transform:rotate(0);}   to{transform:rotate(360deg);} }
@keyframes cw5  { from{transform:rotate(0);}   to{transform:rotate(360deg);} }
@keyframes ccw1 { from{transform:rotate(0);}   to{transform:rotate(-360deg);} }
@keyframes ccw2 { from{transform:rotate(0);}   to{transform:rotate(-360deg);} }
@keyframes ccw3 { from{transform:rotate(0);}   to{transform:rotate(-360deg);} }

@keyframes orb_a{
  from{transform:rotate(0deg)   translateX(118px) rotate(0deg);}
  to  {transform:rotate(360deg) translateX(118px) rotate(-360deg);}
}
@keyframes orb_b{
  from{transform:rotate(72deg)  translateX(88px)  rotate(-72deg);}
  to  {transform:rotate(432deg) translateX(88px)  rotate(-432deg);}
}
@keyframes orb_c{
  from{transform:rotate(144deg) translateX(60px)  rotate(-144deg);}
  to  {transform:rotate(504deg) translateX(60px)  rotate(-504deg);}
}
@keyframes orb_d{
  from{transform:rotate(216deg) translateX(38px)  rotate(-216deg);}
  to  {transform:rotate(576deg) translateX(38px)  rotate(-576deg);}
}
@keyframes corePulse{
  0%,100%{
    box-shadow:0 0 18px var(--red),0 0 40px rgba(220,38,38,.4),inset 0 0 10px rgba(255,255,255,.1);
  }
  50%{
    box-shadow:0 0 36px var(--red),0 0 80px rgba(220,38,38,.6),0 0 120px rgba(220,38,38,.2),inset 0 0 20px rgba(255,255,255,.2);
  }
}
@keyframes spark{
  0%,80%,100%{opacity:0;}
  85%{opacity:1; transform:scale(1.4);}
  90%{opacity:0.4;}
}
@keyframes pistonUD{
  0%,100%{transform:translateY(0);}
  50%{transform:translateY(-26px);}
}
@keyframes pistonLR{
  0%,100%{transform:translateX(0);}
  50%{transform:translateX(-20px);}
}

.mtr-scene{
  width:280px; height:280px; position:relative; flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
  margin: 0 auto;
}

/* 6 concentric rings with different styles & speeds */
.mtr-r1{
  position:absolute; width:268px; height:268px; border-radius:50%;
  border-top:  3px solid var(--red);
  border-right: 2px dashed rgba(220,38,38,0.35);
  border-bottom:3px solid rgba(220,38,38,0.2);
  border-left:  2px dashed rgba(220,38,38,0.35);
  animation: cw1 7s linear infinite;
  box-shadow: 0 0 28px rgba(220,38,38,0.18);
}
.mtr-r2{
  position:absolute; width:232px; height:232px; border-radius:50%;
  border-top:  2px solid var(--ember);
  border-right: 2px solid rgba(249,115,22,0.22);
  border-bottom:2px solid var(--ember);
  border-left:  2px solid rgba(249,115,22,0.22);
  animation: ccw1 4.5s linear infinite;
  box-shadow: 0 0 20px rgba(249,115,22,0.14);
}
.mtr-r3{
  position:absolute; width:194px; height:194px; border-radius:50%;
  border: 3px dashed rgba(251,191,36,0.45);
  animation: cw2 3s linear infinite;
  box-shadow: 0 0 16px rgba(251,191,36,0.12);
}
.mtr-r4{
  position:absolute; width:156px; height:156px; border-radius:50%;
  border-top:  2px solid var(--chrome);
  border-bottom:2px solid var(--chrome);
  border-left:  1px solid transparent;
  border-right: 1px solid transparent;
  animation: ccw2 2.2s linear infinite;
  box-shadow: 0 0 14px rgba(212,196,196,0.12);
}
.mtr-r5{
  position:absolute; width:114px; height:114px; border-radius:50%;
  border: 2px dashed rgba(6,182,212,0.5);
  animation: cw3 1.6s linear infinite;
  box-shadow: 0 0 12px rgba(6,182,212,0.15);
}
.mtr-r6{
  position:absolute; width:78px; height:78px; border-radius:50%;
  border-top: 2px solid rgba(34,197,94,0.6);
  border-bottom: 2px solid rgba(34,197,94,0.3);
  border-left: 1px solid transparent;
  border-right:1px solid transparent;
  animation: ccw3 1s linear infinite;
  box-shadow: 0 0 10px rgba(34,197,94,0.18);
}
/* tick marks on r1 */
.mtr-r1::before{
  content:''; position:absolute; inset:5px; border-radius:50%;
  border: 1px dotted rgba(220,38,38,0.18);
}
.mtr-r2::after{
  content:''; position:absolute; inset:8px; border-radius:50%;
  border-top: 1px solid rgba(249,115,22,0.12);
  border-bottom:1px solid rgba(249,115,22,0.12);
}
/* Orbiting dots */
.mtr-d1,.mtr-d2,.mtr-d3,.mtr-d4{ position:absolute; border-radius:50%; z-index:6; }
.mtr-d1{
  width:13px; height:13px;
  background:var(--red);
  box-shadow:0 0 18px var(--red),0 0 36px rgba(220,38,38,.6);
  animation:orb_a 3.5s linear infinite;
}
.mtr-d2{
  width:10px; height:10px;
  background:var(--ember);
  box-shadow:0 0 14px var(--ember),0 0 28px rgba(249,115,22,.6);
  animation:orb_b 5s linear infinite;
}
.mtr-d3{
  width:8px; height:8px;
  background:var(--gold);
  box-shadow:0 0 12px var(--gold),0 0 24px rgba(251,191,36,.6);
  animation:orb_c 2.4s linear infinite;
}
.mtr-d4{
  width:6px; height:6px;
  background:var(--cyan);
  box-shadow:0 0 10px var(--cyan),0 0 20px rgba(6,182,212,.6);
  animation:orb_d 1.8s linear infinite;
}
/* Core */
.mtr-core{
  position:absolute; z-index:7;
  width:56px; height:56px; border-radius:50%;
  background:radial-gradient(circle, #fff5f5 0%, var(--red) 35%, #1a0505 75%);
  border:2.5px solid rgba(220,38,38,0.8);
  animation:corePulse 2.2s ease-in-out infinite;
  display:flex; align-items:center; justify-content:center;
}
.mtr-core::after{
  content:''; width:14px; height:14px; border-radius:50%;
  background:white; opacity:0.9;
}
/* Spark flash */
.mtr-spark{
  position:absolute; z-index:8;
  width:4px; height:4px; border-radius:50%;
  background:white; box-shadow:0 0 8px white;
  top:calc(50% - 2px); left:calc(50% + 26px);
  animation:spark 2.8s ease-in-out infinite;
}
.mtr-spark2{
  position:absolute; z-index:8;
  width:3px; height:3px; border-radius:50%;
  background:var(--gold); box-shadow:0 0 6px var(--gold);
  top:calc(50% - 54px); left:calc(50% - 1.5px);
  animation:spark 2.8s ease-in-out infinite 1.4s;
}

/* Piston arms */
.mtr-piston-v{
  position:absolute; z-index:2;
  width:6px; height:44px;
  background:linear-gradient(180deg,var(--iron4),rgba(220,38,38,0.3));
  border:1px solid rgba(220,38,38,0.18);
  border-radius:3px;
  top:0; left:calc(50% - 3px);
  animation:pistonUD 1.2s ease-in-out infinite;
  transform-origin:bottom center;
}
.mtr-piston-h{
  position:absolute; z-index:2;
  width:44px; height:6px;
  background:linear-gradient(90deg,var(--iron4),rgba(220,38,38,0.3));
  border:1px solid rgba(220,38,38,0.18);
  border-radius:3px;
  right:0; top:calc(50% - 3px);
  animation:pistonLR 1.8s ease-in-out infinite .6s;
  transform-origin:right center;
}

.mtr-label{
  font-family:var(--fh); font-size:14px; letter-spacing:6px;
  color:var(--red-hi); text-align:center; margin-top:10px;
  text-shadow:0 0 20px rgba(220,38,38,.7);
  text-transform:uppercase;
}
.mtr-sublabel{
  font-family:var(--fd); font-size:8px; letter-spacing:3px;
  color:rgba(220,38,38,0.3); text-align:center; margin-top:3px;
}

/* ══ SECTION LABELS ══ */
.slabel{
  display:flex; align-items:center; gap:12px;
  margin:30px 0 16px; padding-bottom:11px;
  border-bottom:1px solid var(--border);
}
.sid{
  font-family:var(--fh); font-size:10px; letter-spacing:3px; color:var(--iron0);
  background:var(--red); padding:5px 14px; border-radius:2px;
}
.stitle{
  font-family:var(--ft); font-size:16px; font-weight:600;
  color:var(--chrome); letter-spacing:3px; text-transform:uppercase;
}
.slive{ margin-left:auto; font-family:var(--fd); font-size:9px; color:var(--green); letter-spacing:2px; }
.swarn{ margin-left:auto; font-family:var(--fd); font-size:9px; color:var(--ember); letter-spacing:2px; }

/* ══ INSTRUMENT TILES ══ */
.igrid{ display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }
.itile{
  background:var(--iron1); border:1px solid var(--border); border-radius:4px;
  padding:16px 14px 13px; position:relative; overflow:hidden;
  transition:transform .22s,border-color .22s,box-shadow .22s;
}
.itile::before{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.t-r::before{ background:var(--red);   box-shadow:0 1px 14px rgba(220,38,38,.8); }
.t-e::before{ background:var(--ember); box-shadow:0 1px 14px rgba(249,115,22,.7); }
.t-g::before{ background:var(--green); box-shadow:0 1px 14px rgba(34,197,94,.7);  }
.t-c::before{ background:var(--cyan);  box-shadow:0 1px 14px rgba(6,182,212,.7);  }
.t-o::before{ background:var(--gold);  box-shadow:0 1px 14px rgba(251,191,36,.6); }
.itile:hover{ transform:translateY(-3px); border-color:var(--bhi); box-shadow:0 8px 28px rgba(0,0,0,.6); }
.il{ font-family:var(--fd); font-size:8px; letter-spacing:2.5px; color:var(--chr-md); text-transform:uppercase; margin-bottom:9px; }
.iv{ font-family:var(--fh); font-size:32px; line-height:1; letter-spacing:1px; }
.iv.r{ color:var(--red);   text-shadow:0 0 16px rgba(220,38,38,.6); }
.iv.e{ color:var(--ember); text-shadow:0 0 16px rgba(249,115,22,.6); }
.iv.g{ color:var(--green); text-shadow:0 0 16px rgba(34,197,94,.5);  }
.iv.c{ color:var(--cyan);  text-shadow:0 0 16px rgba(6,182,212,.5);  }
.iv.o{ color:var(--gold);  text-shadow:0 0 16px rgba(251,191,36,.5); }
.iu{ font-family:var(--fd); font-size:9px; color:var(--chr-lo); margin-top:5px; }
.ibar{ height:3px; background:var(--iron4); border-radius:99px; margin-top:10px; overflow:hidden; }
.ibf { height:100%; border-radius:99px; animation:barf 1.3s cubic-bezier(.22,1,.36,1) both .35s; transform-origin:left; }
.ibr{ background:linear-gradient(90deg,var(--red),rgba(220,38,38,0.3)); }
.ibe{ background:linear-gradient(90deg,var(--ember),rgba(249,115,22,0.3)); }
.ibg{ background:linear-gradient(90deg,var(--green),rgba(34,197,94,0.3)); }
.ibc{ background:linear-gradient(90deg,var(--cyan),rgba(6,182,212,0.3)); }
.ibo{ background:linear-gradient(90deg,var(--gold),rgba(251,191,36,0.3)); }
@keyframes tileUp{ from{opacity:0;transform:translateY(16px) scale(.97);} to{opacity:1;transform:translateY(0) scale(1);} }
@keyframes barf  { from{transform:scaleX(0);} to{transform:scaleX(1);} }

/* ══ METRIC TILES ══ */
.mgrid{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
.mtile{
  background:var(--iron2); border:1px solid var(--border); border-radius:4px;
  padding:16px 12px; text-align:center; position:relative; overflow:hidden;
  transition:all .22s; animation:tileUp .5s ease both;
}
.mtile::before{
  content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--red),var(--ember),transparent);
  transform:scaleX(0); transition:transform .35s ease; transform-origin:left;
}
.mtile:hover::before{ transform:scaleX(1); }
.mtile:hover{ background:var(--iron3); border-color:var(--bhi); }
.ml{ font-family:var(--fd); font-size:8px; letter-spacing:3px; color:var(--chr-md); margin-bottom:8px; text-transform:uppercase; }
.mv{ font-family:var(--fh); font-size:30px; color:var(--red-hi); text-shadow:0 0 14px rgba(220,38,38,.4); line-height:1; }
.mbar{ height:2px; background:var(--iron4); border-radius:99px; margin-top:10px; overflow:hidden; }
.mfill{ height:100%; background:linear-gradient(90deg,var(--red),var(--ember)); border-radius:99px; animation:barf 1.4s cubic-bezier(.22,1,.36,1) both .4s; transform-origin:left; }

/* ══ ANNUNCIATORS ══ */
.ann{ display:flex; align-items:flex-start; gap:12px; padding:13px 17px; border-radius:4px; margin:8px 0; font-family:var(--fd); font-size:11px; letter-spacing:.3px; }
.ann-ok  { background:var(--grn-lo); border:1px solid rgba(34,197,94,.22);  border-left:3px solid var(--green); color:#7fdd9f; }
.ann-warn{ background:var(--emb-lo); border:1px solid rgba(249,115,22,.22); border-left:3px solid var(--ember); color:#fdba74; }
.ann-crit{ background:var(--red-lo); border:1px solid rgba(220,38,38,.28);  border-left:3px solid var(--red);   color:#fca5a5; animation:critF 1.8s ease-in-out infinite; }
.ann-info{ background:var(--cyn-lo); border:1px solid rgba(6,182,212,.2);   border-left:3px solid var(--cyan);  color:#67e8f9; }
@keyframes critF{ 0%,100%{box-shadow:none;} 50%{box-shadow:0 0 22px rgba(220,38,38,.22);} }

/* ══ RUL CARDS ══ */
.rul-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
.rul-card{
  background:var(--iron2); border:1px solid var(--border); border-radius:4px;
  padding:20px 16px; text-align:center; position:relative; overflow:hidden;
  animation:tileUp .6s cubic-bezier(.34,1.56,.64,1) both;
}
.rul-card::before{ content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.rc-r::before{ background:var(--red);   }
.rc-e::before{ background:var(--ember); }
.rc-g::before{ background:var(--green); }
.rul-label{ font-family:var(--fd); font-size:8px; letter-spacing:3px; color:var(--chr-md); text-transform:uppercase; margin-bottom:10px; }
.rul-val{ font-family:var(--fh); font-size:38px; line-height:1; letter-spacing:1px; }
.rul-val.r{ color:var(--red-hi); text-shadow:0 0 20px rgba(220,38,38,.5); }
.rul-val.e{ color:var(--ember);  text-shadow:0 0 20px rgba(249,115,22,.5); }
.rul-val.g{ color:var(--green);  text-shadow:0 0 20px rgba(34,197,94,.5);  }
.rul-unit{ font-family:var(--fd); font-size:9px; color:var(--chr-lo); margin-top:5px; letter-spacing:1px; }
.rul-bar{ height:4px; background:var(--iron4); border-radius:99px; margin-top:12px; overflow:hidden; }
.rul-fill{ height:100%; border-radius:99px; animation:barf 1.5s cubic-bezier(.22,1,.36,1) both .45s; transform-origin:left; }
.rf-r{ background:linear-gradient(90deg,var(--red),rgba(220,38,38,0.3)); }
.rf-e{ background:linear-gradient(90deg,var(--ember),rgba(249,115,22,0.3)); }
.rf-g{ background:linear-gradient(90deg,var(--green),rgba(34,197,94,0.3)); }

/* ══ RESULT CARD ══ */
.res-card{ border-radius:4px; padding:22px 16px; text-align:center; position:relative; overflow:hidden; animation:resBounce .5s cubic-bezier(.34,1.56,.64,1) both; }
.res-card.ok   { background:var(--grn-lo); border:2px solid rgba(34,197,94,.3); }
.res-card.warn { background:var(--emb-lo); border:2px solid rgba(249,115,22,.3); }
.res-card.fault{ background:var(--red-lo); border:2px solid rgba(220,38,38,.45); animation:faultB 1.3s ease-in-out infinite; }
@keyframes resBounce{ from{transform:scale(.8) rotate(-4deg);opacity:0;} to{transform:scale(1) rotate(0);opacity:1;} }
@keyframes faultB{ 0%,100%{border-color:rgba(220,38,38,.45);} 50%{border-color:var(--red);box-shadow:0 0 28px rgba(220,38,38,.22);} }
.ri{ font-size:44px; line-height:1; margin-bottom:9px; }
.rl{ font-family:var(--fh); font-size:16px; letter-spacing:3px; text-transform:uppercase; }
.rl.ok  { color:var(--green); }
.rl.warn{ color:var(--ember); }
.rl.fault{ color:var(--red); }
.rp{ font-family:var(--fd); font-size:10px; color:var(--chr-md); letter-spacing:1px; margin-top:5px; }

/* ══ PULSE RINGS ══ */
.rw{ position:relative; display:flex; align-items:center; justify-content:center; min-height:80px; }
.ring{ position:absolute; border-radius:50%; border:1.5px solid; opacity:0; animation:rexp 2.2s ease-out infinite; }
.ring:nth-child(2){ animation-delay:.7s; }
.ring:nth-child(3){ animation-delay:1.4s; }
.rr{ border-color:var(--red);   }
.rg{ border-color:var(--green); }
.re{ border-color:var(--ember); }
@keyframes rexp{ 0%{transform:scale(.3);opacity:.9;} 100%{transform:scale(2.6);opacity:0;} }

/* ══ FEATURE CARDS ══ */
.feat-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:24px; }
.feat-card{
  background:var(--iron1); border:1px solid var(--border); border-radius:4px;
  padding:20px 16px; position:relative; overflow:hidden;
  transition:transform .22s,border-color .22s,box-shadow .22s;
  animation:tileUp .55s cubic-bezier(.34,1.56,.64,1) both;
}
.feat-card::before{ content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.fc-r::before{ background:linear-gradient(90deg,var(--red),transparent); }
.fc-e::before{ background:linear-gradient(90deg,var(--ember),transparent); }
.fc-g::before{ background:linear-gradient(90deg,var(--green),transparent); }
.fc-c::before{ background:linear-gradient(90deg,var(--cyan),transparent); }
.fc-o::before{ background:linear-gradient(90deg,var(--gold),transparent); }
.fc-m::before{ background:linear-gradient(90deg,#a855f7,transparent); }
.feat-card:hover{ transform:translateY(-4px); border-color:var(--bhi); box-shadow:0 10px 32px rgba(0,0,0,.6); }
.fc-icon{ font-size:28px; margin-bottom:10px; line-height:1; }
.fc-title{ font-family:var(--ft); font-size:14px; font-weight:600; letter-spacing:2px; text-transform:uppercase; margin-bottom:7px; color:white; }
.fc-desc{ font-family:var(--fd); font-size:11px; color:var(--chr-md); line-height:1.8; }

/* ══ STAT STRIP ══ */
.stat-strip{
  display:flex; margin-top:20px;
  background:var(--iron1); border:1px solid var(--border); border-radius:4px; overflow:hidden;
}
.ssi{ flex:1; padding:16px 12px; text-align:center; border-right:1px solid var(--border); transition:background .22s; }
.ssi:last-child{ border-right:none; }
.ssi:hover{ background:var(--iron2); }
.ssn{ font-family:var(--fh); font-size:22px; line-height:1; margin-bottom:4px; letter-spacing:2px; }
.ssn.r{ color:var(--red);   text-shadow:0 0 14px rgba(220,38,38,.4); }
.ssn.e{ color:var(--ember); text-shadow:0 0 14px rgba(249,115,22,.4); }
.ssn.g{ color:var(--green); text-shadow:0 0 14px rgba(34,197,94,.4); }
.ssn.c{ color:var(--cyan);  text-shadow:0 0 14px rgba(6,182,212,.4); }
.ssl{ font-family:var(--fd); font-size:8px; letter-spacing:2px; color:var(--chr-lo); text-transform:uppercase; }

/* ══ MACHINE HEALTH REPORT ══ */
.report-wrap{
  background:linear-gradient(135deg, var(--iron1), var(--iron2));
  border:1px solid var(--bhi); border-radius:5px;
  padding:28px 28px 24px; position:relative; overflow:hidden;
  margin-top:10px;
  box-shadow: 0 0 60px rgba(220,38,38,0.08);
}
.report-wrap::before{
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,var(--red),var(--ember),var(--gold),var(--green),var(--cyan));
}
.report-wrap::after{
  content:''; position:absolute; top:3px; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.06),transparent);
}
.rpt-hdr{
  font-family:var(--fh); font-size:15px; letter-spacing:4px;
  color:white; text-transform:uppercase; margin-bottom:22px;
  display:flex; align-items:center; gap:12px;
}
.rpt-dot{ width:8px; height:8px; border-radius:50%; background:var(--red); box-shadow:0 0 12px var(--red); animation:blink 2s ease-in-out infinite; }
.rv-grid{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }
.rv-item{
  background:var(--iron0); border:1px solid var(--border); border-radius:4px;
  padding:18px 16px; position:relative; overflow:hidden;
  transition: border-color .2s, box-shadow .2s;
}
.rv-item:hover{ border-color:var(--bhi); box-shadow:0 4px 20px rgba(220,38,38,0.1); }
.rv-item::before{
  content:''; position:absolute; top:0; left:0; bottom:0; width:3px;
  background:var(--red); opacity:0.5;
}
.rv-key{
  font-family:var(--ft); font-size:13px; font-weight:500;
  letter-spacing:2px; color:var(--chr-md);
  text-transform:uppercase; margin-bottom:9px;
  display:flex; align-items:center; gap:7px;
}
.rv-key::before{
  content:'▸'; color:var(--red); font-size:11px; flex-shrink:0;
}
.rv-val{
  font-family:var(--fh); font-size:22px; letter-spacing:2px;
  line-height:1.1;
}
.rv-val.ok  { color:var(--green); }
.rv-val.warn{ color:var(--ember); }
.rv-val.bad { color:var(--red);   }
.rpt-summary{
  background:var(--iron0); border:1px solid var(--border); border-radius:4px;
  padding:20px 18px; font-family:var(--fd); font-size:11px;
  color:var(--chr-md); line-height:2; letter-spacing:.3px;
}
.rpt-verdict{
  margin-top:14px; padding:15px 18px; border-radius:4px;
  font-family:var(--fh); font-size:13px; letter-spacing:4px; text-transform:uppercase;
}
.rpt-verdict.ok  { background:rgba(34,197,94,0.08);  border:1px solid rgba(34,197,94,0.25);  color:var(--green); }
.rpt-verdict.warn{ background:rgba(249,115,22,0.08); border:1px solid rgba(249,115,22,0.25); color:var(--ember); }
.rpt-verdict.bad { background:rgba(220,38,38,0.08);  border:1px solid rgba(220,38,38,0.28);  color:var(--red);   }

/* ══ RADIO INPUT MODE TOGGLE ══ */
[data-testid="stRadio"] > label {
  font-family: var(--ft) !important; font-size: 12px !important;
  font-weight: 600 !important; letter-spacing: 3px !important;
  color: var(--chr-md) !important; text-transform: uppercase !important;
}
[data-testid="stRadio"] > div {
  background: var(--iron2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important; padding: 6px 10px !important;
  gap: 6px !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label {
  font-family: var(--ft) !important; font-size: 13px !important;
  font-weight: 600 !important; letter-spacing: 2px !important;
  color: var(--chr-md) !important; padding: 6px 16px !important;
  border-radius: 3px !important; border: 1px solid transparent !important;
  transition: all .2s !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--red) !important; color: white !important;
  border-color: var(--red) !important;
  box-shadow: 0 0 14px rgba(220,38,38,0.4) !important;
}

/* ══ NUMBER INPUT STYLING ══ */
[data-testid="stNumberInput"] input {
  background: var(--iron2) !important; border: 1px solid var(--border) !important;
  color: var(--red-hi) !important; border-radius: 3px !important;
  font-family: var(--fh) !important; font-size: 18px !important;
  letter-spacing: 2px !important;
  padding: 10px 14px !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stNumberInput"] input:focus {
  border-color: var(--red) !important;
  box-shadow: 0 0 16px rgba(220,38,38,0.25) !important;
  outline: none !important;
}
[data-testid="stNumberInput"] button {
  background: var(--iron3) !important; border-color: var(--border) !important;
  color: var(--red-hi) !important;
}
[data-testid="stNumberInput"] button:hover {
  background: var(--red) !important; color: white !important;
}
/* ══ SCENARIO ENGINE ══ */
.se-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-bottom:18px; }
.se-card{
  background:var(--iron1); border:1px solid var(--border); border-radius:4px;
  padding:18px 16px; position:relative; overflow:hidden;
  animation:tileUp .5s ease both;
}
.se-card::before{ content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--red),var(--ember),transparent); }
.se-param{ font-family:var(--ft); font-size:14px; font-weight:600; letter-spacing:2px;
  color:white; text-transform:uppercase; margin-bottom:12px; }
.se-row{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.se-badge{ font-family:var(--fd); font-size:9px; letter-spacing:1.5px; padding:3px 9px;
  border-radius:2px; flex-shrink:0; text-transform:uppercase; }
.sb-safe{ background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.3);  color:var(--green); }
.sb-warn{ background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.3); color:var(--ember); }
.sb-crit{ background:rgba(220,38,38,0.12); border:1px solid rgba(220,38,38,0.3);  color:var(--red);   }
.sb-ok  { background:rgba(6,182,212,0.10); border:1px solid rgba(6,182,212,0.25); color:var(--cyan);  }
.se-val { font-family:var(--fh); font-size:20px; letter-spacing:2px; }
.se-margin-bar{ height:6px; background:var(--iron4); border-radius:99px; margin-top:10px; overflow:hidden; }
.se-fill{ height:100%; border-radius:99px; animation:barf 1.4s cubic-bezier(.22,1,.36,1) both .3s; transform-origin:left; }
.se-fill.safe{ background:linear-gradient(90deg,var(--green),rgba(34,197,94,0.3)); }
.se-fill.warn{ background:linear-gradient(90deg,var(--ember),rgba(249,115,22,0.3)); }
.se-fill.crit{ background:linear-gradient(90deg,var(--red),rgba(220,38,38,0.3)); }
/* ── WORK ORDER ── */
.wo-wrap{
  background:var(--iron0); border:1px solid var(--bhi); border-radius:4px;
  padding:24px; position:relative; overflow:hidden; margin-top:4px;
}
.wo-wrap::before{ content:''; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--red),var(--ember),var(--gold),transparent); }
.wo-title{ font-family:var(--fh); font-size:16px; letter-spacing:4px; color:white; text-transform:uppercase; margin-bottom:4px; }
.wo-meta{ font-family:var(--fd); font-size:9px; color:var(--chr-lo); letter-spacing:2px; line-height:2; }
.wo-grid{ display:grid; grid-template-columns:1fr 1fr; gap:0;
  border:1px solid var(--border); border-radius:3px; overflow:hidden; margin:14px 0; }
.wo-cell{ padding:11px 14px; border-right:1px solid var(--border); border-bottom:1px solid var(--border); }
.wo-cell:nth-child(even){ border-right:none; }
.wo-cell:nth-last-child(-n+2){ border-bottom:none; }
.wo-ck{ font-family:var(--fd); font-size:8px; letter-spacing:2px; color:var(--chr-lo); text-transform:uppercase; margin-bottom:5px; }
.wo-cv{ font-family:var(--ft); font-size:14px; font-weight:600; letter-spacing:1px; color:var(--chrome); }
.wo-cv.r{ color:var(--red); } .wo-cv.e{ color:var(--ember); } .wo-cv.g{ color:var(--green); }
.wo-task{
  display:flex; align-items:flex-start; gap:12px; padding:10px 14px;
  border-left:2px solid; background:var(--iron1); border-radius:0 3px 3px 0; margin-bottom:6px;
}
.wo-task.p1{ border-color:var(--red);   } .wo-task.p2{ border-color:var(--ember); } .wo-task.p3{ border-color:var(--green); }
.wo-tnum{ font-family:var(--fh); font-size:16px; letter-spacing:1px; flex-shrink:0; }
.wo-tnum.p1{ color:var(--red); } .wo-tnum.p2{ color:var(--ember); } .wo-tnum.p3{ color:var(--green); }
.wo-tbody{ flex:1; }
.wo-ttitle{ font-family:var(--ft); font-size:13px; font-weight:600; letter-spacing:2px; color:white; text-transform:uppercase; margin-bottom:3px; }
.wo-tdesc{ font-family:var(--fd); font-size:10px; color:var(--chr-md); line-height:1.7; letter-spacing:.3px; }
.wo-sig{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:16px; }
.wo-sigbox{ border-top:1px solid var(--border); padding-top:8px; text-align:center;
  font-family:var(--fd); font-size:8px; letter-spacing:2px; color:var(--chr-lo); text-transform:uppercase; }
div.stButton > button {
  background:var(--red) !important; color:white !important;
  font-family:var(--fh) !important; font-size:14px !important;
  letter-spacing:4px !important; text-transform:uppercase !important;
  padding:11px 30px !important; border:none !important; border-radius:3px !important;
  box-shadow:0 4px 20px rgba(220,38,38,0.4) !important;
  transition:all .22s ease !important;
}
div.stButton > button:hover {
  background:var(--ember) !important;
  box-shadow:0 6px 32px rgba(249,115,22,0.5) !important;
  transform:translateY(-2px) !important;
}

/* ══ FORMS ══ */
[data-testid="stFileUploader"]{ background:var(--iron2) !important; border:1px dashed rgba(220,38,38,0.3) !important; border-radius:4px !important; }
[data-testid="stFileUploader"]:hover{ border-color:var(--red) !important; }
[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div{ background:var(--iron2) !important; border:1px solid var(--border) !important; color:white !important; border-radius:3px !important; }
[data-testid="stExpander"]{ background:var(--iron1) !important; border:1px solid var(--border) !important; border-radius:4px !important; }
[data-testid="stSlider"] .st-bq{ background:var(--red) !important; box-shadow:0 0 10px rgba(220,38,38,0.5) !important; }
[data-testid="stSlider"] .st-br{ background:var(--iron4) !important; }

/* ══ TYPOGRAPHY ══ */
h1,h2,h3{ color:white !important; font-family:var(--ft) !important; }
p,li{ color:var(--chrome) !important; font-family:var(--fd) !important; }
label{ color:var(--chr-md) !important; font-family:var(--fd) !important; font-size:10px !important; letter-spacing:1.5px !important; }
.stCaption{ color:var(--chr-lo) !important; font-family:var(--fd) !important; font-size:10px !important; letter-spacing:1px !important; }
[data-testid="stMetricValue"]{ color:var(--red-hi) !important; font-family:var(--fh) !important; }
[data-testid="stSidebar"]{ background:var(--iron0) !important; border-right:1px solid var(--border) !important; }
[data-testid="stSidebar"] *{ color:var(--chrome) !important; }
[data-testid="stDataFrame"]{ border:1px solid var(--border) !important; border-radius:4px !important; }
hr{ border:none !important; border-top:1px solid var(--border) !important; margin:24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# ANIMATED BG
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="bg-scene">
  <div class="bg-grid"></div>
  <div class="bg-sweep-h"></div>
  <div class="bg-sweep-v"></div>
  <div class="bg-ring" style="width:500px;height:500px;top:-120px;left:-180px;animation-delay:0s;animation-duration:6s;"></div>
  <div class="bg-ring" style="width:320px;height:320px;top:10px;left:-60px;animation-delay:2s;animation-duration:4s;"></div>
  <div class="bg-cog"  style="width:440px;height:440px;top:350px;right:-160px;animation:rot_cw 30s linear infinite;"></div>
  <div class="bg-cog"  style="width:280px;height:280px;top:400px;right:-60px;animation:rot_ccw 20s linear infinite;"></div>
  <div class="bg-cog"  style="width:360px;height:360px;top:180px;left:38%;animation:rot_cw 45s linear infinite;"></div>
  <div class="bg-ring" style="width:200px;height:200px;top:60%;left:60%;animation-delay:1.5s;animation-duration:5s;"></div>
  <div class="bg-ring" style="width:130px;height:130px;top:75%;left:20%;animation-delay:3s;animation-duration:3.5s;"></div>
  <div class="bg-vignette"></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="tb-left">
    <div class="tb-logo">⚡</div>
    <div>
      <div class="tb-brand">DYNAMO</div>
      <div class="tb-sub">Motor Intelligence Platform · IEC 60034 · NEMA MG-1 · Fault Prediction</div>
    </div>
  </div>
  <div class="tb-ticker">
    <span class="tick-inner">
      &nbsp;&nbsp;WINDING TEMP: 92°C &nbsp;|&nbsp; STATOR CURRENT: 48A &nbsp;|&nbsp;
      SHAFT SPEED: 2,840 RPM &nbsp;|&nbsp; BEARING VIB: 3.8 mm/s ALERT &nbsp;|&nbsp;
      TORQUE OUTPUT: 312 Nm &nbsp;|&nbsp; INSULATION: 98 MΩ &nbsp;|&nbsp;
      LOAD FACTOR: 88% &nbsp;|&nbsp; EFFICIENCY: 93.4% &nbsp;|&nbsp;
      FAULT PROB: 31.4% WARN &nbsp;|&nbsp; RUL: 6,420 CYCLES &nbsp;|&nbsp;
      COMMUTATOR: NOMINAL &nbsp;|&nbsp; ARMATURE TEMP: 87°C &nbsp;&nbsp;
    </span>
  </div>
  <div class="tb-leds">
    <div class="led-u"><span class="ld ld-g"></span>STATOR</div>
    <div class="led-u"><span class="ld ld-e"></span>ROTOR</div>
    <div class="led-u"><span class="ld ld-c"></span>DRIVE</div>
    <div class="led-u"><span class="ld ld-r"></span>FAULT</div>
  </div>
</div>
<div class="dstrip">
  <div class="di">WINDING-T   <span class="dv w">92°C</span></div>
  <div class="di">STATOR-I    <span class="dv">48 A</span></div>
  <div class="di">SHAFT-RPM   <span class="dv">2,840</span></div>
  <div class="di">TORQUE      <span class="dv">312 Nm</span></div>
  <div class="di">VIB-RMS     <span class="dv w">3.8 mm/s</span></div>
  <div class="di">BEARING-T   <span class="dv w">74°C</span></div>
  <div class="di">INS-R       <span class="dv">98 MΩ</span></div>
  <div class="di">EFFICIENCY  <span class="dv">93.4%</span></div>
  <div class="di">LOAD-FACTOR <span class="dv w">88%</span></div>
  <div class="di">FAULT-P     <span class="dv c">31.4%</span></div>
  <div class="di">RUL         <span class="dv">6,420 cyc</span></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    if lottie_gear: st_lottie(lottie_gear, height=110, key="sb_gear")
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:16px;letter-spacing:5px;color:white;
                padding-bottom:10px;border-bottom:1px solid rgba(220,38,38,0.15);margin-bottom:14px;">
      OPERATOR PANEL
    </div>
    <div style="font-family:'Courier Prime',monospace;font-size:11px;color:#7a6060;line-height:2.5;">
      <span style="color:#dc2626;">[S01]</span> Ingest motor data<br>
      <span style="color:#dc2626;">[S02]</span> Nameplate readings<br>
      <span style="color:#f97316;">[S03]</span> Winding stream<br>
      <span style="color:#dc2626;">[S04]</span> Parameter correlation<br>
      <span style="color:#dc2626;">[S05]</span> Configure model<br>
      <span style="color:#dc2626;">[S06]</span> Bearing trend<br>
      <span style="color:#dc2626;">[S07]</span> Train fault detector<br>
      <span style="color:#dc2626;">[S08]</span> Diagnostic metrics<br>
      <span style="color:#dc2626;">[S09]</span> Parameter ranking<br>
      <span style="color:#f97316;">[S10]</span> RUL estimation<br>
      <span style="color:#ef4444;">[S11]</span> Live fault scan<br>
      <span style="color:#22c55e;">[S12]</span> Health report
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("""<div style="font-family:'Bebas Neue',sans-serif;font-size:14px;letter-spacing:4px;color:white;margin-bottom:10px;">SAMPLE DATASET</div>""", unsafe_allow_html=True)
    sample = pd.DataFrame({
        "cycle_no":         list(range(1,11)),
        "winding_temp_C":   [72,108,55,142,78,115,60,135,74,122],
        "vibration_mm_s":   [1.8,8.5,1.1,10.1,2.4,7.4,1.4,9.3,2.0,8.0],
        "stator_current_A": [36,64,28,80,40,60,30,74,34,70],
        "shaft_rpm":        [1490,2960,1200,3550,1600,2810,1300,3420,1450,3120],
        "torque_Nm":        [175,365,135,450,195,345,145,425,185,385],
        "insulation_MOhm":  [210,42,260,20,185,58,240,28,200,48],
        "failure":          [0,1,0,1,0,1,0,1,0,1]
    })
    st.dataframe(sample, use_container_width=True, height=195)
    st.download_button("⬇ DOWNLOAD SAMPLE CSV", sample.to_csv(index=False).encode(), "dynamo_sample.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════
h_l, h_r = st.columns([3, 1])
with h_l:
    st.markdown("""
    <div style="padding:20px 4px 0;">
      <div style="font-family:'Courier Prime',monospace;font-size:10px;color:#dc2626;
                  letter-spacing:5px;margin-bottom:16px;display:flex;align-items:center;gap:9px;">
        <span style="width:7px;height:7px;border-radius:50%;background:#22c55e;
                     box-shadow:0 0 12px #22c55e;display:inline-block;"></span>
        MOTOR HEALTH SYSTEM ARMED — IEC 60034 · NEMA MG-1 · ISO 10816
      </div>
      <div style="font-family:'Bebas Neue',sans-serif;
                  font-size:clamp(22px,3.8vw,52px); letter-spacing:4px;
                  line-height:0.95; color:white; text-transform:uppercase;
                  text-shadow:0 0 50px rgba(220,38,38,0.25);">
        ELECTRIC MOTOR<br>FAULT DETECTION<br>
        <span style="color:var(--red);background:linear-gradient(90deg,#dc2626,#f97316,#fbbf24);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                     background-clip:text;letter-spacing:6px;">INTELLIGENCE</span>
      </div>
      <div style="font-family:'Courier Prime',monospace;font-size:12px;
                  color:#4a2a2a;margin-top:16px;max-width:600px;line-height:2;">
        DYNAMO monitors winding temperature, bearing vibration, stator current
        and insulation resistance in real time. Train a logistic regression fault
        detector on your motor CSV data and receive live failure probability —
        before the armature burns out.
      </div>
      <div style="display:flex;gap:8px;margin-top:18px;flex-wrap:wrap;">
        <span style="font-family:'Courier Prime',monospace;font-size:9px;letter-spacing:2px;padding:5px 12px;border-radius:2px;color:#dc2626;border:1px solid rgba(220,38,38,0.3);background:rgba(220,38,38,0.06);">IEC 60034</span>
        <span style="font-family:'Courier Prime',monospace;font-size:9px;letter-spacing:2px;padding:5px 12px;border-radius:2px;color:#f97316;border:1px solid rgba(249,115,22,0.3);background:rgba(249,115,22,0.06);">NEMA MG-1</span>
        <span style="font-family:'Courier Prime',monospace;font-size:9px;letter-spacing:2px;padding:5px 12px;border-radius:2px;color:#22c55e;border:1px solid rgba(34,197,94,0.28);background:rgba(34,197,94,0.06);">ISO 10816</span>
        <span style="font-family:'Courier Prime',monospace;font-size:9px;letter-spacing:2px;padding:5px 12px;border-radius:2px;color:#fbbf24;border:1px solid rgba(251,191,36,0.28);background:rgba(251,191,36,0.06);">MODBUS TCP</span>
        <span style="font-family:'Courier Prime',monospace;font-size:9px;letter-spacing:2px;padding:5px 12px;border-radius:2px;color:#06b6d4;border:1px solid rgba(6,182,212,0.28);background:rgba(6,182,212,0.06);">OPC-UA 2.0</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with h_r:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;padding:14px 0 4px;">
      <div class="mtr-scene">
        <div class="mtr-piston-v"></div>
        <div class="mtr-piston-h"></div>
        <div class="mtr-r1"></div>
        <div class="mtr-r2"></div>
        <div class="mtr-r3"></div>
        <div class="mtr-r4"></div>
        <div class="mtr-r5"></div>
        <div class="mtr-r6"></div>
        <div class="mtr-d1"></div>
        <div class="mtr-d2"></div>
        <div class="mtr-d3"></div>
        <div class="mtr-d4"></div>
        <div class="mtr-spark"></div>
        <div class="mtr-spark2"></div>
        <div class="mtr-core"></div>
      </div>
      <div class="mtr-label">DYNAMO</div>
      <div class="mtr-sublabel">3-Phase Induction Motor · Class F</div>
    </div>
    """, unsafe_allow_html=True)

# Feature cards
st.markdown("""
<div class="feat-grid">
  <div class="feat-card fc-r">
    <div class="fc-icon">🌡️</div>
    <div class="fc-title">Winding Temperature</div>
    <div class="fc-desc">Live stator winding thermal monitoring with IEC Class-F (155°C) limit bands. Detects hot-spot formation and inter-turn short circuit thermal signatures.</div>
  </div>
  <div class="feat-card fc-e">
    <div class="fc-icon">🔩</div>
    <div class="fc-title">Bearing Vibration Analysis</div>
    <div class="fc-desc">ISO 10816 velocity RMS monitoring. Identifies inner/outer race defects, cage slip, rotor imbalance and shaft misalignment via anomaly spike detection.</div>
  </div>
  <div class="feat-card fc-g">
    <div class="fc-icon">⚡</div>
    <div class="fc-title">Fault Probability Engine</div>
    <div class="fc-desc">Logistic regression trained on stator current, winding temp, vibration and insulation resistance. Returns real-time P(fault) with ROC-AUC scoring.</div>
  </div>
  <div class="feat-card fc-c">
    <div class="fc-icon">📉</div>
    <div class="fc-title">RUL Degradation Curve</div>
    <div class="fc-desc">Estimates remaining production cycles, operating hours and calendar days to overhaul using an exponential degradation model from ML probability output.</div>
  </div>
  <div class="feat-card fc-o">
    <div class="fc-icon">🧲</div>
    <div class="fc-title">Insulation Resistance Trend</div>
    <div class="fc-desc">Tracks megohm decay from moisture ingress, contamination and thermal ageing. Triggers IEEE 43 "caution" threshold alerts before winding failure.</div>
  </div>
  <div class="feat-card fc-m">
    <div class="fc-icon">🩺</div>
    <div class="fc-title">Machine Health Report</div>
    <div class="fc-desc">Full motor health scorecard: per-metric grades, fault driver ranking, maintenance recommendation and time-to-failure verdict — generated after training.</div>
  </div>
</div>
<div class="stat-strip">
  <div class="ssi"><div class="ssn r">3-Phase</div><div class="ssl">Motor Class</div></div>
  <div class="ssi"><div class="ssn e">≤40ms</div><div class="ssl">Prediction Latency</div></div>
  <div class="ssi"><div class="ssn g">IEC 60034</div><div class="ssl">Compliance</div></div>
  <div class="ssi"><div class="ssn c">Class F</div><div class="ssl">Insulation Rating</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════
# S01 — INGEST
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="slabel"><span class="sid">S01</span><span class="stitle">Motor Data Ingestion — CSV Upload</span><span class="slive">● AWAITING</span></div>', unsafe_allow_html=True)
u1,u2 = st.columns([3,1])
with u1: file = st.file_uploader("LOAD MOTOR SENSOR CSV", type=["csv"])
with u2: st.caption("Numeric sensor columns + binary failure column (0=Healthy · 1=Fault)")

if not file:
    st.markdown('<div class="ann ann-info"><span style="font-size:17px;">📡</span><span>ARMATURE STANDBY — Upload a motor CSV or download the sample dataset from the sidebar panel. Required columns: numeric sensor readings + binary fault label.</span></div>', unsafe_allow_html=True)
    st.stop()

try:
    df = pd.read_csv(file)
    if df.empty: st.error("Dataset is empty."); st.stop()
except Exception as e:
    st.error(f"Ingest error: {e}"); st.stop()

st.markdown(f'<div class="ann ann-ok"><span style="font-size:17px;">✅</span><span>FLUX CONNECTED — {df.shape[0]:,} operating cycles × {df.shape[1]} sensor channels loaded into DYNAMO memory bank.</span></div>', unsafe_allow_html=True)
with st.expander("S01 — RAW MOTOR LOG PREVIEW"):
    st.dataframe(df, use_container_width=True, height=170)

# ══════════════════════════════════════════════════════════════════════
# S02 — NAMEPLATE READINGS
# ══════════════════════════════════════════════════════════════════════
st.markdown('<div class="slabel"><span class="sid">S02</span><span class="stitle">Nameplate Instrument Readings</span><span class="slive">● LIVE</span></div>', unsafe_allow_html=True)
num_all   = df.select_dtypes(include="number").columns.tolist()
bin_cands = [c for c in num_all if df[c].nunique()==2]
miss      = int(df.isnull().sum().sum())
fail_n    = int(df[bin_cands[-1]].sum()) if bin_cands else 0
fail_pct  = round(fail_n/max(1,df.shape[0])*100,1)

cols5 = st.columns(5)
tiles = [
    (cols5[0],"OPERATING CYCLES",str(df.shape[0]),"r","r",100),
    (cols5[1],"SENSOR CHANNELS",str(df.shape[1]),"c","c",80),
    (cols5[2],"MISSING READINGS",str(miss),"r" if miss>0 else "g","r" if miss>0 else "g",100 if miss==0 else min(99,miss*8)),
    (cols5[3],"FAULT EVENTS",str(fail_n),"e","e",min(99,fail_pct*2)),
    (cols5[4],"FAULT RATE",f"{fail_pct}%","r" if fail_pct>30 else "g","r" if fail_pct>30 else "g",min(99,fail_pct*2)),
]
for col,lbl,val,tc,vc,bw in tiles:
    with col:
        st.markdown(f"""<div class="itile t-{tc}" style="animation:tileUp .5s ease both;">
          <div class="il">{lbl}</div>
          <div class="iv {vc}">{val}</div>
          <div class="ibar"><div class="ibf ib{vc}" style="width:{bw}%;"></div></div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# S03 — WINDING STREAM
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="slabel"><span class="sid">S03</span><span class="stitle">Live Winding Parameter Stream</span><span class="slive">● ENERGISED</span></div>', unsafe_allow_html=True)
rs1,rs2 = st.columns([3,1])
with rs1: stream_col = st.selectbox("SELECT MOTOR PARAMETER", num_all)
with rs2:
    st.markdown("<br>", unsafe_allow_html=True)
    stream_n = st.slider("SAMPLE COUNT", 50, 300, 120, step=10)

if st.button("ENERGISE — START WINDING STREAM"):
    mn  = float(df[stream_col].min())
    mx  = float(df[stream_col].max())
    mu  = float(df[stream_col].mean())
    sig = float(df[stream_col].std())*0.35
    np.random.seed(int(time.time())%9999)
    t   = np.arange(stream_n)
    base   = mu+(mx-mu)*0.15*np.sin(2*np.pi*t/40)
    noise  = np.random.normal(0,sig*0.4,stream_n)
    sp_idx = np.random.choice(stream_n,size=max(1,stream_n//22),replace=False)
    spikes = np.zeros(stream_n); spikes[sp_idx] = np.random.uniform(sig*1.8,sig*3.2,len(sp_idx))
    sv     = np.clip(base+noise+spikes,mn,mx)
    upper  = mu+sig*2; lower = mu-sig*2
    sm     = spikes>0

    fig_rt = go.Figure()
    fig_rt.add_trace(go.Scatter(
        x=np.concatenate([t,t[::-1]]),
        y=np.concatenate([np.full(stream_n,upper),np.full(stream_n,lower)]),
        fill='toself',fillcolor='rgba(220,38,38,0.04)',
        line=dict(color='rgba(0,0,0,0)'),name='CONTROL BAND'))
    fig_rt.add_hline(y=upper,line=dict(color='rgba(220,38,38,0.5)',dash='dash',width=1))
    fig_rt.add_hline(y=lower,line=dict(color='rgba(220,38,38,0.5)',dash='dash',width=1))
    fig_rt.add_trace(go.Scatter(x=t,y=sv,mode='lines',name=stream_col,
        line=dict(color='#dc2626',width=2),fill='tozeroy',fillcolor='rgba(220,38,38,0.04)'))
    fig_rt.add_trace(go.Scatter(x=t[sm],y=sv[sm],mode='markers',name='OVERCURRENT/SPIKE',
        marker=dict(color='#fbbf24',size=11,symbol='x',line=dict(color='#fbbf24',width=2.5))))
    fig_rt.add_trace(go.Scatter(x=t[-1:],y=sv[-1:],mode='markers',name='LIVE',
        marker=dict(color='#22c55e',size=15,symbol='circle',line=dict(color='#22c55e',width=2))))
    fig_rt.update_layout(**CHART,height=290,
        xaxis=dict(**ax("CYCLE INDEX")),yaxis=dict(**ax(stream_col.upper())),
        legend=dict(bgcolor="rgba(0,0,0,0.6)",bordercolor="rgba(220,38,38,0.18)",borderwidth=1,
                    font=dict(family="Courier Prime",size=10,color="#8a7070")))
    st.plotly_chart(fig_rt,use_container_width=True)

    sc4 = st.columns(4)
    stats=[
        (f"{sv[-1]:.2f}","LIVE READING","r"),
        (f"{sv.mean():.2f}","MEAN VALUE","e"),
        (f"{sv.std():.2f}","STD DEV","c"),
        (str(sm.sum()),"ANOMALIES","r" if sm.sum()>3 else "g"),
    ]
    for col,(val,lbl,vc) in zip(sc4,stats):
        cmap={'r':'red-hi','e':'ember','c':'cyan','g':'green'}
        with col:
            st.markdown(f"""<div class="mtile"><div class="ml">{lbl}</div>
            <div class="mv" style="color:var(--{cmap.get(vc,'red-hi')});">{val}</div></div>""",
            unsafe_allow_html=True)
else:
    st.markdown('<div class="ann ann-info"><span style="font-size:17px;">⚡</span><span>COMMUTATOR STANDBY — Click "ENERGISE — START WINDING STREAM" to simulate live motor sensor feed with IEC control-limit bands and anomaly detection.</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# S04 — CORRELATION
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="slabel"><span class="sid">S04</span><span class="stitle">Parameter Correlation Matrix</span></div>', unsafe_allow_html=True)
corr = df.corr(numeric_only=True)
fig_hm = px.imshow(corr,text_auto=".2f",
    color_continuous_scale=[[0,"#0e0606"],[0.35,"#1e0f0f"],[0.7,"#dc2626"],[1,"#fbbf24"]])
fig_hm.update_layout(**CHART,height=360,
    coloraxis_colorbar=dict(tickfont=dict(family="Courier Prime",size=9,color="#8a7070")))
fig_hm.update_traces(textfont=dict(family="Courier Prime",size=9,color="white"))
st.plotly_chart(fig_hm,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# S05 — MODEL CONFIG
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="slabel"><span class="sid">S05</span><span class="stitle">Fault Detector Configuration</span></div>', unsafe_allow_html=True)
num_cols = df.select_dtypes(include="number").columns.tolist()
if len(num_cols)<2: st.error("Need at least 2 numeric columns."); st.stop()
mc1,mc2 = st.columns(2)
with mc1: feature_cols = st.multiselect("MOTOR FEATURE PARAMETERS", options=num_cols, default=num_cols[:-1])
with mc2: target_col   = st.selectbox("FAULT LABEL  (0=Healthy · 1=Fault)", options=num_cols, index=len(num_cols)-1)
if not feature_cols:
    st.markdown('<div class="ann ann-warn"><span style="font-size:17px;">⚠️</span><span>SELECT AT LEAST ONE MOTOR PARAMETER AS FEATURE INPUT.</span></div>', unsafe_allow_html=True); st.stop()
if target_col in feature_cols:
    st.markdown('<div class="ann ann-crit"><span style="font-size:17px;">🚨</span><span>FAULT — Fault label cannot be used as feature input.</span></div>', unsafe_allow_html=True); st.stop()

# ══════════════════════════════════════════════════════════════════════
# S06 — BEARING TREND
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="slabel"><span class="sid">S06</span><span class="stitle">Bearing & Armature Trend Monitor</span><span class="slive">● LIVE</span></div>', unsafe_allow_html=True)
tc1,tc2 = st.columns(2)
with tc1: trend_ch = st.selectbox("MONITOR CHANNEL",feature_cols)
with tc2:
    x_opts = [c for c in df.columns if c!=trend_ch]
    x_ax   = st.selectbox("CYCLE / TIME AXIS",x_opts)
fig_tr = go.Figure()
fig_tr.add_trace(go.Scatter(x=df[x_ax],y=df[trend_ch],mode="lines",name=trend_ch,
    line=dict(color="#dc2626",width=1.9),fill="tozeroy",fillcolor="rgba(220,38,38,0.04)"))
if bin_cands:
    fm=df[bin_cands[-1]]==1
    fig_tr.add_trace(go.Scatter(x=df[x_ax][fm],y=df[trend_ch][fm],mode="markers",name="FAULT",
        marker=dict(color="#fbbf24",size=10,symbol="x",line=dict(color="#fbbf24",width=2.5))))
fig_tr.update_layout(**CHART,height=275,
    xaxis=dict(**ax(x_ax.upper())),yaxis=dict(**ax(trend_ch.upper())),
    legend=dict(bgcolor="rgba(0,0,0,0.6)",bordercolor="rgba(220,38,38,0.18)",borderwidth=1,
                font=dict(family="Courier Prime",size=10,color="#8a7070")))
st.plotly_chart(fig_tr,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# S07 — TRAINING
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="slabel"><span class="sid">S07</span><span class="stitle">Logistic Regression Training Engine</span></div>', unsafe_allow_html=True)
tc_a,bc_a = st.columns([2,1])
with tc_a: test_size = st.slider("HOLDOUT TEST RATIO",0.10,0.40,0.20,step=0.05)
with bc_a:
    st.markdown("<br>",unsafe_allow_html=True)
    train_btn = st.button("FIRE TRAINING SEQUENCE")

if train_btn:
    with st.spinner("Charging commutator — training fault detector..."):
        X=df[feature_cols].copy(); y=df[target_col].copy()
        mask=X.notna().all(axis=1)&y.notna()
        X,y=X[mask],y[mask]
        if len(X)<10: st.error("Insufficient cycles — need 10+ clean records."); st.stop()
        strat=y if y.nunique()==2 else None
        X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=test_size,random_state=42,stratify=strat)
        sc_obj=StandardScaler()
        X_tr_s=sc_obj.fit_transform(X_tr); X_te_s=sc_obj.transform(X_te)
        mdl=LogisticRegression(max_iter=1000,random_state=42)
        mdl.fit(X_tr_s,y_tr)
        yp=mdl.predict(X_te_s); ypr=mdl.predict_proba(X_te_s)[:,1]
        st.session_state.update(dict(model=mdl,scaler=sc_obj,feature_cols=feature_cols,
            target_col=target_col,y_test=y_te,y_pred=yp,y_proba=ypr,trained=True,df=df))
    st.markdown('<div class="ann ann-ok"><span style="font-size:17px;">✅</span><span>ARMATURE CHARGED — Logistic regression coefficients computed. RUL and live fault scan are now armed.</span></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════
if not st.session_state.get("trained"):
    st.stop()

mdl    = st.session_state["model"]
sc_obj = st.session_state["scaler"]
fcols  = st.session_state["feature_cols"]
tcol   = st.session_state["target_col"]
y_te   = st.session_state["y_test"]
yp     = st.session_state["y_pred"]
ypr    = st.session_state["y_proba"]
df     = st.session_state["df"]

# S08 — DIAGNOSTICS
st.divider()
st.markdown('<div class="slabel"><span class="sid">S08</span><span class="stitle">Diagnostic Performance Metrics</span><span class="slive">● COMPUTED</span></div>', unsafe_allow_html=True)
acc  = accuracy_score(y_te,yp)
prec = precision_score(y_te,yp,zero_division=0)
rec  = recall_score(y_te,yp,zero_division=0)
f1   = f1_score(y_te,yp,zero_division=0)
mc4  = st.columns(4)
for col,lbl,val in [(mc4[0],"DETECTION ACCURACY",acc),(mc4[1],"FAULT PRECISION",prec),(mc4[2],"FAULT RECALL",rec),(mc4[3],"F1 HARMONIC",f1)]:
    with col:
        st.markdown(f"""<div class="mtile"><div class="ml">{lbl}</div>
        <div class="mv">{val:.1%}</div>
        <div class="mbar"><div class="mfill" style="width:{val*100:.0f}%;"></div></div>
        </div>""", unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)
v1,v2 = st.columns(2)
with v1:
    cm=confusion_matrix(y_te,yp)
    fig_cm=ff.create_annotated_heatmap(cm.tolist(),
        x=["HEALTHY","FAULT"],y=["HEALTHY","FAULT"],
        colorscale=[[0,"#0e0606"],[0.5,"rgba(220,38,38,0.38)"],[1,"#dc2626"]],
        showscale=False,font_colors=["white"])
    fig_cm.update_layout(**CHART,height=320,
        title=dict(text="CONFUSION MATRIX",font=dict(family="Bebas Neue",size=13,color="#8a7070")))
    st.plotly_chart(fig_cm,use_container_width=True)
with v2:
    fpr_v,tpr_v,_=roc_curve(y_te,ypr); rocauc=auc(fpr_v,tpr_v)
    fig_roc=go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_v,y=tpr_v,mode="lines",name=f"AUC={rocauc:.3f}",
        line=dict(color="#dc2626",width=2.4),fill="tozeroy",fillcolor="rgba(220,38,38,0.05)"))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
        line=dict(dash="dot",color="rgba(100,60,60,0.3)",width=1),name="RANDOM"))
    fig_roc.update_layout(**CHART,height=320,
        title=dict(text="ROC CURVE",font=dict(family="Bebas Neue",size=13,color="#8a7070")),
        xaxis=dict(**ax("FALSE POSITIVE RATE")),yaxis=dict(**ax("TRUE POSITIVE RATE")),
        legend=dict(bgcolor="rgba(0,0,0,0.6)",bordercolor="rgba(220,38,38,0.18)",borderwidth=1,
                    font=dict(family="Courier Prime",size=10,color="#8a7070")))
    st.plotly_chart(fig_roc,use_container_width=True)

# S09 — FEATURE IMPORTANCE
st.divider()
st.markdown('<div class="slabel"><span class="sid">S09</span><span class="stitle">Parameter Fault Driver Ranking</span></div>', unsafe_allow_html=True)
coef_df=pd.DataFrame({
    "Parameter":fcols,
    "Weight":np.abs(mdl.coef_[0]),
    "Role":["FAULT DRIVER" if c>0 else "FAULT SUPPRESSOR" for c in mdl.coef_[0]]
}).sort_values("Weight",ascending=True)
fig_fi=px.bar(coef_df,x="Weight",y="Parameter",orientation="h",color="Role",
    color_discrete_map={"FAULT DRIVER":"#dc2626","FAULT SUPPRESSOR":"#22c55e"})
fig_fi.update_traces(marker_line_width=0,opacity=0.9)
fig_fi.update_layout(**CHART,height=max(260,len(fcols)*55),bargap=0.3,
    title=dict(text="MOTOR PARAMETER FAULT COEFFICIENT",font=dict(family="Bebas Neue",size=13,color="#8a7070")),
    xaxis=dict(**ax("COEFFICIENT MAGNITUDE")),yaxis=dict(**ax()),
    legend=dict(bgcolor="rgba(0,0,0,0.6)",bordercolor="rgba(220,38,38,0.18)",borderwidth=1,
                font=dict(family="Courier Prime",size=10,color="#8a7070")))
st.plotly_chart(fig_fi,use_container_width=True)
if len(fcols)>=2:
    sc1_,sc2_ = st.columns(2)
    xc=sc1_.selectbox("X PARAMETER",fcols,index=0); yc=sc2_.selectbox("Y PARAMETER",fcols,index=min(1,len(fcols)-1))
    fig_sc=px.scatter(df,x=xc,y=yc,color=df[tcol].astype(str),
        color_discrete_map={"0":"#22c55e","1":"#dc2626"})
    fig_sc.update_traces(marker=dict(size=9,opacity=0.85,line=dict(width=0.5,color="rgba(0,0,0,0.3)")))
    fig_sc.update_layout(**CHART,height=340,
        title=dict(text="PARAMETER SPACE — HEALTHY vs FAULT",font=dict(family="Bebas Neue",size=13,color="#8a7070")),
        xaxis=dict(**ax(xc.upper())),yaxis=dict(**ax(yc.upper())),
        legend=dict(bgcolor="rgba(0,0,0,0.6)",bordercolor="rgba(220,38,38,0.18)",borderwidth=1,
                    font=dict(family="Courier Prime",size=10,color="#8a7070")))
    st.plotly_chart(fig_sc,use_container_width=True)

# S10 — RUL
st.divider()
st.markdown('<div class="slabel"><span class="sid">S10</span><span class="stitle">Remaining Useful Life Estimation</span><span class="slive">● ARMED</span></div>', unsafe_allow_html=True)
mean_proba = float(np.mean(ypr))
health     = max(0.0, 1.0-mean_proba)
rul_cycles = int(health*9500*(1-mean_proba*0.4))
rul_hrs    = round(rul_cycles*0.25,1)
rul_days   = round(rul_hrs/24,1)
health_pct = round(health*100,1)

rc1,rc2,rc3 = st.columns(3)
with rc1:
    st.markdown(f"""<div class="rul-card rc-r">
      <div class="rul-label">REMAINING CYCLES</div>
      <div class="rul-val r">{rul_cycles:,}</div>
      <div class="rul-unit">Production Cycles Until Overhaul</div>
      <div class="rul-bar"><div class="rul-fill rf-r" style="width:{min(99,health_pct)}%;"></div></div>
    </div>""", unsafe_allow_html=True)
with rc2:
    st.markdown(f"""<div class="rul-card rc-e">
      <div class="rul-label">DAYS TO OVERHAUL</div>
      <div class="rul-val e">{rul_days}</div>
      <div class="rul-unit">Calendar Days</div>
      <div class="rul-bar"><div class="rul-fill rf-e" style="width:{min(99,health_pct)}%;"></div></div>
    </div>""", unsafe_allow_html=True)
with rc3:
    st.markdown(f"""<div class="rul-card rc-g">
      <div class="rul-label">ARMATURE HEALTH INDEX</div>
      <div class="rul-val g">{health_pct}%</div>
      <div class="rul-unit">Motor Health Score</div>
      <div class="rul-bar"><div class="rul-fill rf-g" style="width:{health_pct}%;"></div></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
cyc_arr  = np.linspace(0,rul_cycles*1.5,200)
k        = mean_proba*0.0005+0.00008
health_c = np.exp(-k*cyc_arr)*100
fig_rul  = go.Figure()
fig_rul.add_vrect(x0=rul_cycles*0.8,x1=rul_cycles,
    fillcolor="rgba(220,38,38,0.06)",line_width=0,annotation_text="MAINTENANCE ZONE")
fig_rul.add_vline(x=rul_cycles,line=dict(color="#f97316",dash="dash",width=1.5))
fig_rul.add_trace(go.Scatter(x=cyc_arr,y=health_c,mode="lines",name="HEALTH CURVE",
    line=dict(color="#dc2626",width=2.3),fill="tozeroy",fillcolor="rgba(220,38,38,0.04)"))
fig_rul.add_hline(y=20,line=dict(color="#fbbf24",dash="dot",width=1.3),annotation_text="CRITICAL WINDING LIMIT")
fig_rul.update_layout(**CHART,height=280,
    title=dict(text="MOTOR DEGRADATION CURVE — ARMATURE HEALTH vs OPERATING CYCLES",
               font=dict(family="Bebas Neue",size=13,color="#8a7070")),
    xaxis=dict(**ax("PRODUCTION CYCLES")),yaxis=dict(**ax("HEALTH INDEX (%)")),
    legend=dict(bgcolor="rgba(0,0,0,0.6)",bordercolor="rgba(220,38,38,0.18)",borderwidth=1,
                font=dict(family="Courier Prime",size=10,color="#8a7070")))
st.plotly_chart(fig_rul,use_container_width=True)

# S11 — FAULT SCAN
st.divider()
st.markdown('<div class="slabel"><span class="sid">S11</span><span class="stitle">Live Fault Prediction Scan</span><span class="swarn">● ARMED</span></div>', unsafe_allow_html=True)
st.caption("// DIAL IN MOTOR PARAMETERS — FIRE SCAN TO RETURN FAULT PROBABILITY")

# Input mode toggle
mode_col, _ = st.columns([2, 3])
with mode_col:
    input_mode = st.radio(
        "PARAMETER INPUT MODE",
        ["🎚️  SLIDER", "⌨️  MANUAL ENTRY"],
        horizontal=True,
        label_visibility="visible"
    )

st.markdown("<br>", unsafe_allow_html=True)
n   = len(fcols)
usr = {}

if input_mode == "🎚️  SLIDER":
    sl = st.columns(min(3, n))
    for i, col in enumerate(fcols):
        mn_v  = float(df[col].min())
        mx_v  = float(df[col].max())
        mean_v= float(df[col].mean())
        with sl[i % min(3, n)]:
            usr[col] = st.slider(
                col.upper(), mn_v, mx_v, mean_v,
                step=round((mx_v - mn_v) / 100, 4),
                help=f"Range: {mn_v:.2f} → {mx_v:.2f}  |  Mean: {mean_v:.2f}"
            )
else:
    # Manual number entry — 2 columns of number inputs
    ni_cols = st.columns(2)
    for i, col in enumerate(fcols):
        mn_v  = float(df[col].min())
        mx_v  = float(df[col].max())
        mean_v= float(df[col].mean())
        step  = round((mx_v - mn_v) / 100, 4)
        with ni_cols[i % 2]:
            st.markdown(f"""
            <div style="font-family:'Teko',sans-serif;font-size:11px;font-weight:500;
                        letter-spacing:2px;color:#7a6060;text-transform:uppercase;
                        margin-bottom:2px;padding-left:2px;">
              {col.upper()}
              <span style="font-family:'Courier Prime',monospace;font-size:9px;
                           color:#3a2828;margin-left:8px;">
                [{mn_v:.2f} → {mx_v:.2f}]
              </span>
            </div>""", unsafe_allow_html=True)
            raw = st.number_input(
                label=col,
                min_value=mn_v,
                max_value=mx_v,
                value=mean_v,
                step=step,
                format="%.4f",
                label_visibility="collapsed",
                key=f"ni_{col}"
            )
            usr[col] = raw

st.markdown("<br>",unsafe_allow_html=True)
pb1,pb2 = st.columns([2,1])
with pb1: run_btn = st.button("FIRE — EXECUTE FAULT SCAN")
with pb2:
    if lottie_scan: st_lottie(lottie_scan,height=68,key="ls_run")

if run_btn:
    inp   = sc_obj.transform(pd.DataFrame([usr]))
    pred  = mdl.predict(inp)[0]
    proba = mdl.predict_proba(inp)[0][1]
    risk  = proba*100

    if risk<40:
        state="ok";    rr_cls="rg"; gc="#22c55e"
        stxt="HEALTHY — WINDING WITHIN RATED LIMITS"
        ahtml='<div class="ann ann-ok"><span style="font-size:17px;">✅</span><span>MOTOR HEALTHY — All parameters within IEC 60034 limits. No maintenance intervention required. Continue routine monitoring per NEMA MG-1 schedule.</span></div>'
        asrc=lottie_success; ak="r_ok"
    elif risk<70:
        state="warn";  rr_cls="re"; gc="#f97316"
        stxt="CAUTION — BEARING DEGRADATION DETECTED"
        ahtml='<div class="ann ann-warn"><span style="font-size:17px;">⚠️</span><span>CAUTION — Elevated winding fault probability. Schedule bearing inspection within 48h. Run insulation resistance test. Consider 15% load derating.</span></div>'
        asrc=lottie_warning; ak="r_warn"
    else:
        state="fault"; rr_cls="rr"; gc="#dc2626"
        stxt="CRITICAL — IMMINENT ARMATURE FAILURE"
        ahtml='<div class="ann ann-crit"><span style="font-size:17px;">🚨</span><span>CRITICAL — DE-ENERGISE MOTOR IMMEDIATELY. Winding failure probability exceeds safe threshold. Disconnect from supply and perform full stator rewind or replacement.</span></div>'
        asrc=lottie_warning; ak="r_fault"

    st.markdown(ahtml,unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    g1,g2,g3 = st.columns([2,1,1])
    with g1:
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number",value=risk,
            number=dict(suffix="%",font=dict(size=54,family="Bebas Neue",color=gc)),
            title=dict(
                text=f"FAULT PROBABILITY<br><span style='font-size:10px;font-family:Courier Prime,monospace;color:{gc};letter-spacing:3px;'>{stxt}</span>",
                font=dict(family="Bebas Neue",size=14,color="#8a7070")),
            gauge=dict(
                axis=dict(range=[0,100],
                          tickfont=dict(family="Courier Prime",size=9,color="#8a7070"),
                          tickcolor="#3a2828"),
                bar=dict(color=gc,thickness=0.24),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=1,bordercolor="rgba(220,38,38,0.15)",
                steps=[
                    dict(range=[0,40],  color="rgba(34,197,94,0.05)"),
                    dict(range=[40,70], color="rgba(249,115,22,0.06)"),
                    dict(range=[70,100],color="rgba(220,38,38,0.09)"),
                ],
                threshold=dict(line=dict(color="white",width=2),value=risk)
            )
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Courier Prime",color="#8a7070"),
            margin=dict(l=18,r=18,t=90,b=14),height=420)
        st.plotly_chart(fig_g,use_container_width=True)

    with g2:
        st.markdown("<br>",unsafe_allow_html=True)
        if asrc: st_lottie(asrc,height=150,key=ak)
        st.markdown(f"""
        <div class="rw" style="margin-top:10px;">
          <div class="ring {rr_cls}" style="width:55px;height:55px;"></div>
          <div class="ring {rr_cls}" style="width:55px;height:55px;"></div>
          <div class="ring {rr_cls}" style="width:55px;height:55px;"></div>
        </div>""", unsafe_allow_html=True)

    with g3:
        st.markdown("<br><br>",unsafe_allow_html=True)
        pred_icon  = "⚠️" if pred==1 else "✅"
        pred_label = "FAULT DETECTED" if pred==1 else "HEALTHY"
        st.markdown(f"""
        <div class="res-card {state}">
          <div class="ri">{pred_icon}</div>
          <div class="rl {state}">{pred_label}</div>
          <div class="rp">P(fault) = {proba:.4f}</div>
        </div><br>
        <div class="mtile" style="text-align:center;padding:14px 12px;">
          <div class="ml">RISK LEVEL</div>
          <div class="mv" style="color:{gc};text-shadow:0 0 14px {gc}66;">{risk:.1f}%</div>
          <div class="mbar" style="margin-top:10px;">
            <div class="mfill" style="width:{risk:.0f}%;background:{gc};"></div>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# S12 — MACHINE HEALTH REPORT
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="slabel"><span class="sid">S12</span><span class="stitle">Machine Health Report</span><span class="slive">● GENERATED</span></div>', unsafe_allow_html=True)

acc   = accuracy_score(y_te,yp)
prec  = precision_score(y_te,yp,zero_division=0)
rec   = recall_score(y_te,yp,zero_division=0)
f1    = f1_score(y_te,yp,zero_division=0)
fpr_r,tpr_r,_ = roc_curve(y_te,ypr); rocauc = auc(fpr_r,tpr_r)
mean_proba     = float(np.mean(ypr))
health_pct     = round(max(0.0,1.0-mean_proba)*100,1)
rul_cycles     = int(max(0.0,1.0-mean_proba)*9500*(1-mean_proba*0.4))
rul_days       = round(rul_cycles*0.25/24,1)

if health_pct>70:   ov_cls="ok";   ov_txt="MOTOR IN GOOD HEALTH — CONTINUE OPERATION";          ov_icon="✅"
elif health_pct>40: ov_cls="warn"; ov_txt="MOTOR DEGRADING — SCHEDULE OVERHAUL";                 ov_icon="⚠️"
else:               ov_cls="bad";  ov_txt="CRITICAL — DE-ENERGISE AND WITHDRAW FROM SERVICE";    ov_icon="🚨"

def grade(v,g,w):
    return ("ok","NOMINAL") if v>=g else (("warn","MARGINAL") if v>=w else ("bad","BELOW SPEC"))

acc_c,acc_g   = grade(acc,  0.85,0.70)
prec_c,prec_g = grade(prec, 0.80,0.65)
rec_c,rec_g   = grade(rec,  0.80,0.65)
f1_c,f1_g     = grade(f1,   0.80,0.65)
roc_c,roc_g   = grade(rocauc,0.85,0.70)
hlth_c,hlth_g = grade(health_pct/100,0.70,0.40)

abs_coef = np.abs(mdl.coef_[0])
top_idx  = int(np.argmax(abs_coef))
top_feat = fcols[top_idx]
top_dir  = "drives fault probability HIGHER" if mdl.coef_[0][top_idx]>0 else "suppresses fault probability"

if health_pct>70:
    maint_rec="No corrective action required. Maintain IEC 60034 preventive schedule. Next planned bearing regreasing per NEMA MG-1 calendar. Continue online vibration monitoring via ISO 10816."
elif health_pct>40:
    maint_rec=f"Schedule bearing inspection and winding resistance test within {max(1,int(rul_days*0.3))} days. Check insulation class compliance. Verify cooling air circulation. Consider 15% load derating until cleared."
else:
    maint_rec="DE-ENERGISE MOTOR IMMEDIATELY. Full stator rewind or motor replacement required. Test driven equipment for collateral damage. Do not re-energise until megohm test confirms insulation integrity >100 MΩ."

st.markdown(f"""
<div class="report-wrap">
  <div class="rpt-hdr">
    <span class="rpt-dot"></span>
    {ov_icon} DYNAMO MACHINE HEALTH REPORT — COMPREHENSIVE MOTOR DIAGNOSTIC
  </div>
  <div class="rv-grid">
    <div class="rv-item">
      <div class="rv-key">Armature Health Index</div>
      <div class="rv-val {hlth_c}">{health_pct}% — {hlth_g}</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">Remaining Useful Life</div>
      <div class="rv-val {hlth_c}">{rul_cycles:,} cycles · {rul_days} days</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">Mean Fault Probability</div>
      <div class="rv-val {'bad' if mean_proba>0.6 else 'warn' if mean_proba>0.3 else 'ok'}">{mean_proba*100:.1f}%</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">Detection Accuracy</div>
      <div class="rv-val {acc_c}">{acc:.1%} — {acc_g}</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">Fault Precision / Recall</div>
      <div class="rv-val {prec_c}">{prec:.1%} · {rec:.1%}</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">ROC-AUC Score</div>
      <div class="rv-val {roc_c}">{rocauc:.3f} — {roc_g}</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">F1 Harmonic Score</div>
      <div class="rv-val {f1_c}">{f1:.1%} — {f1_g}</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">Primary Fault Driver</div>
      <div class="rv-val warn">{top_feat.upper()}</div>
    </div>
    <div class="rv-item">
      <div class="rv-key">Driver Influence Direction</div>
      <div class="rv-val warn">↑ {top_feat} {top_dir}</div>
    </div>
  </div>
  <div class="rpt-summary">
    <span style="color:white;font-family:'Bebas Neue',sans-serif;font-size:13px;letter-spacing:3px;">DIAGNOSTIC SUMMARY</span><br><br>
    The DYNAMO fault detector was trained on <strong style="color:#ef4444;">{len(fcols)} motor parameters</strong>
    across <strong style="color:#ef4444;">{len(y_te)} test cycles</strong>.
    Detection accuracy reached <strong style="color:#ef4444;">{acc:.1%}</strong>
    with a ROC-AUC of <strong style="color:#ef4444;">{rocauc:.3f}</strong>,
    indicating <em style="color:#d4c4c4;">{'excellent' if rocauc>0.9 else 'good' if rocauc>0.75 else 'moderate'}</em>
    fault discrimination capability across the test partition.<br><br>
    The dominant fault driver is <strong style="color:#ef4444;">{top_feat.upper()}</strong>,
    which {top_dir}.
    Mean fault probability across the test set is <strong style="color:#ef4444;">{mean_proba*100:.1f}%</strong>,
    mapping to an armature health index of <strong style="color:#ef4444;">{health_pct}%</strong>
    and an estimated remaining life of <strong style="color:#ef4444;">{rul_cycles:,} production cycles</strong>
    ({rul_days} calendar days).<br><br>
    <span style="color:white;font-family:'Bebas Neue',sans-serif;font-size:13px;letter-spacing:3px;">MAINTENANCE RECOMMENDATION</span><br><br>
    {maint_rec}
  </div>
  <div class="rpt-verdict {ov_cls}">
    {ov_icon} &nbsp; VERDICT: {ov_txt}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# S13 — PREDICTIVE FAILURE SCENARIO ENGINE + WORK ORDER GENERATOR
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="slabel"><span class="sid">S13</span><span class="stitle">Predictive Failure Scenario Engine</span><span class="slive">● EXCLUSIVE</span></div>', unsafe_allow_html=True)
st.caption("// FINDS EXACT TIPPING POINTS FOR EACH PARAMETER · MONTE CARLO RISK SIMULATION · AUTO WORK ORDER")

# ── Build current operating point from df means ──
current_vals = {c: float(df[c].mean()) for c in fcols}
sweep_pts    = 120

tab1, tab2, tab3 = st.tabs([
    "🎯  TIPPING POINT FINDER",
    "🎲  MONTE CARLO SIMULATION",
    "📋  WORK ORDER GENERATOR",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1 — TIPPING POINT FINDER
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div style="font-family:'Courier Prime',monospace;font-size:11px;color:#4a2a2a;
                line-height:2;margin-bottom:20px;padding:12px 16px;
                background:rgba(220,38,38,0.04);border-left:2px solid rgba(220,38,38,0.3);border-radius:0 3px 3px 0;">
    For each motor parameter, all others are held at their current mean while the target parameter
    is swept from minimum to maximum. The exact values where fault probability crosses
    <span style="color:#f97316;">40% (CAUTION)</span> and
    <span style="color:#dc2626;">70% (CRITICAL)</span> are identified.
    This gives the maintenance engineer a precise <strong style="color:white;">safe operating envelope</strong> for every sensor channel.
    </div>
    """, unsafe_allow_html=True)

    tipping_results = []
    for target_feat in fcols:
        mn_t  = float(df[target_feat].min())
        mx_t  = float(df[target_feat].max())
        sweep = np.linspace(mn_t, mx_t, sweep_pts)
        probs = []
        for sv in sweep:
            row = {c: current_vals[c] for c in fcols}
            row[target_feat] = sv
            p = mdl.predict_proba(sc_obj.transform(pd.DataFrame([row])))[0][1]
            probs.append(p)
        probs = np.array(probs)

        # Find crossing points
        caution_cross = None
        critical_cross = None
        safe_range_pct = 100.0
        for idx in range(len(probs)-1):
            if caution_cross is None and (
                (probs[idx] < 0.40 and probs[idx+1] >= 0.40) or
                (probs[idx] >= 0.40 and probs[idx+1] < 0.40)
            ):
                caution_cross = round(float(sweep[idx]), 4)
            if critical_cross is None and (
                (probs[idx] < 0.70 and probs[idx+1] >= 0.70) or
                (probs[idx] >= 0.70 and probs[idx+1] < 0.70)
            ):
                critical_cross = round(float(sweep[idx]), 4)

        # Safe margin = % of range below caution threshold
        safe_count = np.sum(probs < 0.40)
        safe_range_pct = round(safe_count / sweep_pts * 100, 1)

        current_p = float(mdl.predict_proba(sc_obj.transform(
            pd.DataFrame([current_vals])))[0][1])

        tipping_results.append({
            "feat": target_feat, "sweep": sweep, "probs": probs,
            "caution": caution_cross, "critical": critical_cross,
            "safe_pct": safe_range_pct, "current_p": current_p,
            "current_v": current_vals[target_feat],
            "mn": mn_t, "mx": mx_t,
        })

    # Render tipping point charts
    for tr in tipping_results:
        feat       = tr["feat"]
        sweep      = tr["sweep"]
        probs      = tr["probs"]
        cur_v      = tr["current_v"]
        caution    = tr["caution"]
        critical   = tr["critical"]
        safe_pct   = tr["safe_pct"]

        if safe_pct >= 70:    risk_cls = "safe"; risk_lbl = "WIDE MARGIN"
        elif safe_pct >= 40:  risk_cls = "warn"; risk_lbl = "NARROW MARGIN"
        else:                 risk_cls = "crit"; risk_lbl = "NEAR LIMIT"

        fig_tp = go.Figure()

        # Shaded risk zones
        fig_tp.add_hrect(y0=0,   y1=0.40, fillcolor="rgba(34,197,94,0.05)",  line_width=0)
        fig_tp.add_hrect(y0=0.40,y1=0.70, fillcolor="rgba(249,115,22,0.06)", line_width=0)
        fig_tp.add_hrect(y0=0.70,y1=1.0,  fillcolor="rgba(220,38,38,0.07)",  line_width=0)

        # Threshold lines
        fig_tp.add_hline(y=0.40, line=dict(color="rgba(249,115,22,0.6)",dash="dash",width=1.2),
                         annotation_text="CAUTION 40%", annotation_font_size=9,
                         annotation_font_color="#f97316")
        fig_tp.add_hline(y=0.70, line=dict(color="rgba(220,38,38,0.7)",dash="dash",width=1.2),
                         annotation_text="CRITICAL 70%", annotation_font_size=9,
                         annotation_font_color="#dc2626")

        # Probability curve
        fig_tp.add_trace(go.Scatter(
            x=sweep, y=probs, mode="lines", name="P(FAULT)",
            line=dict(color="#dc2626", width=2.5),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.04)"
        ))

        # Current operating point
        cur_p_sweep = float(mdl.predict_proba(sc_obj.transform(
            pd.DataFrame([{**current_vals}])))[0][1])
        cur_col = "#22c55e" if cur_p_sweep < 0.40 else ("#f97316" if cur_p_sweep < 0.70 else "#dc2626")
        fig_tp.add_trace(go.Scatter(
            x=[cur_v], y=[cur_p_sweep], mode="markers", name="CURRENT",
            marker=dict(color=cur_col, size=14, symbol="diamond",
                        line=dict(color="white", width=1.5))
        ))

        # Caution crossing marker
        if caution is not None:
            fig_tp.add_vline(x=caution, line=dict(color="rgba(249,115,22,0.5)",dash="dot",width=1))
            fig_tp.add_annotation(x=caution, y=0.40, text=f"CAUTION<br>{caution:.2f}",
                font=dict(family="Courier Prime",size=8,color="#f97316"),
                showarrow=True, arrowhead=2, arrowcolor="#f97316", ax=20, ay=-30)

        # Critical crossing marker
        if critical is not None:
            fig_tp.add_vline(x=critical, line=dict(color="rgba(220,38,38,0.5)",dash="dot",width=1))
            fig_tp.add_annotation(x=critical, y=0.70, text=f"CRITICAL<br>{critical:.2f}",
                font=dict(family="Courier Prime",size=8,color="#dc2626"),
                showarrow=True, arrowhead=2, arrowcolor="#dc2626", ax=20, ay=-30)

        fig_tp.update_layout(**CHART, height=240,
            title=dict(text=f"{feat.upper()} — SAFE OPERATING ENVELOPE  |  Safe Range: {safe_pct}%  |  Status: {risk_lbl}",
                       font=dict(family="Bebas Neue",size=13,color="#8a7070")),
            xaxis=dict(**ax(feat.upper())),
            yaxis=dict(**ax("P(FAULT)"), range=[0,1]),
            legend=dict(bgcolor="rgba(0,0,0,0.5)",bordercolor="rgba(220,38,38,0.18)",borderwidth=1,
                        font=dict(family="Courier Prime",size=10,color="#8a7070")),
            showlegend=True)
        st.plotly_chart(fig_tp, use_container_width=True)

        # Summary pill row
        caution_txt  = f"{caution:.3f}" if caution is not None else "NOT FOUND"
        critical_txt = f"{critical:.3f}" if critical is not None else "NOT FOUND"
        safe_cls     = {"safe":"sb-safe","warn":"sb-warn","crit":"sb-crit"}[risk_cls]
        st.markdown(f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px;">
          <div class="se-badge sb-ok">CURRENT: {cur_v:.3f}</div>
          <div class="se-badge sb-warn">⚠ CAUTION THRESHOLD: {caution_txt}</div>
          <div class="se-badge sb-crit">🚨 CRITICAL THRESHOLD: {critical_txt}</div>
          <div class="se-badge {safe_cls}">SAFE RANGE: {safe_pct}% — {risk_lbl}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 2 — MONTE CARLO
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div style="font-family:'Courier Prime',monospace;font-size:11px;color:#4a2a2a;
                line-height:2;margin-bottom:20px;padding:12px 16px;
                background:rgba(220,38,38,0.04);border-left:2px solid rgba(220,38,38,0.3);border-radius:0 3px 3px 0;">
    Simulates <strong style="color:white;">5,000 future operating scenarios</strong> by adding realistic
    Gaussian noise to current mean values (±1 std dev per parameter).
    Shows the probability distribution of fault occurrence — revealing
    <span style="color:#f97316;">how likely a fault is even if you think everything is normal.</span>
    </div>
    """, unsafe_allow_html=True)

    mc_n = 5000
    np.random.seed(42)
    mc_samples = np.zeros((mc_n, len(fcols)))
    for j, col in enumerate(fcols):
        mu_c  = float(df[col].mean())
        sig_c = float(df[col].std()) * 0.5
        mc_samples[:, j] = np.clip(
            np.random.normal(mu_c, sig_c, mc_n),
            float(df[col].min()), float(df[col].max())
        )

    mc_scaled = sc_obj.transform(mc_samples)
    mc_probs  = mdl.predict_proba(mc_scaled)[:, 1]

    healthy_pct = round(float(np.mean(mc_probs < 0.40)) * 100, 1)
    caution_pct = round(float(np.mean((mc_probs >= 0.40) & (mc_probs < 0.70))) * 100, 1)
    critical_pct= round(float(np.mean(mc_probs >= 0.70)) * 100, 1)
    p95         = round(float(np.percentile(mc_probs, 95)) * 100, 1)
    p50         = round(float(np.percentile(mc_probs, 50)) * 100, 1)

    # Stats row
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc_stats = [
        (mc1, "HEALTHY RUNS",   f"{healthy_pct}%",  "g"),
        (mc2, "CAUTION RUNS",   f"{caution_pct}%",  "e"),
        (mc3, "CRITICAL RUNS",  f"{critical_pct}%", "r"),
        (mc4, "MEDIAN P(FAULT)",f"{p50}%",           "c"),
        (mc5, "95th PERCENTILE",f"{p95}%",           "r" if p95>60 else "e"),
    ]
    cmap2 = {"r":"red-hi","e":"ember","g":"green","c":"cyan"}
    for col, lbl, val, vc in mc_stats:
        with col:
            st.markdown(f"""<div class="mtile"><div class="ml">{lbl}</div>
            <div class="mv" style="color:var(--{cmap2[vc]});">{val}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Histogram of fault probabilities
    fig_mc = go.Figure()
    bins = np.linspace(0, 1, 51)

    # Colour each bin
    for i in range(len(bins)-1):
        mask = (mc_probs >= bins[i]) & (mc_probs < bins[i+1])
        cnt  = int(np.sum(mask))
        mid  = (bins[i]+bins[i+1])/2
        col_b = "#22c55e" if mid < 0.40 else ("#f97316" if mid < 0.70 else "#dc2626")
        fig_mc.add_trace(go.Bar(
            x=[round(mid, 3)], y=[cnt],
            marker_color=col_b, marker_opacity=0.8,
            marker_line_width=0, showlegend=False,
            width=0.018
        ))

    # Threshold lines
    fig_mc.add_vline(x=0.40, line=dict(color="rgba(249,115,22,0.7)",dash="dash",width=1.5),
                     annotation_text="CAUTION", annotation_font_size=9,
                     annotation_font_color="#f97316")
    fig_mc.add_vline(x=0.70, line=dict(color="rgba(220,38,38,0.8)",dash="dash",width=1.5),
                     annotation_text="CRITICAL", annotation_font_size=9,
                     annotation_font_color="#dc2626")
    fig_mc.add_vline(x=p50/100, line=dict(color="rgba(6,182,212,0.7)",dash="dot",width=1.2),
                     annotation_text=f"P50={p50}%", annotation_font_size=9,
                     annotation_font_color="#06b6d4")

    fig_mc.update_layout(**CHART, height=320, bargap=0.05,
        title=dict(text=f"MONTE CARLO FAULT PROBABILITY DISTRIBUTION — {mc_n:,} SIMULATED SCENARIOS",
                   font=dict(family="Bebas Neue",size=13,color="#8a7070")),
        xaxis=dict(**ax("P(FAULT)"), range=[0,1]),
        yaxis=dict(**ax("SCENARIO COUNT")))
    st.plotly_chart(fig_mc, use_container_width=True)

    # Risk interpretation
    if critical_pct > 20:
        mc_verdict = "HIGH OPERATIONAL RISK"
        mc_cls     = "ann-crit"
        mc_icon    = "🚨"
        mc_msg     = f"In {critical_pct}% of simulated operating conditions this motor will exceed the critical fault threshold. Immediate inspection is warranted — normal parameter variation alone is sufficient to trigger failure."
    elif caution_pct + critical_pct > 35:
        mc_verdict = "ELEVATED OPERATIONAL RISK"
        mc_cls     = "ann-warn"
        mc_icon    = "⚠️"
        mc_msg     = f"{caution_pct + critical_pct:.1f}% of simulated scenarios exceed the caution threshold. The motor has limited margin. Plan maintenance within the next scheduled window."
    else:
        mc_verdict = "LOW OPERATIONAL RISK"
        mc_cls     = "ann-ok"
        mc_icon    = "✅"
        mc_msg     = f"{healthy_pct}% of simulated operating scenarios remain in the healthy zone. Motor has adequate safety margin under realistic operating variation."

    st.markdown(f'<div class="ann {mc_cls}"><span style="font-size:17px;">{mc_icon}</span><span><strong>{mc_verdict}</strong> — {mc_msg}</span></div>', unsafe_allow_html=True)

    # Parameter sensitivity (how much each one varies fault prob)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="slabel" style="margin-top:0;"><span class="sid" style="font-size:9px;">SENSITIVITY</span><span class="stitle" style="font-size:13px;">Parameter Risk Contribution</span></div>', unsafe_allow_html=True)

    sensitivities = []
    for j, col in enumerate(fcols):
        hi = np.copy(mc_samples); hi[:, j] = float(df[col].quantile(0.90))
        lo = np.copy(mc_samples); lo[:, j] = float(df[col].quantile(0.10))
        p_hi = float(np.mean(mdl.predict_proba(sc_obj.transform(hi))[:, 1]))
        p_lo = float(np.mean(mdl.predict_proba(sc_obj.transform(lo))[:, 1]))
        sensitivities.append({"Parameter": col, "Delta": abs(p_hi - p_lo) * 100,
                               "Direction": "↑ increases risk" if p_hi > p_lo else "↓ decreases risk"})

    sens_df = pd.DataFrame(sensitivities).sort_values("Delta", ascending=True)
    fig_sens = px.bar(sens_df, x="Delta", y="Parameter", orientation="h",
        color="Delta",
        color_continuous_scale=[[0,"#22c55e"],[0.5,"#f97316"],[1,"#dc2626"]])
    fig_sens.update_traces(
        text=sens_df["Direction"].values,
        textposition="outside",
        textfont=dict(family="Courier Prime",size=9,color="#8a7070"),
        marker_line_width=0, opacity=0.9
    )
    fig_sens.update_layout(**CHART, height=max(240, len(fcols)*55), bargap=0.3, showlegend=False,
        title=dict(text="PARAMETER SENSITIVITY — FAULT PROB SWING (P90 vs P10 OPERATING POINT)",
                   font=dict(family="Bebas Neue",size=12,color="#8a7070")),
        xaxis=dict(**ax("FAULT PROBABILITY SWING (%)")),
        yaxis=dict(**ax()),
        coloraxis_showscale=False)
    st.plotly_chart(fig_sens, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# TAB 3 — WORK ORDER GENERATOR
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div style="font-family:'Courier Prime',monospace;font-size:11px;color:#4a2a2a;
                line-height:2;margin-bottom:20px;padding:12px 16px;
                background:rgba(220,38,38,0.04);border-left:2px solid rgba(220,38,38,0.3);border-radius:0 3px 3px 0;">
    Auto-generates a structured <strong style="color:white;">IEC 60034 / NEMA MG-1 maintenance work order</strong>
    based on your model results, tipping point analysis and health report.
    Download as a text file and hand directly to your maintenance crew.
    </div>
    """, unsafe_allow_html=True)

    import datetime
    wo_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    wo_id   = f"WO-{datetime.datetime.now().strftime('%Y%m%d%H%M')}"

    # Determine priority tasks from tipping points
    tasks_html = ""
    tasks_txt  = ""
    task_num   = 1

    # Always include model-driven tasks
    top_abs    = fcols[int(np.argmax(np.abs(mdl.coef_[0])))]
    top_coef   = float(mdl.coef_[0][int(np.argmax(np.abs(mdl.coef_[0])))])

    # Priority 1: critical fault prob
    acc   = accuracy_score(y_te, yp)
    f1    = f1_score(y_te, yp, zero_division=0)
    mp    = float(np.mean(ypr))
    health_n = round(max(0.0, 1.0 - mp) * 100, 1)
    rul_c = int(max(0.0, 1.0 - mp) * 9500 * (1 - mp * 0.4))
    rul_d = round(rul_c * 0.25 / 24, 1)

    if mp > 0.60:
        p1_cls="p1"; p1_prio="PRIORITY 1 — IMMEDIATE"
        p1_task="DE-ENERGISE AND WITHDRAW MOTOR FROM SERVICE"
        p1_desc=f"Mean fault probability {mp*100:.1f}% exceeds safe operating limit. Motor must be isolated and subject to full diagnostic before re-energisation. Perform winding resistance test, insulation test (IEEE 43) and bearing inspection."
    elif mp > 0.30:
        p1_cls="p2"; p1_prio="PRIORITY 2 — WITHIN 48h"
        p1_task="SCHEDULE BEARING INSPECTION AND WINDING TEST"
        p1_desc=f"Fault probability {mp*100:.1f}% approaching caution threshold. Perform vibration spectrum analysis, insulation resistance measurement and winding temperature check. Reduce load by 15% if symptoms persist."
    else:
        p1_cls="p3"; p1_prio="PRIORITY 3 — ROUTINE"
        p1_task="CONTINUE ROUTINE MONITORING — NO IMMEDIATE ACTION"
        p1_desc=f"Fault probability {mp*100:.1f}% within acceptable range. Maintain standard IEC 60034 preventive schedule. Log all readings and compare at next scheduled outage."

    # Generate parameter-specific tasks from tipping point analysis
    near_limit_params = []
    for tr in tipping_results:
        c_thresh = tr["caution"]
        if c_thresh is not None:
            cur_to_thresh = abs(tr["current_v"] - c_thresh)
            full_range    = tr["mx"] - tr["mn"]
            margin_pct    = (cur_to_thresh / max(1e-9, full_range)) * 100
            if margin_pct < 20:
                near_limit_params.append((tr["feat"], round(margin_pct, 1), tr["current_v"], c_thresh))

    tasks_html += f"""
    <div class="wo-task {p1_cls}">
      <div class="wo-tnum {p1_cls}">01</div>
      <div class="wo-tbody">
        <div class="wo-ttitle">{p1_task}</div>
        <div class="wo-tdesc">{p1_desc}</div>
        <div style="margin-top:5px;font-family:'Courier Prime',monospace;font-size:8px;letter-spacing:1.5px;
                    padding:2px 8px;border-radius:2px;display:inline-block;
                    background:rgba(220,38,38,0.1);color:#f97316;">{p1_prio}</div>
      </div>
    </div>
    """
    tasks_txt += f"\n[TASK 01] {p1_task}\n         {p1_prio}\n         {p1_desc}\n"
    task_num = 2

    # Dominant fault driver task
    tasks_html += f"""
    <div class="wo-task p2">
      <div class="wo-tnum p2">{task_num:02d}</div>
      <div class="wo-tbody">
        <div class="wo-ttitle">INSPECT PRIMARY FAULT DRIVER: {top_abs.upper()}</div>
        <div class="wo-tdesc">ML analysis identified {top_abs.upper()} as the dominant fault driver
          (coefficient magnitude: {abs(top_coef):.4f}, direction: {"increases" if top_coef>0 else "suppresses"} fault risk).
          Focus diagnostic effort on this parameter first. Verify sensor calibration and cross-check with adjacent measurements.</div>
        <div style="margin-top:5px;font-family:'Courier Prime',monospace;font-size:8px;letter-spacing:1.5px;
                    padding:2px 8px;border-radius:2px;display:inline-block;
                    background:rgba(249,115,22,0.1);color:#f97316;">PRIORITY 2 — WITHIN 48h</div>
      </div>
    </div>
    """
    tasks_txt += f"\n[TASK {task_num:02d}] INSPECT PRIMARY FAULT DRIVER: {top_abs.upper()}\n         PRIORITY 2 — WITHIN 48h\n"
    task_num += 1

    # Near-limit parameters
    for param, margin, cur_v, thresh in near_limit_params[:3]:
        tasks_html += f"""
        <div class="wo-task p2">
          <div class="wo-tnum p2">{task_num:02d}</div>
          <div class="wo-tbody">
            <div class="wo-ttitle">NEAR-LIMIT PARAMETER: {param.upper()}</div>
            <div class="wo-tdesc">Current value {cur_v:.3f} is within {margin:.1f}% of the caution threshold
              ({thresh:.3f}). Monitor closely and investigate root cause.
              Check associated wiring, sensor mounting and mechanical coupling.</div>
            <div style="margin-top:5px;font-family:'Courier Prime',monospace;font-size:8px;letter-spacing:1.5px;
                        padding:2px 8px;border-radius:2px;display:inline-block;
                        background:rgba(249,115,22,0.1);color:#f97316;">PRIORITY 2 — MONITOR CLOSELY</div>
          </div>
        </div>
        """
        tasks_txt += f"\n[TASK {task_num:02d}] NEAR-LIMIT: {param.upper()} (margin: {margin:.1f}%)\n"
        task_num += 1

    # Monte Carlo risk task
    mc_risk_txt = f"CRITICAL {critical_pct}%" if critical_pct>15 else f"CAUTION {caution_pct+critical_pct:.1f}%" if caution_pct+critical_pct>25 else f"LOW {healthy_pct}% healthy"
    tasks_html += f"""
    <div class="wo-task p3">
      <div class="wo-tnum p3">{task_num:02d}</div>
      <div class="wo-tbody">
        <div class="wo-ttitle">MONTE CARLO RISK REVIEW — {mc_risk_txt}</div>
        <div class="wo-tdesc">5,000-scenario simulation shows {healthy_pct}% healthy / {caution_pct}% caution / {critical_pct}% critical.
          Median fault probability: {p50}%, worst-case 95th percentile: {p95}%.
          Update maintenance interval if critical percentage exceeds 10%.</div>
        <div style="margin-top:5px;font-family:'Courier Prime',monospace;font-size:8px;letter-spacing:1.5px;
                    padding:2px 8px;border-radius:2px;display:inline-block;
                    background:rgba(34,197,94,0.08);color:#22c55e;">PRIORITY 3 — NEXT SCHEDULED OUTAGE</div>
      </div>
    </div>
    """
    tasks_txt += f"\n[TASK {task_num:02d}] MONTE CARLO RISK: {mc_risk_txt}\n"

    # Render work order
    health_cls = "g" if health_n > 70 else ("e" if health_n > 40 else "r")
    st.markdown(f"""
    <div class="wo-wrap">
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:16px;">
        <div>
          <div class="wo-title">⚡ DYNAMO MAINTENANCE WORK ORDER</div>
          <div class="wo-meta">
            WO NUMBER: {wo_id} &nbsp;|&nbsp; GENERATED: {wo_date}<br>
            STANDARD: IEC 60034 · NEMA MG-1 · ISO 10816 &nbsp;|&nbsp; SOURCE: ML DIAGNOSTIC
          </div>
        </div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;letter-spacing:2px;
                    color:{'#dc2626' if mp>0.6 else '#f97316' if mp>0.3 else '#22c55e'};
                    text-shadow:0 0 20px currentColor;">
          {'CRITICAL' if mp>0.6 else 'CAUTION' if mp>0.3 else 'HEALTHY'}
        </div>
      </div>

      <div class="wo-grid">
        <div class="wo-cell"><div class="wo-ck">Motor Health Index</div>
          <div class="wo-cv {health_cls}">{health_n}%</div></div>
        <div class="wo-cell"><div class="wo-ck">Mean Fault Probability</div>
          <div class="wo-cv {'r' if mp>0.6 else 'e' if mp>0.3 else 'g'}">{mp*100:.1f}%</div></div>
        <div class="wo-cell"><div class="wo-ck">Estimated RUL</div>
          <div class="wo-cv {health_cls}">{rul_c:,} cycles · {rul_d} days</div></div>
        <div class="wo-cell"><div class="wo-ck">Primary Fault Driver</div>
          <div class="wo-cv e">{top_abs.upper()}</div></div>
        <div class="wo-cell"><div class="wo-ck">Model Accuracy / F1</div>
          <div class="wo-cv {'g' if acc>0.85 else 'e'}">{acc:.1%} · {f1:.1%}</div></div>
        <div class="wo-cell"><div class="wo-ck">Monte Carlo Risk</div>
          <div class="wo-cv {'r' if critical_pct>15 else 'e' if critical_pct>5 else 'g'}">{critical_pct}% critical scenarios</div></div>
      </div>

      <div style="font-family:'Bebas Neue',sans-serif;font-size:13px;letter-spacing:3px;
                  color:var(--chr-md);margin-bottom:10px;">MAINTENANCE TASKS</div>
      {tasks_html}

      <div class="wo-sig">
        <div class="wo-sigbox">____________________<br>Maintenance Engineer</div>
        <div class="wo-sigbox">____________________<br>Supervisor Approval</div>
        <div class="wo-sigbox">____________________<br>Date Completed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Build plain-text version for download
    wo_text = f"""
================================================================================
  DYNAMO MOTOR MAINTENANCE WORK ORDER
  {wo_id}
================================================================================
Generated   : {wo_date}
Standard    : IEC 60034 / NEMA MG-1 / ISO 10816 / IEEE 43
Source      : ML Diagnostic — Logistic Regression Fault Detector

MOTOR HEALTH SUMMARY
─────────────────────────────────────────────────────────────────────
Health Index          : {health_n}%
Mean Fault Probability: {mp*100:.1f}%
Estimated RUL         : {rul_c:,} cycles ({rul_d} calendar days)
Primary Fault Driver  : {top_abs.upper()}
Model Accuracy        : {acc:.1%}
F1 Score              : {f1:.1%}
Monte Carlo (5000 sim): {healthy_pct}% healthy / {caution_pct}% caution / {critical_pct}% critical
95th Percentile Risk  : {p95}%

PARAMETER TIPPING POINTS
─────────────────────────────────────────────────────────────────────
"""
    for tr in tipping_results:
        wo_text += f"  {tr['feat'].upper():<28} Current: {tr['current_v']:.3f}  Caution@: {str(tr['caution']):<12} Critical@: {str(tr['critical']):<12} Safe Range: {tr['safe_pct']}%\n"

    wo_text += f"""
MAINTENANCE TASKS
─────────────────────────────────────────────────────────────────────
{tasks_txt}

SIGN-OFF
─────────────────────────────────────────────────────────────────────
Maintenance Engineer: _______________________  Date: __________
Supervisor Approval : _______________________  Date: __________

================================================================================
  DYNAMO · IEC 60034 · NEMA MG-1 · ISO 10816 · STREAMLIT · SCIKIT-LEARN
================================================================================
"""
    st.markdown("<br>", unsafe_allow_html=True)
    dl1, dl2, _ = st.columns([1,1,2])
    with dl1:
        st.download_button(
            "⬇ DOWNLOAD WORK ORDER (.TXT)",
            wo_text.encode("utf-8"),
            file_name=f"DYNAMO_{wo_id}.txt",
            mime="text/plain"
        )
    with dl2:
        wo_csv_rows = [["Parameter","Current Value","Caution Threshold","Critical Threshold","Safe Range %"]]
        for tr in tipping_results:
            wo_csv_rows.append([tr["feat"], f"{tr['current_v']:.4f}", str(tr["caution"]), str(tr["critical"]), f"{tr['safe_pct']}%"])
        wo_csv = "\n".join(",".join(str(x) for x in row) for row in wo_csv_rows)
        st.download_button(
            "⬇ DOWNLOAD TIPPING POINTS (.CSV)",
            wo_csv.encode("utf-8"),
            file_name=f"DYNAMO_TippingPoints_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ══════════════════════════════════════════════════════════════════════
# S14 — AI MOTOR DIAGNOSTIC ASSISTANT (Claude-powered)
# ══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<style>
/* ── AI CHAT INTERFACE ── */
.ai-wrap{
  background:linear-gradient(135deg,var(--iron1),var(--iron0));
  border:1px solid var(--bhi); border-radius:6px;
  padding:0; overflow:hidden; position:relative;
  box-shadow:0 0 60px rgba(220,38,38,0.10);
}
.ai-wrap::before{
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,var(--red),var(--ember),var(--gold),var(--green),var(--cyan),var(--red));
  background-size:200% 100%; animation:aiRainbow 4s linear infinite;
}
@keyframes aiRainbow{ from{background-position:0% 0%;} to{background-position:200% 0%;} }

.ai-header{
  padding:16px 22px; border-bottom:1px solid var(--border);
  display:flex; align-items:center; gap:14px;
  background:rgba(220,38,38,0.04);
}
.ai-logo{
  width:42px; height:42px; border-radius:50%; flex-shrink:0;
  background:conic-gradient(var(--red),var(--ember),var(--gold),var(--red));
  display:flex; align-items:center; justify-content:center; font-size:20px;
  animation:logoBeat 3s ease-in-out infinite;
  box-shadow:0 0 20px rgba(220,38,38,0.4);
}
.ai-htext{ flex:1; }
.ai-hname{
  font-family:'Bebas Neue',sans-serif; font-size:18px; letter-spacing:5px;
  color:white; text-transform:uppercase; line-height:1;
}
.ai-hsub{
  font-family:'Courier Prime',monospace; font-size:9px; color:var(--chr-lo);
  letter-spacing:3px; margin-top:3px; text-transform:uppercase;
}
.ai-status{
  font-family:'Courier Prime',monospace; font-size:9px; color:var(--green);
  letter-spacing:2px; display:flex; align-items:center; gap:6px;
}
.ai-dot{ width:7px; height:7px; border-radius:50%; background:var(--green);
  box-shadow:0 0 10px var(--green); animation:blink 2.5s ease-in-out infinite; }

.chat-history{
  padding:18px 20px; min-height:200px; max-height:520px;
  overflow-y:auto; display:flex; flex-direction:column; gap:14px;
}
.chat-history::-webkit-scrollbar{ width:3px; }
.chat-history::-webkit-scrollbar-thumb{ background:var(--red); border-radius:2px; }

.msg-user{
  align-self:flex-end; max-width:78%;
  background:rgba(220,38,38,0.12); border:1px solid rgba(220,38,38,0.28);
  border-radius:12px 12px 2px 12px; padding:12px 16px;
  font-family:'Courier Prime',monospace; font-size:12px; color:var(--chrome);
  line-height:1.8; letter-spacing:.3px;
}
.msg-user .msg-label{
  font-family:'Teko',sans-serif; font-size:10px; font-weight:600;
  letter-spacing:3px; color:rgba(220,38,38,0.7); margin-bottom:5px;
  text-transform:uppercase;
}
.msg-ai{
  align-self:flex-start; max-width:88%;
  background:rgba(6,182,212,0.05); border:1px solid rgba(6,182,212,0.18);
  border-radius:2px 12px 12px 12px; padding:14px 17px;
  font-family:'Courier Prime',monospace; font-size:12px; color:var(--chrome);
  line-height:1.9; letter-spacing:.2px;
}
.msg-ai .msg-label{
  font-family:'Teko',sans-serif; font-size:10px; font-weight:600;
  letter-spacing:3px; color:rgba(6,182,212,0.7); margin-bottom:6px;
  text-transform:uppercase; display:flex; align-items:center; gap:7px;
}
.msg-ai .msg-label::before{
  content:''; width:6px; height:6px; border-radius:50%;
  background:var(--cyan); box-shadow:0 0 8px var(--cyan); flex-shrink:0;
  animation:blink 2s ease-in-out infinite;
}
.msg-thinking{
  align-self:flex-start;
  background:rgba(6,182,212,0.04); border:1px solid rgba(6,182,212,0.12);
  border-radius:2px 12px 12px 12px; padding:12px 16px;
  font-family:'Courier Prime',monospace; font-size:11px; color:rgba(6,182,212,0.5);
  letter-spacing:2px; display:flex; align-items:center; gap:10px;
}
.think-dots{ display:flex; gap:5px; }
.think-dots span{
  width:6px; height:6px; border-radius:50%; background:var(--cyan);
  animation:thinkBounce 1.2s ease-in-out infinite;
}
.think-dots span:nth-child(2){ animation-delay:.2s; }
.think-dots span:nth-child(3){ animation-delay:.4s; }
@keyframes thinkBounce{
  0%,80%,100%{ transform:translateY(0); opacity:0.3; }
  40%{ transform:translateY(-6px); opacity:1; }
}
.quick-prompts{
  padding:10px 20px 14px; border-top:1px solid var(--border);
  display:flex; gap:8px; flex-wrap:wrap; background:rgba(0,0,0,0.15);
}
.qp-btn{
  font-family:'Courier Prime',monospace; font-size:9px; letter-spacing:1.5px;
  padding:5px 12px; border-radius:3px; cursor:pointer;
  border:1px solid rgba(220,38,38,0.25); background:rgba(220,38,38,0.06);
  color:rgba(220,38,38,0.7); text-transform:uppercase; transition:all .2s;
  white-space:nowrap;
}
.qp-btn:hover{ background:rgba(220,38,38,0.15); border-color:var(--red); color:var(--red-hi); }
.ai-disclaimer{
  font-family:'Courier Prime',monospace; font-size:8px; color:rgba(58,40,40,0.5);
  letter-spacing:1.5px; text-align:center; padding:10px 20px 14px;
  border-top:1px solid rgba(220,38,38,0.06);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="slabel"><span class="sid">S14</span><span class="stitle">AI Motor Diagnostic Assistant</span><span class="slive">● CLAUDE-POWERED</span></div>', unsafe_allow_html=True)

# Build the motor context string from all computed results
acc_s  = accuracy_score(y_te, yp)
prec_s = precision_score(y_te, yp, zero_division=0)
rec_s  = recall_score(y_te, yp, zero_division=0)
f1_s   = f1_score(y_te, yp, zero_division=0)
fpr_s, tpr_s, _ = roc_curve(y_te, ypr)
rocauc_s = auc(fpr_s, tpr_s)
mp_s     = float(np.mean(ypr))
health_s = round(max(0.0, 1.0 - mp_s) * 100, 1)
rul_s    = int(max(0.0, 1.0 - mp_s) * 9500 * (1 - mp_s * 0.4))
rul_d_s  = round(rul_s * 0.25 / 24, 1)
top_i_s  = int(np.argmax(np.abs(mdl.coef_[0])))
top_f_s  = fcols[top_i_s]
top_c_s  = float(mdl.coef_[0][top_i_s])

# Sensor summary table string
sensor_lines = []
for col in fcols:
    mn_c = float(df[col].min()); mx_c = float(df[col].max())
    mu_c = float(df[col].mean()); sd_c = float(df[col].std())
    sensor_lines.append(f"  {col:<32} mean={mu_c:.3f}  std={sd_c:.3f}  min={mn_c:.3f}  max={mx_c:.3f}")
sensor_table = "\n".join(sensor_lines)

coef_lines = []
for col, c in zip(fcols, mdl.coef_[0]):
    coef_lines.append(f"  {col:<32} coef={c:+.4f}  ({'FAULT DRIVER' if c>0 else 'FAULT SUPPRESSOR'})")
coef_table = "\n".join(coef_lines)

MOTOR_CONTEXT = f"""
You are DYNAMO — an expert AI motor diagnostic assistant embedded in a predictive maintenance platform.
You have full access to the following real data from a trained logistic regression fault detector.

=== MOTOR HEALTH SNAPSHOT ===
Health Index             : {health_s}%
Mean Fault Probability   : {mp_s*100:.1f}%
Remaining Useful Life    : {rul_s:,} cycles ({rul_d_s} calendar days)
Overall Verdict          : {'CRITICAL — WITHDRAW FROM SERVICE' if mp_s>0.6 else 'CAUTION — SCHEDULE MAINTENANCE' if mp_s>0.3 else 'HEALTHY — CONTINUE OPERATION'}

=== MODEL PERFORMANCE ===
Accuracy  : {acc_s:.1%}
Precision : {prec_s:.1%}
Recall    : {rec_s:.1%}
F1 Score  : {f1_s:.1%}
ROC-AUC   : {rocauc_s:.3f}
Test Cycles Used: {len(y_te)}

=== FEATURE PARAMETERS (training data statistics) ===
{sensor_table}

=== LOGISTIC REGRESSION COEFFICIENTS ===
{coef_table}

Primary fault driver: {top_f_s.upper()} (coef={top_c_s:+.4f}, {'increases' if top_c_s>0 else 'decreases'} fault probability)

=== APPLICABLE STANDARDS ===
IEC 60034, NEMA MG-1, ISO 10816 (vibration), IEEE 43 (insulation resistance),
IEC 60085 (insulation classes: A=105°C, B=130°C, F=155°C, H=180°C)

You are talking to a maintenance engineer. Be direct, technical, practical and concise.
Use motor engineering terminology. Give specific numbers from the data above.
Format responses clearly with short paragraphs. Never refuse to help.
If you don't know something, say so and suggest what test to run.
"""

# Session state for chat history
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

# Header
st.markdown("""
<div class="ai-wrap">
  <div class="ai-header">
    <div class="ai-logo">🤖</div>
    <div class="ai-htext">
      <div class="ai-hname">DYNAMO AI — Motor Diagnostic Assistant</div>
      <div class="ai-hsub">Powered by Claude · Trained on your motor data · IEC 60034 · NEMA MG-1</div>
    </div>
    <div class="ai-status"><div class="ai-dot"></div>ONLINE</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Quick prompt buttons
st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
quick_prompts = [
    "Why is my motor at risk?",
    "What should I inspect first?",
    "Explain the fault probability",
    "When should I schedule maintenance?",
    "What does the tipping point analysis mean?",
    "Generate a maintenance checklist",
    "Explain the dominant fault driver",
    "How reliable is this model?",
]

qrow = st.columns(4)
for i, qp in enumerate(quick_prompts):
    with qrow[i % 4]:
        if st.button(qp, key=f"qp_{i}"):
            st.session_state["ai_prefill"] = qp

# Chat history display
chat_container = st.container()
with chat_container:
    if not st.session_state.ai_messages:
        st.markdown("""
        <div style="padding:18px 20px;">
          <div class="msg-ai">
            <div class="msg-label">DYNAMO AI</div>
            Motor diagnostic system online. I have full access to your trained model results —
            health index <strong style="color:#dc2626;">{hp}%</strong>,
            fault probability <strong style="color:#f97316;">{fp:.1f}%</strong>,
            RUL <strong style="color:#22c55e;">{rul:,} cycles</strong>,
            primary fault driver <strong style="color:#fbbf24;">{tf}</strong>.<br><br>
            Ask me anything: why the motor is at risk, what to inspect, what the coefficients mean,
            how to interpret tipping points, or what maintenance tasks to prioritise.
          </div>
        </div>
        """.format(hp=health_s, fp=mp_s*100, rul=rul_s, tf=top_f_s.upper()),
        unsafe_allow_html=True)
    else:
        html_msgs = "<div class='chat-history'>"
        for msg in st.session_state.ai_messages:
            if msg["role"] == "user":
                html_msgs += f'<div class="msg-user"><div class="msg-label">ENGINEER</div>{msg["content"]}</div>'
            else:
                safe_content = msg["content"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                html_msgs += f'<div class="msg-ai"><div class="msg-label">DYNAMO AI</div>{safe_content}</div>'
        html_msgs += "</div>"
        st.markdown(html_msgs, unsafe_allow_html=True)

# Input box
prefill_val = st.session_state.pop("ai_prefill", "")
inp_col, btn_col = st.columns([5, 1])
with inp_col:
    user_input = st.text_input(
        "ASK THE DIAGNOSTIC AI",
        value=prefill_val,
        placeholder="e.g. Why is vibration driving most of the fault risk? What should I check first?",
        label_visibility="collapsed",
        key="ai_input"
    )
with btn_col:
    send_btn = st.button("SEND ⚡", key="ai_send")

if (send_btn or (user_input and user_input != prefill_val)) and user_input.strip():
    st.session_state.ai_messages.append({"role": "user", "content": user_input.strip()})

    with st.spinner("DYNAMO AI analysing..."):
        try:
            api_messages = [{"role": "system", "content": MOTOR_CONTEXT}] if False else []
            # Build messages with system prompt embedded in first user message
            messages_for_api = []
            for i, msg in enumerate(st.session_state.ai_messages):
                if i == 0 and msg["role"] == "user":
                    messages_for_api.append({
                        "role": "user",
                        "content": f"[MOTOR SYSTEM CONTEXT]\n{MOTOR_CONTEXT}\n\n[ENGINEER QUESTION]\n{msg['content']}"
                    })
                else:
                    messages_for_api.append(msg)

            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 1000,
                    "system": MOTOR_CONTEXT,
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.ai_messages
                    ]
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                ai_reply = "".join(
                    block.get("text", "")
                    for block in data.get("content", [])
                    if block.get("type") == "text"
                )
                st.session_state.ai_messages.append({"role": "assistant", "content": ai_reply})
            else:
                err = response.json().get("error", {}).get("message", "Unknown error")
                st.session_state.ai_messages.append({
                    "role": "assistant",
                    "content": f"API error ({response.status_code}): {err}\n\nPlease check your API key or network connection."
                })
        except Exception as e:
            st.session_state.ai_messages.append({
                "role": "assistant",
                "content": f"Connection error: {str(e)}\n\nEnsure you have internet access and a valid Anthropic API key configured."
            })

    st.rerun()

# Clear chat button
if st.session_state.ai_messages:
    if st.button("CLEAR CONVERSATION", key="ai_clear"):
        st.session_state.ai_messages = []
        st.rerun()

st.markdown("""
<div class="ai-disclaimer">
  DYNAMO AI · POWERED BY CLAUDE · RESPONSES BASED ON YOUR TRAINED MODEL DATA ·
  NOT A SUBSTITUTE FOR QUALIFIED ENGINEERING JUDGEMENT · IEC 60034 · NEMA MG-1
</div>
""", unsafe_allow_html=True)

# FOOTER
st.divider()
st.markdown("""
<div style="text-align:center;padding:12px 0 6px;font-family:'Courier Prime',monospace;
            font-size:8px;color:rgba(58,40,40,0.55);letter-spacing:3px;">
  DYNAMO MOTOR INTELLIGENCE PLATFORM · LOGISTIC REGRESSION · IEC 60034 · NEMA MG-1 · ISO 10816 ·
  IEEE 43 · MODBUS TCP · STREAMLIT · SCIKIT-LEARN · PLOTLY
</div>
""", unsafe_allow_html=True)