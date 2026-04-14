import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import ast
import json
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Peru Quant Dashboard", layout="wide")

st.title("🎈 My new Streamlit app")

REFRESH = 10000

BANKROLL = st.sidebar.number_input("Bankroll ($)", 10, 1_000_000, 90)
KELLY_FRACTION = st.sidebar.slider("Kelly fraction", 0.0, 1.0, 0.25)

MIN_BET = 1
MAX_BET_FRAC = 0.10
MIN_EDGE = 0.0

st_autorefresh(interval=REFRESH, key="refresh")

# =========================
# FETCH
# =========================
SLUGS = [
    "peru-presidential-election-first-round-2nd-place",
    "peru-presidential-election-first-round-3rd-place"
]

BASE = "https://gamma-api.polymarket.com/events/slug/"

@st.cache_data(ttl=30)
def fetch():
    events = []
    for slug in SLUGS:
        try:
            r = requests.get(BASE + slug, timeout=10)
            if r.status_code != 200:
                continue

            data = r.json()

            if isinstance(data, dict):
                events.append(data)
            elif isinstance(data, list):
                events.extend([e for e in data if isinstance(e, dict)])

        except:
            continue

    return events

data = fetch()

# =========================
# SAFE PARSE
# =========================
def safe(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            try:
                return ast.literal_eval(x)
            except:
                return []
    return []

def to_float_list(x):
    out = []
    for i in safe(x):
        try:
            out.append(float(i))
        except:
            continue
    return out

# =========================
# FEATURES
# =========================
def momentum(p): return np.mean(np.diff(p)) if len(p) > 1 else 0
def volatility(p): return np.std(p)

def entropy(p):
    p = np.clip(np.array(p), 1e-6, 1)
    p = p / np.sum(p)
    return -np.sum(p * np.log(p + 1e-9))

def dominance(p): return np.max(p) - np.mean(p)
def imbalance(p): return np.max(p) - np.min(p)

# =========================
# REGIME
# =========================
def market_regime(prices):
    p = np.array(prices)
    return np.std(p) + abs(np.mean(p) - 0.5)

# =========================
# BAYES / CONFIDENCE
# =========================
def confidence_score(ent, vol):
    base = 1 / (1 + ent + vol)
    return np.clip(base, 0.1, 1.0)

def bayesian_shrinkage(model_p, market_p, confidence):
    return confidence * model_p + (1 - confidence) * market_p

# =========================
# KELLY (VARIANCE-AWARE)
# =========================
def variance_kelly(p, q, signal_var):
    if p <= q:
        return 0

    k = (p - q) / (1 - q + 1e-6)
    risk_adj = 1 / (1 + 5 * signal_var)

    k = k * risk_adj
    return max(0, min(k, 0.05))

# =========================
# MODEL (LOGIT + REGIME)
# =========================
def model_prob(market_prob, mom, vol, ent, dom, imb, prices):

    regime = market_regime(prices)

    skew_adj = np.tanh(2 * (0.5 - abs(market_prob - 0.5)))

    mom_n = np.tanh(mom)
    dom_n = np.tanh(dom)
    imb_n = np.tanh(imb)
    ent_n = 1 - np.tanh(ent)
    vol_n = 1 - np.tanh(vol)

    raw = (
        0.25 * mom_n +
        0.20 * dom_n +
        0.20 * imb_n +
        0.20 * ent_n +
        0.15 * vol_n
    )

    raw = raw / (1 + regime)

    logit = np.log((market_prob + 1e-6) / (1 - market_prob + 1e-6))
    logit = logit + 0.8 * raw

    prob = 1 / (1 + np.exp(-logit))
    prob = prob * skew_adj + market_prob * (1 - skew_adj)

    return np.clip(prob, 1e-4, 1 - 1e-4)

# =========================
# ARBITRAGE DETECTION
# =========================
def detect_arbitrage(df):
    if df.empty:
        return []

    grouped = df.groupby("candidate")["model_prob"].mean()
    mean = grouped.mean()

    out = []
    for c, p in grouped.items():
        dev = p - mean
        if abs(dev) > 0.15:
            out.append({"candidate": c, "arb_signal": dev})

    return out

# =========================
# BUILD DATA
# =========================
rows = []

for event in data:
    if not isinstance(event, dict):
        continue

    for market in event.get("markets", []):
        outcomes = safe(market.get("outcomes"))
        prices = to_float_list(market.get("outcomePrices"))

        if len(outcomes) == 0 or len(prices) == 0:
            continue

        mom = momentum(prices)
        vol = volatility(prices)
        ent = entropy(prices)
        dom = dominance(prices)
        imb = imbalance(prices)

        for o, p in zip(outcomes, prices):

            m_prob = float(p)

            mod_prob = model_prob(m_prob, mom, vol, ent, dom, imb, prices)

            conf = confidence_score(ent, vol)

            mod_prob = bayesian_shrinkage(mod_prob, m_prob, conf)

            signal_var = np.var([mom, vol, ent, dom, imb])

            edge = mod_prob - m_prob

            if edge < MIN_EDGE:
                continue

            kelly = variance_kelly(mod_prob, m_prob, signal_var)

            bet = BANKROLL * kelly * KELLY_FRACTION
            bet = min(bet, BANKROLL * MAX_BET_FRAC)

            if bet < MIN_BET:
                continue

            rows.append({
                "candidate": o,
                "market_prob": m_prob,
                "model_prob": mod_prob,
                "edge": edge,
                "bet": round(bet, 2)
            })

df = pd.DataFrame(rows)

# =========================
# EMPTY CASE
# =========================
if df.empty:
    st.warning("No valid bets")
    st.stop()

df = df.sort_values("edge", ascending=False)

# =========================
# DASHBOARD
# =========================
st.title("🇵🇪 Peru Election Quant Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Max Edge", f"{df['edge'].max():.4f}")
col2.metric("Avg Edge", f"{df['edge'].mean():.4f}")
col3.metric("Exposure", f"{df['bet'].sum():.2f} $")

st.subheader("Opportunities")
st.dataframe(df, use_container_width=True)

st.plotly_chart(px.bar(df, x="candidate", y="edge", color="edge"))

st.plotly_chart(px.scatter(
    df,
    x="market_prob",
    y="model_prob",
    size="bet",
    color="edge",
    hover_name="candidate"
))

st.plotly_chart(px.histogram(df, x="edge", nbins=20))

# =========================
# ARBITRAGE PANEL
# =========================
arb = detect_arbitrage(df)
if arb:
    st.subheader("⚡ Arbitrage Signals")
    st.dataframe(pd.DataFrame(arb))