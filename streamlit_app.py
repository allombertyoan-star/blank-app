import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import ast
from streamlit_autorefresh import st_autorefresh

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Peru Quant Dashboard", layout="wide")

REFRESH = 10000
BANKROLL = st.sidebar.number_input("Bankroll ($)", 10, 1_000_000, 90)
KELLY_FRACTION = st.sidebar.slider("Kelly fraction", 0.0, 1.0, 0.25)

MIN_BET = 1
MAX_BET_FRAC = 0.10
MIN_EDGE = 0.02

st_autorefresh(interval=REFRESH, key="refresh")

# =========================
# DATA FETCH
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
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return []

# =========================
# FEATURES (IMPROVED)
# =========================
def momentum(p):
    p = np.array(p)
    return np.mean(np.diff(p)) if len(p) > 1 else 0

def volatility(p):
    return np.std(p)

def dominance(p):
    p = np.array(p)
    return np.max(p) - np.mean(p)

def imbalance(p):
    p = np.array(p)
    return np.max(p) - np.min(p)

def entropy(p):
    p = np.clip(np.array(p), 1e-6, 1-1e-6)
    return -np.mean(p*np.log(p) + (1-p)*np.log(1-p))

# =========================
# NORMALISATION
# =========================
def z(x):
    x = np.array(x)
    return (x - np.mean(x)) / (np.std(x) + 1e-9)

# =========================
# MODEL (RESIDUAL CORRECTION)
# =========================
def model_delta(features):
    mom, vol, ent, dom, imb, mkt = features

    signal = (
        0.25 * mom +
        -0.20 * vol +
        -0.15 * ent +
        0.20 * dom +
        0.20 * imb +
        0.10 * (0.5 - abs(mkt - 0.5))
    )

    return 0.5 * np.tanh(signal)

# =========================
# KELLY STABILISÉ
# =========================
def kelly(p, q):
    edge = p - q
    var = q * (1 - q) + 1e-6
    k = edge / var
    return max(0, min(k * 0.3, 0.05))

# =========================
# BUILD DATASET
# =========================
rows = []

for event in data:
    if not isinstance(event, dict):
        continue

    markets = event.get("markets", [])
    if not isinstance(markets, list):
        continue

    for market in markets:

        outcomes = safe(market.get("outcomes", []))
        prices = safe(market.get("outcomePrices", []))

        try:
            prices = [float(p) for p in prices]
        except:
            continue

        if len(prices) < 3:
            continue

        mom = momentum(prices)
        vol = volatility(prices)
        ent = entropy(prices)
        dom = dominance(prices)
        imb = imbalance(prices)

        for o, p in zip(outcomes, prices):

            mkt = float(p)

            # FILTER MARKET EXTREMES
            if mkt < 0.05 or mkt > 0.95:
                continue

            features = (
                mom,
                vol,
                ent,
                dom,
                imb,
                mkt
            )

            delta = model_delta(features)
            model_prob = np.clip(mkt + delta, 0, 1)

            edge = model_prob - mkt

            if edge < MIN_EDGE:
                continue

            k = kelly(model_prob, mkt)
            bet = BANKROLL * k * KELLY_FRACTION
            bet = min(bet, BANKROLL * MAX_BET_FRAC)

            if bet < MIN_BET:
                continue

            rows.append({
                "candidate": o,
                "market_prob": mkt,
                "model_prob": model_prob,
                "edge": edge,
                "momentum": mom,
                "volatility": vol,
                "entropy": ent,
                "dominance": dom,
                "imbalance": imb,
                "kelly": k,
                "bet": round(bet, 2)
            })

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No valid bets")
    st.stop()

df = df.sort_values("edge", ascending=False)

# =========================
# DASHBOARD
# =========================
st.title("🇵🇪 Peru Quant Dashboard (Optimized Model)")

c1, c2, c3 = st.columns(3)

c1.metric("Max Edge", f"{df['edge'].max():.4f}")
c2.metric("Avg Entropy", f"{df['entropy'].mean():.4f}")
c3.metric("Exposure", f"{df['bet'].sum():.2f}$")

st.subheader("Opportunities")
st.dataframe(df, use_container_width=True)

fig = px.bar(df, x="candidate", y="edge", color="edge")
st.plotly_chart(fig)

fig2 = px.scatter(df, x="market_prob", y="model_prob", size="bet", color="edge")
st.plotly_chart(fig2)

fig3 = px.histogram(df, x="edge", nbins=20)
st.plotly_chart(fig3)