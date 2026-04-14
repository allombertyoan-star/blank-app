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
st.set_page_config(page_title="Peru Quant Dashboard v2", layout="wide")

REFRESH = 10000
BANKROLL = st.sidebar.number_input("Bankroll ($)", 10, 1_000_000, 90)
KELLY_FRACTION = st.sidebar.slider("Kelly fraction", 0.0, 1.0, 0.25)

MIN_BET = 1
st_autorefresh(interval=REFRESH, key="refresh")

# =========================
# GAMMA API
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
# SAFE PARSER
# =========================
def safe(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except:
        return []

def to_float_list(x):
    x = safe(x)
    out = []
    for i in x:
        try:
            out.append(float(i))
        except:
            continue
    return out

# =========================
# FEATURES (robustes)
# =========================
def momentum(p):
    p = np.array(p)
    return np.mean(np.diff(p)) if len(p) > 1 else 0

def volatility(p):
    p = np.std(p)

    # regime correction
    return np.tanh(p)

def gamma_feature(p):
    p = np.array(p)
    if len(p) < 3:
        return 0
    return np.mean(np.diff(np.diff(p)))

def delta_feature(p):
    p = np.array(p)
    return p[-1] - p[0] if len(p) > 1 else 0

def entropy(p):
    p = np.clip(np.array(p), 1e-6, 1-1e-6)
    return -np.mean(p*np.log(p) + (1-p)*np.log(1-p))

def imbalance(p):
    p = np.array(p)
    return np.max(p) - np.min(p)

# =========================
# MODEL (RESIDUAL + REGIME)
# =========================
def model_delta(features):
    mom, vol, ent, dom, imb, gam, delt, mkt = features

    signal = (
        0.20 * mom +
        0.15 * gam +
        0.15 * delt +
        0.15 * dom +
        0.10 * imb +
        -0.20 * vol +
        -0.10 * ent +
        0.10 * (0.5 - abs(mkt - 0.5))
    )

    return 0.5 * np.tanh(signal)

# =========================
# KELLY (stabilisé)
# =========================
def kelly(p, q):
    edge = p - q
    var = q * (1 - q) + 1e-6
    k = edge / var
    return max(0, min(k * 0.4, 0.2))

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
        prices_raw = market.get("outcomePrices", [])

        prices = to_float_list(prices_raw)

        if len(prices) < 3:
            continue

        # FEATURES
        mom = momentum(prices)
        vol = volatility(prices)
        gam = gamma_feature(prices)
        delt = delta_feature(prices)
        ent = entropy(prices)
        imb = imbalance(prices)

        for o, p in zip(outcomes, prices):

            mkt = float(p)

            # soft clamp (avoid useless extremes only)
            if mkt < 0.01 or mkt > 0.99:
                continue

            features = (
                mom,
                vol,
                ent,
                imb,
                imb,
                gam,
                delt,
                mkt
            )

            delta = model_delta(features)

            # ANCHOR MARKET (important fix)
            model_prob = np.clip(mkt + delta, 0, 1)

            edge = model_prob - mkt

            # IMPORTANT: no hard filtering → ranking system
            k = kelly(model_prob, mkt)

            bet = BANKROLL * k * KELLY_FRACTION
            bet = max(bet, 0)

            rows.append({
                "candidate": o,
                "market_prob": mkt,
                "model_prob": model_prob,
                "edge": edge,
                "gamma": gam,
                "delta": delt,
                "momentum": mom,
                "volatility": vol,
                "entropy": ent,
                "imbalance": imb,
                "kelly": k,
                "bet": round(bet, 2)
            })

df = pd.DataFrame(rows)

# =========================
# DEBUG SAFE MODE
# =========================
st.write("Events:", len(data))
st.write("Signals:", len(df))

if df.empty:
    st.warning("No data from API or parsing issue (check Gamma response structure)")
    st.stop()

# =========================
# RANKING SYSTEM (IMPORTANT CHANGE)
# =========================
df = df.sort_values("edge", ascending=False)

# =========================
# DASHBOARD
# =========================
st.title("🇵🇪 Peru Quant Dashboard — Gamma Enhanced Model")

c1, c2, c3 = st.columns(3)

c1.metric("Max Edge", f"{df['edge'].max():.4f}")
c2.metric("Avg Gamma", f"{df['gamma'].mean():.4f}")
c3.metric("Total Exposure", f"{df['bet'].sum():.2f}$")

st.subheader("Top Opportunities")
st.dataframe(df, use_container_width=True)

fig = px.bar(df.head(20), x="candidate", y="edge", color="edge")
st.plotly_chart(fig)

fig2 = px.scatter(df, x="market_prob", y="model_prob", size="bet", color="edge")
st.plotly_chart(fig2)

fig3 = px.histogram(df, x="edge", nbins=30)
st.plotly_chart(fig3)