import streamlit as st
st.title("🎈 My new Streamlit app")
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

# 🔴 bankroll adaptée micro-capital
BANKROLL = st.sidebar.number_input("Bankroll ($)", 10, 1_000_000, 90)
KELLY_FRACTION = st.sidebar.slider("Kelly fraction", 0.0, 1.0, 0.25)

MIN_BET = 1          # minimum réaliste Polymarket
MAX_BET_FRAC = 0.10  # max 10% bankroll par trade
MIN_EDGE = 0.02      # seuil plus réaliste

st_autorefresh(interval=REFRESH, key="refresh")

# =========================
# FETCH (SLUG DIRECT)
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

            # 🔴 FIX structure API
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
# FEATURES (LEVEL 2)
# =========================
def momentum(prices):
    p = np.array(prices)
    return np.mean(np.diff(p)) if len(p) > 1 else 0

def volatility(prices):
    return np.std(prices)

def entropy(prices):
    p = np.array(prices)
    p = p / np.sum(p)
    return -np.sum(p * np.log(p + 1e-9))

def dominance(prices):
    p = np.array(prices)
    return np.max(p) - np.mean(p)

def imbalance(prices):
    p = np.array(prices)
    return np.max(p) - np.min(p)

# =========================
# MODEL (LEVEL 3 SIMPLIFIED)
# =========================
def model_prob(market_prob, mom, vol, ent, dom, imb):

    signal = (
        0.35 * market_prob +
        0.15 * (0.5 + np.tanh(mom)) +
        0.15 * (1 - vol) +
        0.15 * (1 - ent) +
        0.10 * dom +
        0.10 * imb
    )

    # shrinkage bayésien
    return 0.7 * signal + 0.3 * 0.5

# =========================
# SAFE KELLY (CRUCIAL FIX)
# =========================
def safe_kelly(p, q):
    """
    p = model prob
    q = market prob
    """

    if p <= q:
        return 0

    # edge relatif
    k = (p - q) / (1 - q)

    # shrinkage anti-overfitting
    k = k * 0.5

    # cap agressif (très important micro bankroll)
    return max(0, min(k, 0.05))


# =========================
# BUILD DATA
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

        if len(prices) < 2:
            continue

        try:
            prices = [float(p) for p in prices]
        except:
            continue

        mom = momentum(prices)
        vol = volatility(prices)
        ent = entropy(prices)
        dom = dominance(prices)
        imb = imbalance(prices)

        for o, p in zip(outcomes, prices):

            m_prob = p
            mod_prob = model_prob(m_prob, mom, vol, ent, dom, imb)

            edge = mod_prob - m_prob

            # 🔴 filtre qualité
            if edge < MIN_EDGE:
                continue

            kelly = safe_kelly(mod_prob, m_prob)

            bet = BANKROLL * kelly * KELLY_FRACTION

            # 🔴 contraintes bankroll
            bet = min(bet, BANKROLL * MAX_BET_FRAC)

            if bet < MIN_BET:
                continue

            rows.append({
                "candidate": o,
                "market_prob": m_prob,
                "model_prob": mod_prob,
                "edge": edge,
                "momentum": mom,
                "volatility": vol,
                "entropy": ent,
                "dominance": dom,
                "imbalance": imb,
                "kelly": kelly,
                "bet": round(bet, 2)
            })

df = pd.DataFrame(rows)

# =========================
# EMPTY CASE
# =========================
if df.empty:
    st.warning("No valid bets (filters too strict or no edge)")
    st.stop()

df = df.sort_values("edge", ascending=False)

# =========================
# DASHBOARD
# =========================
st.title("🇵🇪 Peru Election Quant Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Max Edge", f"{df['edge'].max():.4f}")
col2.metric("Avg Entropy", f"{df['entropy'].mean():.4f}")
col3.metric("Total Exposure", f"{df['bet'].sum():.2f} $")

# =========================
# TABLE
# =========================
st.subheader("Opportunities")

st.dataframe(df, width="stretch")

# =========================
# EDGE BAR
# =========================
fig = px.bar(
    df,
    x="candidate",
    y="edge",
    color="edge",
    title="Edge per Candidate"
)

st.plotly_chart(fig, key="edge_bar_unique")

# =========================
# SCATTER
# =========================
fig2 = px.scatter(
    df,
    x="market_prob",
    y="model_prob",
    size="bet",
    color="edge",
    hover_name="candidate",
    title="Market vs Model"
)

st.plotly_chart(fig2, key="scatter_unique")

# =========================
# DISTRIBUTION
# =========================
fig3 = px.histogram(
    df,
    x="edge",
    nbins=20,
    title="Edge Distribution"
)

st.plotly_chart(fig3, key="hist_unique")