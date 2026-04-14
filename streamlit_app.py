import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import json
import ast
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Peru Quant Dashboard v5", layout="wide")

REFRESH = 10000
BANKROLL = st.sidebar.number_input("Bankroll ($)", 10, 1_000_000, 90)
KELLY_FRACTION = st.sidebar.slider("Kelly fraction", 0.0, 1.0, 0.25)

st_autorefresh(interval=REFRESH, key="refresh")

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
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, dict):
                    events.append(data)
                elif isinstance(data, list):
                    events.extend(data)
        except:
            continue
    return events

data = fetch()

def parse_maybe_json(x):
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

def momentum(p): return np.mean(np.diff(p)) if len(p) > 1 else 0
def volatility(p): return np.tanh(np.std(p))
def gamma(p): return np.mean(np.diff(np.diff(p))) if len(p) > 2 else 0
def delta(p): return p[-1] - p[0] if len(p) > 1 else 0

rows = []

for event in data:
    markets = event.get("markets", [])

    for m in markets:

        outcomes = parse_maybe_json(m.get("outcomes"))
        prices = parse_maybe_json(m.get("outcomePrices"))

        prices = [float(p) for p in prices if str(p).replace('.', '', 1).isdigit()]

        if len(outcomes) == 0 or len(prices) == 0:
            continue

        mom = momentum(prices)
        vol = volatility(prices)
        gam = gamma(prices)
        delt = delta(prices)

        for o, p in zip(outcomes, prices):

            mkt = float(p)

            model = mkt + 0.1 * mom + 0.05 * gam + 0.05 * delt
            model = np.clip(model, 0, 1)

            edge = model - mkt

            rows.append({
                "candidate": o,
                "market_prob": mkt,
                "model_prob": model,
                "edge": edge,
                "gamma": gam,
                "delta": delt,
                "momentum": mom
            })

df = pd.DataFrame(rows)

st.write("Events:", len(data))
st.write("Signals:", len(df))

if df.empty:
    st.warning("Still empty → API structure or parsing mismatch")
    st.json(data[0]["markets"][0])
    st.stop()

df = df.sort_values("edge", ascending=False)

st.title("Peru Quant Dashboard v5")

st.dataframe(df, use_container_width=True)

st.plotly_chart(px.bar(df.head(20), x="candidate", y="edge"))