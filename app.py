# Prave Stocks — BUY/HOLD/SELL using TA + news sentiment
# For learning only (not financial advice).

import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Try FinBERT (Transformers). If not installed, fall back to VADER (lightweight).
FINBERT_OK = False
try:
    from transformers import pipeline  # will only work if transformers+torch are installed
    FINBERT_OK = True
except Exception:
    FINBERT_OK = False

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Prave Stocks", page_icon="📈", layout="wide")
st.title("📈 Prave Stocks")
st.caption("Demo: technicals + news sentiment → simple BUY / HOLD / SELL")

# --------------------------
# Helpers
# --------------------------
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, period_days: int = 240):
    end = datetime.utcnow()
    start = end - timedelta(days=period_days)
    df = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    df = df.dropna()
    return df

def add_indicators(df: pd.DataFrame):
    out = df.copy()
    # SMAs
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    out["SMA200"] = out["Close"].rolling(200).mean()
    # RSI
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out["RSI"] = 100 - (100 / (1 + rs))
    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]
    return out

@st.cache_data(show_spinner=False)
def load_news(ticker: str, limit: int = 15):
    """Headlines via yfinance; empty dataframe if none."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []
    rows = []
    for n in items[:limit]:
        title = n.get("title", "")
        link = n.get("link", "")
        pub = n.get("publisher", "")
        ts = n.get("providerPublishTime", None)
        dt = datetime.utcfromtimestamp(ts) if ts else None
        rows.append({"title": title, "link": link, "publisher": pub, "time": dt})
    return pd.DataFrame(rows)

@st.cache_resource(show_spinner=True)
def load_sentiment():
    """Return ('finbert' or 'vader', model_obj)."""
    if FINBERT_OK:
        try:
            clf = pipeline("text-classification", model="ProsusAI/finbert", truncation=True)
            return "finbert", clf
        except Exception:
            pass
    return "vader", SentimentIntensityAnalyzer()

def score_headlines(df_news: pd.DataFrame, mode: str, model):
    if df_news.empty:
        return df_news.assign(label=[], score=[], signed=[])
    titles = df_news["title"].tolist()
    if mode == "finbert":
        preds = model(titles)
        labels = [p["label"] for p in preds]
        scores = [float(p["score"]) for p in preds]
        signed = []
        for lab, sc in zip(labels, scores):
            if lab.upper() == "POSITIVE":
                signed.append(+sc)
            elif lab.upper() == "NEGATIVE":
                signed.append(-sc)
            else:
                signed.append(0.0)
    else:  # VADER
        labels, scores, signed = [], [], []
        for t in titles:
            s = model.polarity_scores(t)["compound"]  # [-1..+1]
            signed.append(float(s))
            if s >= 0.05:
                labels.append("POSITIVE"); scores.append(float(s))
            elif s <= -0.05:
                labels.append("NEGATIVE"); scores.append(float(abs(s)))
            else:
                labels.append("NEUTRAL"); scores.append(0.0)
    out = df_news.copy()
    out["label"] = labels
    out["score"] = scores
    out["signed"] = signed
    return out

def decide(ta_row: pd.Series, news_score: float):
    score = 0
    reasons = []
    # RSI
    rsi = ta_row["RSI"]
    if pd.notna(rsi):
        if rsi < 30:
            score += 1; reasons.append(f"RSI {rsi:.1f} (oversold)")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI {rsi:.1f} (overbought)")
    # MACD
    macd = ta_row["MACD"]; sig = ta_row["MACD_SIGNAL"]
    if pd.notna(macd) and pd.notna(sig):
        if macd > sig:
            score += 1; reasons.append("MACD > Signal (bullish momentum)")
        else:
            score -= 1; reasons.append("MACD < Signal (bearish momentum)")
    # Trend filter
    close = ta_row["Close"]; sma200 = ta_row["SMA200"]
    if pd.notna(sma200):
        if close > sma200:
            score += 1; reasons.append("Price above 200-day SMA (uptrend)")
        else:
            score -= 1; reasons.append("Price below 200-day SMA (downtrend)")
    # News
    if news_score >= 0.10:
        score += 1; reasons.append(f"News sentiment {news_score:+.2f} (bullish)")
    elif news_score <= -0.10:
        score -= 1; reasons.append(f"News sentiment {news_score:+.2f} (bearish)")
    else:
        reasons.append(f"News sentiment {news_score:+.2f} (neutral)")
    # Final call
    if score >= 2:
        call = "BUY"
    elif score <= -2:
        call = "SELL"
    else:
        call = "HOLD"
    return call, score, reasons

def make_chart(df: pd.DataFrame, ticker: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    for col, name in [("SMA20","SMA20"),("SMA50","SMA50"),("SMA200","SMA200")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=name), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], mode="lines", name="Signal"), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], name="Hist"), row=2, col=1)
    fig.update_layout(title=f"{ticker} — price & indicators", xaxis_rangeslider_visible=False, height=720)
    return fig

# --------------------------
# UI
# --------------------------
left, right = st.columns([1, 2])
with left:
    st.subheader("Choose a ticker")
    ticker = st.text_input("Symbol", value="AAPL").strip().upper()
    days = st.slider("History (days)", min_value=120, max_value=720, value=240, step=30)
    run_btn = st.button("Analyze")
with right:
    st.info("This is a demo. Do your own research. Not financial advice.", icon="⚠️")

if run_btn:
    try:
        df0 = load_prices(ticker, period_days=days)
        if df0.empty:
            st.error("Couldn’t load prices. Try another ticker.")
        else:
            df = add_indicators(df0)
            news = load_news(ticker, limit=15)

            mode, model = load_sentiment()
            news_scored = score_headlines(news, mode, model)
            news_avg = float(news_scored["signed"].mean()) if not news_scored.empty else 0.0

            latest = df.iloc[-1]
            call, score, reasons = decide(latest, news_avg)

            st.subheader(f"Call: **{call}** (score {score:+d})")
            st.write("**Why:**")
            for r in reasons:
                st.write("• " + r)

            st.plotly_chart(make_chart(df, ticker), use_container_width=True)

            st.subheader("Latest headlines")
            if news_scored.empty:
                st.write("No headlines found.")
            else:
                show = news_scored.copy()
                if show["time"].notna().any():
                    show["time"] = show["time"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(show[["time","publisher","title","label","score","link"]], use_container_width=True)
    except Exception as ex:
        st.exception(ex)
