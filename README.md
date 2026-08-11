# Praveen-stocks

A lightweight Streamlit prototype for transparent stock analysis using technical indicators and headline sentiment.

This project predates TickerLume and served as an early experimentation ground for combining market signals with explainable BUY / HOLD / SELL outputs.

## What it does

- Downloads historical market data with `yfinance`
- Computes SMA20, SMA50, SMA200, RSI, MACD, and MACD histogram
- Scores recent headlines using FinBERT when available, with VADER as a lightweight fallback
- Combines technical and sentiment signals into a simple rule-based BUY / HOLD / SELL output
- Displays candlestick, moving-average, and MACD visualizations with Plotly
- Shows the reasons behind each signal instead of presenting a black-box recommendation

## Why it matters

The core product principle behind this prototype is explainability: a user should be able to see which signals contributed to a result. That principle later evolved into TickerLume, where deterministic product logic owns ratings and numeric outputs while AI is used to explain the supplied evidence.

## Stack

`Python` · `Streamlit` · `pandas` · `NumPy` · `yfinance` · `Plotly` · `FinBERT` · `VADER`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Related Product

**TickerLume** — AI-powered market intelligence that turns live signals into explainable investment insight.

https://tickerlume.com

## Disclaimer

This project is for research, learning, and product experimentation only. It is not financial advice.
