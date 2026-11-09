# MarketLens AI

A Streamlit-based experimental stock-analysis dashboard for Indian NSE and BSE tickers.

## Features
- Downloads daily historical prices using `yfinance`.
- Calculates RSI, moving averages, MACD, returns, volatility, and volume ratio.
- Trains a Random Forest classifier using a chronological train/test split.
- Displays the model's next-session directional estimate and historical holdout accuracy.

## Important limitations
This is an educational prototype, not a trading system or financial adviser. The model can be wrong; historical test accuracy does not guarantee future performance. Yahoo Finance data through `yfinance` is an unofficial data interface and may be delayed, incomplete, or unavailable. Do not make trades solely from this output.

## Run locally (Windows)
1. Install Python 3.10 or newer from https://www.python.org/downloads/.
2. Extract the project ZIP.
3. Open the extracted `marketlens_ai_project` folder.
4. In File Explorer, click the address bar, type `cmd`, and press Enter.
5. Run:

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the local dashboard. If it does not, open the `Local URL` printed in the terminal (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud
1. Upload the contents of this folder to the root of your GitHub repository.
2. Keep `app.py` and `requirements.txt` at the repository root and keep the `stock_agent` folder intact.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select `Abhishva/marketlens-ai`, branch `main`, and main file path `app.py`.
5. Click **Deploy**.

## Project structure

```text
marketlens_ai_project/
├── app.py
├── requirements.txt
├── README.md
└── stock_agent/
    ├── __init__.py
    ├── data.py
    ├── features.py
    └── model.py
```
