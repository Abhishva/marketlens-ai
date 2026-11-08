
import streamlit as st
import pandas as pd
import yfinance as yf
from stock_agent.data import get_stock_data
from stock_agent.features import add_features
from stock_agent.model import train_model, predict_next_day

st.set_page_config(page_title="MarketLens AI", page_icon="📈", layout="wide")

st.title("📈 MarketLens AI")
st.caption("AI-powered stock market analysis for Indian investors")

st.warning(
    "This tool provides experimental market analysis, not guaranteed "
    "predictions or financial advice."
)

market = st.selectbox("Choose market", ["NSE", "BSE"])
symbol = st.text_input("Enter stock symbol", "RELIANCE").strip().upper()
period = st.selectbox("Historical data", ["1y", "2y", "5y"], index=1)

if st.button("Analyze Stock", type="primary"):
    if not symbol:
        st.error("Please enter a stock symbol.")
    else:
        ticker = symbol + (".NS" if market == "NSE" else ".BO")

        with st.spinner("Analyzing market data..."):
            try:
                data = get_stock_data(ticker, period)

                if data is None or data.empty or len(data) < 100:
                    st.error("Not enough historical data for this stock.")
                else:
                    data = add_features(data)
                    model, accuracy = train_model(data)
                    prediction = predict_next_day(model, data)

                    latest = data.iloc[-1]
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Latest Close", f"₹{latest['Close']:.2f}")
                    col2.metric("RSI", f"{latest['RSI']:.2f}")
                    col3.metric("Test Accuracy", f"{accuracy:.1f}%")

                    st.subheader("AI Prediction")
                    st.metric(
                        "Next-session direction",
                        prediction["direction"],
                        f"Estimated probability: {prediction['confidence']:.1f}%"
                    )

                    st.subheader("Price History")
                    st.line_chart(data["Close"])

                    st.subheader("Recent Market Data")
                    st.dataframe(data.tail(10), use_container_width=True)

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.info("Check the stock symbol and try again.")
