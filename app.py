import streamlit as st
from stock_agent.data import get_stock_data
from stock_agent.features import add_features
from stock_agent.model import train_model, predict_next_day

st.set_page_config(page_title="MarketLens AI", page_icon="📈", layout="wide")

st.title("📈 MarketLens AI")
st.caption("Experimental AI-powered stock analysis for Indian NSE and BSE shares")

st.info(
    "Educational research tool only. Predictions are uncertain, may be wrong, "
    "and are not investment advice. Always do your own research."
)

with st.sidebar:
    st.header("Analysis settings")
    market = st.selectbox("Market", ["NSE", "BSE"])
    symbol = st.text_input("Stock symbol", value="RELIANCE").strip().upper()
    period = st.selectbox("Historical period", ["1y", "2y", "5y"], index=1)
    st.caption("Examples: RELIANCE, TCS, INFY, SBIN")

if st.button("Analyze stock", type="primary", use_container_width=True):
    if not symbol:
        st.error("Enter a stock symbol.")
        st.stop()

    ticker = f"{symbol}.NS" if market == "NSE" else f"{symbol}.BO"

    with st.spinner(f"Downloading and analyzing {ticker}..."):
        try:
            raw = get_stock_data(ticker, period)
            if raw is None or raw.empty or len(raw) < 120:
                st.error("Not enough price history was returned. Check the symbol or try another period.")
                st.stop()

            featured = add_features(raw)
            model, accuracy, feature_columns = train_model(featured)
            prediction = predict_next_day(model, featured, feature_columns)

            latest = featured.dropna().iloc[-1]
            a, b, c = st.columns(3)
            a.metric("Latest close", f"₹{latest['Close']:,.2f}")
            b.metric("RSI (14)", f"{latest['RSI']:.1f}")
            c.metric("Historical test accuracy", f"{accuracy:.1f}%")

            st.subheader("Next-session model estimate")
            left, right = st.columns(2)
            left.metric("Model direction", prediction["direction"])
            right.metric("Model probability", f"{prediction['confidence']:.1f}%")
            st.caption(
                "Probability is the model's estimated probability for its selected class, "
                "not a calibrated guarantee of a price move."
            )

            st.subheader("Closing price history")
            st.line_chart(raw["Close"])

            st.subheader("Recent data and indicators")
            st.dataframe(featured.tail(10), use_container_width=True)

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption("If this continues, check the ticker suffix, internet connection, and deployment logs.")
