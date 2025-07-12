import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Nifty 50 stocks (symbol: NSE Yahoo format)
nifty_50 = {
    "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Wipro": "WIPRO.NS",
    "HUL": "HINDUNILVR.NS"
}

st.title("📈 NIFTY 50 Share Price Predictor & Advisor")

stock_name = st.selectbox("Select a NIFTY 50 Stock", list(nifty_50.keys()))
ticker = nifty_50[stock_name]

data = yf.download(ticker, period="6mo")
st.write(f"### Historical Data for {stock_name}", data.tail())

# Feature: Use last N days' closing price to predict next
data['Prediction'] = data['Close'].shift(-1)
df = data[['Close', 'Prediction']].dropna()

X = np.array(df['Close']).reshape(-1, 1)
y = np.array(df['Prediction']).reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)

# Predict next closing price
latest_price = data['Close'].iloc[-1]
predicted_price = model.predict(np.array([latest_price]).reshape(1, -1))[0][0]

st.subheader("📊 Prediction Result")
st.write(f"🔹 Last Closing Price: ₹{latest_price:.2f}")
st.write(f"🔮 Predicted Next Price: ₹{predicted_price:.2f}")

if predicted_price > latest_price:
    st.success("💡 Suggestion: **HOLD** the stock (price expected to rise)")
else:
    st.error("⚠️ Suggestion: **SELL** the stock (price may drop)")
