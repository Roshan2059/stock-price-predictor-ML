import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("stock Price Predictor Web App")

stock = st.text_input('Enter the stock:', "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year - 10, end.month, end.day)

google_data = yf.download(stock, start, end)

model = load_model('latest_stock_price_predictor_model.keras')
st.subheader('Stock Data')
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    return fig

