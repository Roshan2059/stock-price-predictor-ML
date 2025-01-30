# import streamlit as st
# import pandas as pd
# import numpy as np
# from keras.models import load_model
# import matplotlib.pyplot as plt
# import yfinance as yf

# st.title("stock Price Predictor Web App")

# stock = st.text_input('Enter the stock:', "GOOG")

# from datetime import datetime
# end = datetime.now()
# start = datetime(end.year - 10, end.month, end.day)

# google_data = yf.download(stock, start, end)

# model = load_model('latest_stock_price_predictor_model.keras')
# st.subheader('Stock Data')
# st.write(google_data)

# splitting_len = int(len(google_data)*0.7)
# x_test = pd.DataFrame(google_data.Close[splitting_len:])

# def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(values, 'Orange')
#     plt.plot(full_data.Close, 'b')
#     if extra_data:
#         plt.plot(extra_dataset)
#     return fig

# st.subheader('Original Close Price and MA for 250 days')
# google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data))

# st.subheader('Original Close Price and MA for 200 days')
# google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data))

# st.subheader('Original Close Price and MA for 100 days')
# google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data))

# st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
# # google_data['MA_for_100_days'] = google_data.Close.rolling(250).mean()
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(x_test[['Close']])

# x_data = []
# y_data = []

# for i in range(100, len(scaled_data)):
#     x_data.append(scaled_data[i-100:i])
#     y_data.append(scaled_data[i])
    
# x_data, y_data = np.array(x_data), np.array(y_data)

# predections = model.predict(x_data)

# inv_pre = scaler.inverse_transform(predections)
# inv_y_test = scaler.inverse_transform(y_data)

# plotting_data = pd.DataFrame(
#     {
#         'original_test_data': inv_y_test.reshape(-1),
#         'predictions': inv_pre.reshape(-1)
#     },
#     index = google_data.index[splitting_len+100:]
# )
# st.subheader('Original values vs Predicted values')
# st.write(plotting_data)

# st.subheader('Original Close Price vs Predicted Close Price')
# fig = plt.figure(figsize=(15,6))
# plt.plot(pd.concat([google_data.Close[:splitting_len+100], plotting_data], axis=0))
# plt.legend(["Data- not used", "Original Test Data", "predicted Test Data"])
# st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import joblib

# Title
st.title("Stock Price Predictor Web App")

# User Inputs
stock = st.text_input("Enter the stock symbol (e.g., GOOG):", "GOOG")
future_date = st.date_input("Select a date (up to 7 days from today):", datetime.now().date())
model_choice = st.radio("Select the model:", ["LSTM", "Linear Regression"])
predict_button = st.button("Predict")

# Load models and scaler
lstm_model = load_model("saved_models/pretrained_stock_model.h5")
linear_model = joblib.load("saved_models/linear_regression_model.pkl")
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.min_ = np.load("saved_models/scaler_minmax.npy")
scaler.scale_ = np.load("saved_models/scaler_scale.npy")

# Fetch stock data
st.subheader("Fetching Stock Data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 10)  # Last 10 years
data = yf.download(stock, start=start_date, end=end_date)

if data.empty:
    st.error("Invalid stock symbol or no data available.")
else:
    # Display historical data
    st.subheader("Stock Data (In $)")
    st.write(data)

    # Prepare test data for predictions
    train_size = int(len(data) * 0.7)
    test_data = data["Close"][train_size:].values
    scaled_test_data = scaler.transform(test_data.reshape(-1, 1))

    
    # Predict future prices
    if predict_button:
        st.subheader("Predicted Prices")
        future_days = min((future_date - datetime.now().date()).days, 7)

        if model_choice == "LSTM":
            # LSTM Predictions
            last_window = data["Close"].values[-100:]
            future_predictions = []

            for _ in range(future_days):
                scaled_window = scaler.transform(last_window.reshape(-1, 1))
                scaled_window = scaled_window.reshape(1, 100, 1)
                future_pred = lstm_model.predict(scaled_window)[0, 0]
                next_price = scaler.inverse_transform([[future_pred]])[0, 0]
                future_predictions.append(next_price)
                last_window = np.append(last_window[1:], next_price)

        elif model_choice == "Linear Regression":
            # Linear Regression Predictions
            x_future = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
            lr_predictions = linear_model.predict(x_future)
            future_predictions = scaler.inverse_transform(lr_predictions).flatten()

        # Display predictions
        pred_dates = [datetime.now().date() + timedelta(days=i) for i in range(1, future_days + 1)]
        future_df = pd.DataFrame({"Date": pred_dates, "Predicted Close Price": future_predictions})
        st.write(future_df)

        # Plot future predictions
        # st.subheader("Future Predictions")
        plt.figure(figsize=(15, 6))
        plt.plot(pred_dates, future_predictions, label="Future Predictions", color="green")
        plt.legend()
    

# Future Predictions Plot
if predict_button:
    st.subheader("Future Predictions")
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(pred_dates, future_predictions, label="Future Predictions", color="green")
    ax2.set_title("Future Predictions for Stock Prices")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Predicted Close Price")
    ax2.legend()
    st.pyplot(fig2)

# Historical Data Visualization
st.subheader("Original Close Price and Moving Averages (MAs)")
fig1, ax1 = plt.subplots(figsize=(15, 6))
data['MA_250'] = data['Close'].rolling(250).mean()
data['MA_200'] = data['Close'].rolling(200).mean()
data['MA_100'] = data['Close'].rolling(100).mean()
ax1.plot(data['Close'], label="Close Price", color="blue")
ax1.plot(data['MA_250'], label="MA for 250 Days", color="orange")
ax1.plot(data['MA_200'], label="MA for 200 Days", color="green")
ax1.plot(data['MA_100'], label="MA for 100 Days", color="red")
ax1.set_title("Close Price with Moving Averages")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)




