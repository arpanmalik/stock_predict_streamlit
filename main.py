# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


st.title("Stock price prediction")

user_input = st.text_input("Enter Stock Symbol for coming 30 days trend", "AAPL")
st.write("Stock prediction for ", user_input)

path = 'stock.csv.csv'

if user_input == 'apple' or user_input == 'AAPL':
  path = 'stock.csv.csv'

if user_input == 'INFY' or user_input=='infosys':
  path = 'stock3.csv'

if user_input == "TSLA" or user_input=="tesla":
  path = 'stock2.csv'

if user_input == "WIT" or user_input=='wipro':
  path = 'stock4.csv'

if user_input == "google":
  path = 'stock5.csv'

if user_input == 'amazon' or user_input=='AMZN':
  path = 'stock6.csv'

if user_input == 'microsoft' or user_input=='MSFT':
  path = 'stock7.csv'

if user_input == 'UBER' or user_input=='uber':
  path = 'stock8.csv'

df = pd.read_csv(path, low_memory=False)
st.subheader("Data from 2010-01-01 to 2019-12-31")
st.write(df.describe())

st.subheader("CLosing Price vs Time CHart")
ma100 = df.close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.close)
st.pyplot(fig)

st.subheader("CLosing Price vs Time CHart & 100 Days Moving Average")
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.close)
st.pyplot(fig)

st.subheader("CLosing Price vs Time CHart & 200 Days Moving Average")
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.close)
st.pyplot(fig)

st.subheader("CLosing Price vs Time CHart & moving 100 days vs moving 200 days")
st.write("Green Colour for 100 moving days Average")
st.write("Red colour for 200 moving days Average")
ma100 = df.close.rolling(100).mean()
ma200 = df.close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df.close,)
st.pyplot(fig)

st.write("Note:- Graph of moving 100 above moving 200 is the indicator of upward trend of the market")
st.write("Note:- Graph of moving 100 below moving 200 is the indicator of downward trend of the market")


data_train = pd.DataFrame(df['close'][0:int(len(df)*0.7)])
data_test = pd.DataFrame(df['close'][int(len(df)*0.7):int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))

data_train_arr = scaler.fit_transform(data_train)



model = load_model('keras_model (2).h5')
past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)

input_data = scaler.fit_transform(final_df)

#st.write(final_df.head(50))
#st.write(final_df.tail(50))

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)



#st.write(x_test)
y_predicted = model.predict(x_test)



scal = scaler.scale_
#st.write(scal[0])
scale_factor = 1/scal[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
#st.write(y_predicted)
#st.write(y_test)

st.subheader("Original Trend")
figi = plt.figure(figsize=(12,6))
plt.plot(y_test, 'r', label="Original Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(figi)

st.subheader("Predicted Trend")
figi = plt.figure(figsize=(12,6))
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(figi)

st.subheader("Predicted Trend VS Original Trend")
figi = plt.figure(figsize=(12,6))
plt.plot(y_test, 'g', label="Original Price")
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(figi)


# model = Sequential()
# model.add(LSTM(units=50, activation="relu", return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=60, activation="relu", return_sequences=True))
# model.add(Dropout(0.3))
# model.add(LSTM(units=80, activation="relu", return_sequences=True))
# model.add(Dropout(0.4))
# model.add(LSTM(units=120, activation="relu"))
# model.add(Dropout(0.5))
#
# model.add(Dense(units=1))
#
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.fit(x_train, y_train, epochs=30)
#
# past_100_days = data_train.tail(100)
# final_df = past_100_days.append(data_test, ignore_index=True)
# input_data = scaler.fit_transform(final_df)
#
# x_test = []
# y_test = []
# for i in range(100, input_data.shape[0]):
#   x_test.append(input_data[i-100:i])
#   y_test.append(input_data[i,0])
#
# x_test, y_test = np.array(x_test), np.array(y_test)
# y_predicted = model.predict(x_test)


# plt.figure(figsize=(12,6))
# plt.plot(y_test, 'b', label="Original Price")
# plt.plot(y_predicted, 'r', label="Predicted Price")
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.legend()
# plt.show()