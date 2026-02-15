import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="RetailPulse Optimizer", layout="wide")

def load_data():
    data = pd.read_csv('retail_sales.csv')
    data['date'] = pd.to_datetime(data['date'])
    return data

def forecast_sales(model, data, scaler, window_size=30):
    last_window = data[-window_size:].reshape(1, window_size, 1)
    prediction = model.predict(last_window)
    return scaler.inverse_transform(prediction)

st.title("📊 RetailPulse: Sales Intelligence")

df = load_data()
values = df['sales'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical Sales Trend")
    st.line_chart(df.set_index('date'))

with col2:
    st.subheader("Inventory Optimization")
    try:
        model = tf.keras.models.load_model('retail_pulse_model.keras')
        next_sales = forecast_sales(model, scaled_data, scaler)
        
        st.metric("Predicted Next Sales", f"{int(next_sales[0][0])} Units")
        
        status = "Optimal" if next_sales < 140 else "Low Stock Warning"
        st.info(f"Inventory Status: {status}")
    except Exception as e:
        st.warning("Model engine is warming up. Please ensure train_model.py has finished.")
        
st.divider()
st.caption("RetailPulse Optimizer | Argiansyah Galih Permata")