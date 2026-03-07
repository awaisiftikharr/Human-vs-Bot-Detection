import streamlit as st
import joblib
import pandas as pd

# Load your trained AI brain
model = joblib.load('bot_detector.pkl')

st.title("🤖 AI Human vs. Bot Detector")
st.write("Enter movement data to see if the user is a Real Human or a Robot.")

# User Input Sliders
dist = st.slider("Movement Distance (px)", 100, 500, 300)
time = st.slider("Time Taken (seconds)", 1.0, 20.0, 5.0)
jitter = st.slider("Movement Jitter (shakiness)", 0.0, 10.0, 2.0)

if st.button("Analyze Behavior"):
    # Prepare data for the model
    data = pd.DataFrame([[dist, time, jitter]], columns=['Distance', 'Time', 'Jitter'])
    prediction = model.predict(data)
    
    if prediction[0] == 1:
        st.success("✅ Result: REAL HUMAN")
    else:
        st.error("🚨 Result: ROBOT DETECTED")