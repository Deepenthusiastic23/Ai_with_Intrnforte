import streamlit as st
import numpy as np
from model import train

# Title
st.title("Simple AI Regression App")
st.subheader("Predict y = 2x + 3")

# Train model
model = train()

# Sidebar input
st.sidebar.header("Input")
x_value = st.sidebar.slider("Select value of x", 1, 10, 1)

# Convert input to array
input_array = np.array([[x_value]])

# Prediction
prediction = model.predict(input_array)

# Output
st.write(f"### Input (x): {x_value}")
st.write(f"### Predicted Output (y): {prediction[0]:.2f}")