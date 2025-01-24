import streamlit as st
import tensorflow as tf
import numpy as np

# Load the trained model
try:
    model = tf.keras.models.load_model("new_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Title for the web app
st.title("ANN Prediction App")

# Input fields for x1 and x2
x1 = st.number_input("Enter x1 (numeric value):", value=0.0, step=0.1, format="%.2f")
x2 = st.number_input("Enter x2 (numeric value):", value=0.0, step=0.1, format="%.2f")

# Predict button
if st.button("Predict"):
    if 'model' in locals():
        # Prepare the input data for the model
        input_data = np.array([[x1, x2]])
        try:
            prediction = model.predict(input_data)
            # Convert prediction to binary output (0 or 1)
            output = 1 if prediction[0][0] > 0.5 else 0
            # Display the result
            st.success(f"The predicted output is: {output}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Model not loaded. Please ensure the file 'ann_model.h5' is in the correct location.")
