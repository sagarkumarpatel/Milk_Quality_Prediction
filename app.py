import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE # Import SMOTE even if not directly used in prediction, needed for context if you retrain

# Load the trained model, scaler, and PCA
try:
    rf_pca_top = joblib.load("rf_pca_top.pkl")
    scaler_top = joblib.load("scaler_top.pkl")
    pca_top = joblib.load("pca_top.pkl")
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'rf_pca_top_for_export.pkl', 'scaler_top_for_export.pkl', and 'pca_top_for_export.pkl' are in the same directory.")
    st.stop() # Stop the app if model files are not found


top_features = ["pH", "Temperature", "Fat", "Odor", "Taste", "Colour", "Turbidity"]

# Recommended ranges based on dataset
feature_ranges = {
    "pH": (3.0, 10.0),
    "Temperature": (30, 75),
    "Fat": (0, 5),
    "Odor": (0, 1),
    "Taste": (0, 1),
    "Colour": (200, 255), # Adjusted range based on dataset description
    "Turbidity": (0, 1)
}

# Streamlit App Title and Description
st.title("Milk Quality Prediction")
st.write("Enter the milk features below to predict its quality.")

# Create input fields for user to enter feature values
user_input = {}
st.sidebar.header("Enter Milk Features")

for feature in top_features:
    low, high = feature_ranges[feature]
    # Ensure consistent types for slider arguments by casting to float
    user_input[feature] = st.sidebar.slider(f"{feature} ({low}-{high})", float(low), float(high), float((low + high) / 2)) # Use slider for numerical input


# Create a button to trigger prediction
if st.sidebar.button("Predict Milk Quality"):
    # Convert user input to DataFrame
    user_data_df = pd.DataFrame([user_input])

    # Ensure the order of columns matches the training data
    user_data_df = user_data_df[top_features]

    # Scale the input
    user_data_scaled = scaler_top.transform(user_data_df)

    # Apply PCA transformation
    user_data_pca = pca_top.transform(user_data_scaled)

    # Predict class and probabilities
    prediction = rf_pca_top.predict(user_data_pca)[0]
    prediction_proba = rf_pca_top.predict_proba(user_data_pca)[0]

    # Map numeric prediction to label
    label_map = {0: "Low", 1: "Medium", 2: "High"}
    predicted_quality = label_map[prediction]

    # Display prediction
    st.subheader("Prediction:")
    st.success(f"Predicted Milk Quality: **{predicted_quality}**")

    # Show prediction probabilities
    st.subheader("Prediction Probabilities:")
    for i, label in label_map.items():
        st.write(f"{label}: {prediction_proba[i]*100:.2f}%")

st.write("---")
st.write("Note: This prediction is based on a model trained on the provided dataset.")