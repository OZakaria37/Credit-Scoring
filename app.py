# app.py
import streamlit as st
import polars as pl
import numpy as np
import joblib

# Load model and preprocessing
model = joblib.load("credit_model.pkl")
pipeline = joblib.load("credit_pipeline.pkl")
reversed_score_map = {0: "Poor", 1: "Standard", 2: "Good"}

st.title("Credit Score Prediction")
st.write("Upload a CSV file with customer data to get predictions.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    input_df = pl.read_csv(uploaded_file, infer_schema_length=100000)

    st.write("Input Data Preview:")
    st.dataframe(input_df.head())

    if "Credit_Score" in input_df.columns:
        input_df.drop("Credit_Score")

    features = pipeline.transform(input_df)
    df_id = features["ID"]
    features = features.drop("ID")

    # Predict
    predictions = model.predict(features)
    # Decoding
    Decode = np.vectorize(lambda x: reversed_score_map.get(x, None))
    predictions = Decode(predictions)

    # Append predictions to the original dataframe
    output_df = input_df.with_columns(pl.Series("Predicted_Credit_Score", predictions))
    output_df = output_df.select("ID", predictions)
    st.write("Predictions:")
    st.dataframe(output_df)

    # Optional: allow download
    csv = output_df.write_csv()
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="credit_score_predictions.csv",
        mime="text/csv",
    )
