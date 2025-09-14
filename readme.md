# 🏦 Credit Score Classification

## This project predicts customer credit scores (Poor, Standard, Good) from financial and demographic data using XGBoost and Random Forest. 
## It includes full EDA, feature engineering, model training, and a Streamlit front-end that allows users to upload CSV files and get predictions.
## Based on www.kaggle.com/datasets/parisrohan/credit-score-classification

# ⚙️ Installation

## Install dependencies:

## pip install -r requirements.txt

# 🚀 Usage
## Run Streamlit locally
## streamlit run app.py

## Upload a CSV file (with the same schema as training) to get predictions in a results table.

# 🐳 Docker

## Build the Docker image:

## docker build -t credit_scoring_app .

## Run the container:

## docker run -p 8501:8501 credit_scoring_app

## Now open: http://localhost:8501

# 🧠 Model

## Algorithms tried: XGBoost, Random Forest

## Evaluation Metric: F1-Score (chosen due to class imbalance)

## Best Result: ~0.80 macro-F1 (misclassifications mostly between adjacent classes, e.g. Good ↔ Standard)