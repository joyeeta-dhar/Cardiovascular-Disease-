# Cardiovascular Disease Analysis and Prediction

This project is a machine learning-based web application built with Streamlit to analyze and predict cardiovascular disease risks. It extracts insights from cardiovascular data and provides an interactive platform for exploratory data analysis (EDA) and real-time risk prediction based on user-provided health metrics.

## Features

- **Exploratory Data Analysis (EDA):** Visualizations and insights derived from the dataset to understand various health metrics.
- **Machine Learning Model:** Uses a trained machine learning model (`model.joblib`) for predicting cardiovascular disease likelihood.
- **Interactive Web App:** A premium-styled Streamlit application tailored for health metric inputs and outputs.
- **Prediction Pipeline:** Well-structured model logic and processing pipelines that mimic the original Jupyter Notebook workflow.

## Project Structure

- `app.py`: The main Streamlit web application.
- `model_logic.py`: Machine learning and data preprocessing logic.
- `eda_visuals.py`: Scripts handling data visualization components.
- `requirements.txt`: Python package dependencies.
- `model.joblib`: Pre-trained machine learning model.
- `cardio_data_processed.csv`: The clean data used for analysis and modeling.
- `cardiovascular-disease (1).ipynb`: The baseline Jupyter Notebook documenting the primary data analysis and model training steps.

## How to Run

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
