# 🇵🇰 Pakistan Inflation Survival Dashboard (AI Predictor)

## Overview
The **Pakistan Inflation Survival Dashboard** is a data-driven Python application designed to help citizens navigate the economic pressures of inflation. Using historical proxy data representing the Pakistan Bureau of Statistics (PBS) and State Bank of Pakistan (SBP), this tool forecasts the future prices of essential commodities (Wheat, Sugar, Petrol) and utilizes Machine Learning to offer personalized budgeting advice.

## Core Features & ML Integration
- **AI Price Forecasting (Time Series):** Utilizes Scikit-Learn's `LinearRegression` to analyze 12 months of historical commodity prices and project the trendline for the next 3-6 months.
- **Budget Survival AI (KMeans Clustering):** Uses unsupervised machine learning (`KMeans`) to cluster a user's monthly budget into distinct risk profiles compared to a synthetic dataset of Pakistani households, generating tailored survival tips.
- **Interactive Data Visualization:** Implements `plotly.express` for responsive time-series charts and city-wide comparative heatmaps.
- **Automated Data Management:** If the primary dataset is missing, the system catches the error and generates realistic, randomized proxy data using `numpy`.

## How to Run the Application
1. **Prerequisites:** Ensure Python is installed.
2. **Install Required Libraries:** ```bash
   pip install streamlit pandas numpy scikit-learn plotly