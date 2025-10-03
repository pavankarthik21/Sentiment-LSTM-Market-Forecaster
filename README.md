
# Sentiment LSTM Market Forecaster (HackUTD)

HackUTD project. Predict short-term market movement by combining FinancialBERT sentiment with LSTM on price history.

## Demo
<img width="1600" height="800" alt="image" src="https://github.com/user-attachments/assets/088b04c9-7c7c-48c0-b6ef-d9c6fe0ad38b" />


## How to run (simple)
1) Make sure Python is installed.
2) Run: `python main.py`

## Why this is interesting
Most models use only price. Adding news sentiment makes the signal less noisy.


## Project Overview
Objective: To build a prediction model that leverages both numerical stock data and sentiment analysis to provide more accurate forecasts.
Data Sources: Historical stock data from financial institutions like Goldman Sachs and JP Morgan, combined with news articles for sentiment extraction.
Technologies Used: FinancialBERT for sentiment analysis, LSTM models for time series prediction, and FastAPI for deploying an interactive user interface.

This project was developed as part of the HackUTD hackathon to address the challenge of improving stock price prediction accuracy by integrating sentiment analysis with traditional financial data. The aim was to enhance forecasting reliability by combining historical stock prices with public sentiment from news articles.
