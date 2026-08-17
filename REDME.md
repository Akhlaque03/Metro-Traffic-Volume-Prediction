# 🚦 Metro Traffic Volume Prediction

An end-to-end machine learning project for predicting **hourly metro traffic volume** using weather conditions, time-based features, and calendar information.

The project includes data preprocessing, feature engineering, regression model comparison, hyperparameter tuning, feature importance analysis, and deployment using Streamlit.

### 🚀 Live Demo

[**Metro Traffic Volume Prediction — Live App**](https://akhlaque03-metro-traffic-prediction.streamlit.app/)

# ---

## 📌 Project Overview

Metro traffic volume varies significantly with changes in weather conditions, time of day, weekdays, weekends, and holidays.

This project uses machine learning regression techniques to predict **hourly traffic volume** based on traffic-related, weather, and calendar features.

Multiple regression models were trained and evaluated, followed by hyperparameter tuning of selected models to identify the best-performing model for deployment.

The final **Tuned LightGBM Regressor** achieved an **R² Score of 0.9820** and was deployed as an interactive Streamlit web application.

# ---

## 🎯 Objective

The main objective of this project is to develop an accurate machine learning model for predicting hourly metro traffic volume and deploy it as an interactive web application.

### Key Objectives

* Analyze factors influencing metro traffic volume
* Perform data preprocessing and feature engineering
* Train and compare multiple regression models
* Evaluate models using MAE, MSE, RMSE, R², and Adjusted R²
* Perform hyperparameter tuning on selected models
* Identify the most important features affecting predictions
* Select and deploy the best-performing model
* Provide an interactive interface for real-time traffic volume prediction


# ---

## 🔄 Machine Learning Workflow

```text
Data Collection
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
Categorical Feature Encoding
      ↓
Train-Test Split
      ↓
Baseline Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Final Model Selection
      ↓
Feature Importance Analysis
      ↓
Model Deployment
      ↓
Streamlit Web Application
```
