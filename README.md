Sales Forecast Prediction
Project Overview

This project is a Machine Learning model that predicts future sales based on past sales and advertising data.
It uses Linear Regression to understand the relationship between advertising/media features (like TV, Radio, Newspaper, etc.) and product sales (Fan, AC, Mobile, Laptop, etc.).

The project also includes:
Manual data entry for predictions.
Bulk predictions by uploading CSV/Excel files.
History tracking of previous predictions.
Comparison of past records.
Visualizations of inputs vs. predicted sales.

Dataset
The dataset used contains advertising/media expenses and sales data for multiple products:
Features: TV, Radio, Newspaper, Oven, Fan, AC, Projector, Heater, Table Fan, Mobile Phone, Laptop, Tab
Target: Sales.

Model:
Algorithm: Linear Regression
Libraries Used:
pandas
scikit-learn
matplotlib
joblib
The model is trained, saved as .pkl, and later loaded for making predictions.

Results & Features:
Predict sales for custom values.
Bulk predictions using CSV/Excel.
Store prediction history with date/time.
Compare previous records using graphs.
Visual charts (bar graphs, line graphs) for insights.