# Code_Alpha_Stock_Prediction
Batch:  July: Data Science : M1

![Stock Prediction using LSTM](https://github.com/Soumyabk3/Code_Alpha_Stock_Prediction/blob/main/Stock%20Prediction.png)

## Problem Statement

Build a machine learning model to forecast future product sales based on factors such as advertising spending, target audience segmentation, and advertising platform choice.


## Introduction

This project utilizes Long Short-Term Memory (LSTM) networks to predict stock prices based on historical data. It involves preprocessing data, training an LSTM model, and visualizing the results.

## Dependencies

To run this project, you need to install the following Python libraries:

- numPy
- pandas
- matplotlib
- seaborn
- scikit-learn
- IPython
- plotly
- Warnings
- keras
- yfinance
- pandas_datareader

## Tools and techniques
- Language: Python

- Algorithm: LSTM

- Integrated Development Environment : Jupyter Notebook

## Installation

To install the specified packages (numpy, pandas, matplotlib, seaborn, scikit-learn, and ipython), etc, follow these steps:


1. *Installing Dependencies*:

```bash
  pip install pandas numpy matplotlib seaborn scikit-learn keras yfinance pandas-datareader plotly

```
 
2. *To look at the dependencies type the command on the cmd*:

```bash
pip list

```


3. * you can use the pip freeze command to get a list of all installed packages along with their versions*:

```bash
pip freeze

```

```bash
ipython==8.4.0
matplotlib==3.5.2
numpy==1.22.4
pandas==1.4.2
scikit-learn==1.1.1
seaborn==0.11.2
```

## Summary

- Data Collection:

Obtain historical stock price data from a reliable source such as Yahoo Finance.
Modify the data fetching code in the provided notebooks (stock_prediction.ipynb) to retrieve stock data.

- Data Preprocessing:

Clean and preprocess the data to handle missing values and format it for LSTM input.
- LSTM Model Training:

Train an LSTM model using the preprocessed historical stock price data.
Configure the LSTM architecture with appropriate input shape, number of layers, and neurons per layer.
- Prediction and Evaluation:

Use the trained LSTM model to make predictions on test data.
Evaluate the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and visual inspection of predicted vs. actual prices.
- Visualization:

Visualize the historical stock prices, predicted prices, and evaluation metrics using matplotlib, seaborn, and plotly.

## Conclusion
In this project, we successfully applied Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. 
In a nutshell, this project showcases the application of LSTM neural networks for stock price prediction, highlighting their capability to handle time series data and learn complex dependencies over extended periods. The insights gained contribute to the field of financial forecasting, offering a foundation for future enhancements and applications in predictive analytics.
## Authors

- [@Soumyabk3](https://github.com/Soumyabk3)

