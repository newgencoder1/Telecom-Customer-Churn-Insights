# Medical Insurance Cost Prediction using Linear Regression
# Author: Manas Gupta

# Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('insurance.csv')

# Data Collection & Basic Inspection
print("First 5 rows:")
print(insurance_dataset.head())
print("\nDataset Shape:", insurance_dataset.shape)
print("\nMissing Values:", insurance_dataset.isnull().sum())
print("\nStatistical Measures:")
print(insurance_dataset.describe())

# Data Pre-Processing: Encoding categorical features
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Splitting Features and Target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Splitting data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(f"\nTraining data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

# Model Training (Linear Regression)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Model Evaluation
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared value (Training):', r2_train)

test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared value (Test):', r2_test)

# Building a Predictive System Example
input_data = (31, 1, 25.74, 0, 1, 0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = regressor.predict(input_data_reshaped)
print(f'\nThe predicted medical insurance cost is USD {prediction[0]:.2f}')