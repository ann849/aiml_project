import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split #data splitting

from sklearn.linear_model import LinearRegression #building regression model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #evaluating model performance


df = pd.read_csv("ro.csv")   #Load the dataset

print("Dataset loaded successfully!")
print(df.head())


print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())


X = df.drop("Price ($)", axis=1)  
y = df["Price ($)"]                

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(   #training and testing the data
    X, y, test_size=0.2, random_state=42
)

print("\nTraining and testing data split done!")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

#to make the predictions
y_pred = model.predict(X_test)

print("\nSample Predictions:")
print("Predicted Prices:", y_pred[:5])
print("Actual Prices:", y_test[:5].values)

#evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)


print("\nModel training and evaluation completed successfully!")



