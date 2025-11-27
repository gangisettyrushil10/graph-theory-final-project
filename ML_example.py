import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the data 
url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
df = pd.read_csv(url)  

# columns: TV, radio, newspaper, sales
print("First 5 rows of the dataset:")
print(df.head())


# input is TV, radio, and newspaper, target is sales
X = df[['TV', 'radio', 'newspaper']]  
y = df['sales'] 

# Standardizing 
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size=0.2, random_state=42)

# train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on test set
y_pred = model.predict(X_test)

# plot actual values vs. predicted values 
plt.figure(figsize=(10, 6))

# actual sales as a scatter plot
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual Sales', alpha=0.7)

# add a line of best fit
m, b = np.polyfit(y_test, y_pred, 1)
plt.plot(y_test, m*y_test + b, color='red', label='Best Fit Line', linewidth=2)

# add a 45-degree line for reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'g--', label='Perfect Prediction Line')

plt.title('Actual vs Predicted Sales (Advertising Dataset)')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.legend()
plt.show()

# evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

