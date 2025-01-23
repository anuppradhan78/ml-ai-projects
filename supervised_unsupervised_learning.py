import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Linear Regression is used to predict a continuous outcome variable based on one or more predictor variables. Here, we generate some data, split it into training and testing sets, train a model, and visualize the results.

# Generate some data
X = np.array([i for i in range(10)]).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(10, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
