import pandas as pd

# Load the dataset
data = pd.read_csv('california_housing_test (1).csv')

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Calculate basic statistics
print(data.describe())

# If there are missing values, handle them (e.g., fill with mean)
data.fillna(data.mean(), inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot to visualize potential linear relationships
sns.pairplot(data)
plt.show()

# Display a heatmap to identify correlations
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

from sklearn.model_selection import train_test_split

# Define features and target variable
X = data.drop('medv', axis=1)
y = data['medv']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display metrics
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.show()

# Example with a single feature
X_single = data[['rm']]  # 'rm' is the average number of rooms per dwelling
y_single = data['medv']

# Split the data
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(X_single, y_single, test_size=0.2, random_state=42)

# Train the model
model_single = LinearRegression()
model_single.fit(X_train_single, y_train_single)

# Make predictions
y_pred_single = model_single.predict(X_test_single)

# Plot the regression line
plt.scatter(X_test_single, y_test_single, color='blue')
plt.plot(X_test_single, y_pred_single, color='red')
plt.xlabel("Average Number of Rooms")
plt.ylabel("Median Value of Homes")
plt.title("Regression Line")
plt.show()
