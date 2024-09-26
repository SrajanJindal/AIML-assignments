import pandas as pd

# Import data
df = pd.read_csv('california_housing_test.csv')

# Display dataset details
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("First five rows:\n", df.head())
print("Size of dataset:", df.size)
print("Number of missing values:\n", df.isnull().sum())
print("Sum of numerical columns:\n", df.sum(numeric_only=True))
print("Average of numerical columns:\n", df.mean(numeric_only=True))
print("Minimum values:\n", df.min(numeric_only=True))
print("Maximum values:\n", df.max(numeric_only=True))

# Export data
df.to_csv('exported_dataset.csv', index=False)
