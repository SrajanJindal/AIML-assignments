import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('california_housing_test.csv')

# Initial Analysis
print("Dataset Info:\n", df.info())
print("Summary Statistics:\n", df.describe())

# Visualizations
sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Handling missing values
df.fillna(df.mean(), inplace=True)

# Checking for outliers
sns.boxplot(data=df)
plt.show()
