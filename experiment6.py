import pandas as pd
from scipy import stats

# Load your dataset
df = pd.read_csv('california_housing_test (1).csv')

# Handling missing values
# Fill missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Handling outliers
# Calculate z-scores for numerical columns
z_scores = stats.zscore(df.select_dtypes(include=[float, int]))

# Get absolute z-scores
abs_z_scores = abs(z_scores)

# Filter entries where all z-scores are less than 3 (i.e., within 3 standard deviations)
filtered_entries = (abs_z_scores < 3).all(axis=1)

# Apply the filter to the dataframe
df = df[filtered_entries]

# Display the cleaned dataset
print("Data after handling missing values and outliers:\n", df)

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_dataset.csv', index=False)
