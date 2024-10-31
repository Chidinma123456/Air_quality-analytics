import pandas as pd  # Importing pandas for data manipulation
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for data visualization
import seaborn as sns  # Importing seaborn for enhanced data visualization

# Load the dataset into a pandas DataFrame
data = pd.read_csv('airquality.csv')

# Display the first few rows of the dataset to understand the structure
print("Data Head:\n", data.head())

# Check for missing values in the dataset
print("\nMissing Values:\n", data.isna().sum())


# Impute missing values with the mean for numeric columns
data['Ozone'] = data['Ozone'].fillna(data['Ozone'].mean())
data['Solar.R'] = data['Solar.R'].fillna(data['Solar.R'].mean())
data['Wind'] = data['Wind'].fillna(data['Wind'].mean())

# Verify that there are no more missing values
print("\nMissing Values After Imputation:\n", data.isna().sum())


# Define a function to remove outliers using the Interquartile Range (IQR) method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile Range (IQR)
    lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
    upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from 'Ozone', 'Solar.R', and 'Wind' columns
data_clean = remove_outliers(data, 'Ozone')
data_clean = remove_outliers(data_clean, 'Solar.R')
data_clean = remove_outliers(data_clean, 'Wind')

# Display the summary of the cleaned data
print("\nSummary of Cleaned Data:\n", data_clean.describe())


#descriptive statistics for the cleaned data
summary_stats = data_clean.describe()

# Print the summary statistics
print("\nDescriptive Statistics:\n", summary_stats)


#normalize the data using Min-Max Scaling
data_normalized = (data_clean - data_clean.min()) / (data_clean.max() - data_clean.min())

#display the first few rows of the normalized data
print("\nNormalized Data:\n", data_normalized.head())


# Pairplot to visualize relationships between variables
sns.pairplot(data_clean)
plt.show()

# Boxplot to visualize the distribution and detect any remaining outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=data_clean)
plt.title('Boxplot of Cleaned Data')
plt.show()


from scipy import stats

# Perform a t-test to compare the means of Ozone levels before and after cleaning
t_stat, p_value = stats.ttest_ind(data['Ozone'].dropna(), data_clean['Ozone'])

# Print the results of the t-test
print("\nT-Test Results:\n", "t-statistic:", t_stat, "p-value:", p_value)


# Final summary of findings
print("\nConclusion:\n")
print("The air quality data has been successfully cleaned, and descriptive statistics have been calculated.")
print("Data normalization and visualization provided additional insights into the distribution and relationships between variables.")
print("The t-test indicates whether the cleaning process significantly altered the mean values, providing a basis for further decision-making.")
print("The cleaned data is now ready for use in machine learning models or further statistical analysis.")