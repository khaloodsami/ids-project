import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset
file_path = r"C:\Users\Administrator\Downloads\archive\aac_shelter_outcomes.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)
print("Model saved in directory:", os.getcwd())

# Step 2: Inspect the dataset
print("First 5 rows of the dataset:")
print(df.head())  # Display the first few rows

print("\nDataset Information:")
print(df.info())  # Data types and non-null counts

print("\nSummary Statistics:")
print(df.describe())  # Summary statistics for numerical columns

# Step 3: Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Handle missing values
df['age_upon_outcome'] = df['age_upon_outcome'].fillna('Unknown')
df['outcome_subtype'] = df['outcome_subtype'].fillna('Unknown')
df['sex_upon_outcome'] = df['sex_upon_outcome'].fillna('Unknown')
df['name'] = df['name'].fillna('Unknown')

# Step 4: Convert 'age_upon_outcome' to numeric
def convert_to_years(age_str):
    if pd.isnull(age_str) or not isinstance(age_str, str) or not age_str.strip():
        return np.nan  # Return NaN for missing or empty values
    
    # Extract numeric part from the age string
    num = ''.join([char for char in age_str if char.isdigit()])
    if not num:
        return np.nan  # Return NaN if no digits are found
    
    num = float(num)
    
    # Convert based on unit found in the string
    if 'week' in age_str.lower():
        return num / 52  # Convert weeks to years
    elif 'month' in age_str.lower():
        return num / 12  # Convert months to years
    else:
        return num  # Assume it's already in years if no unit is found

df['Age Numeric'] = df['age_upon_outcome'].apply(convert_to_years)

# Step 5: Basic EDA (Exploratory Data Analysis)

# Outcome Type Distribution
print("\nOutcome Type Distribution:")
print(df['outcome_type'].value_counts())
sns.countplot(x='outcome_type', data=df)
plt.title("Distribution of Outcome Types")
plt.xticks(rotation=45)
plt.show()

# Animal Type Distribution
print("\nAnimal Type Distribution:")
print(df['animal_type'].value_counts())
sns.countplot(x='animal_type', hue='outcome_type', data=df)
plt.title("Outcome Types by Animal Type")
plt.show()

# Step 6: Summary Statistics After Fixing
print("\nSummary Statistics After Fixing:")
print(df.describe())
print("\nOutcome Type Counts:")
print(df['outcome_type'].value_counts())
print("\nAnimal Type Counts:")
print(df['animal_type'].value_counts())

# Step 7: Summary Statistics for 'Age Numeric'
print("\nSummary Statistics for 'Age Numeric':")
print(df['Age Numeric'].describe())

# Plot Age Distribution
sns.histplot(df['Age Numeric'], kde=True)
plt.title("Age Distribution")
plt.show()

# Boxplot for Age by Outcome Type
sns.boxplot(x='outcome_type', y='Age Numeric', data=df)
plt.title("Age by Outcome Type")
plt.xticks(rotation=45)
plt.show()
#correlation matrix
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# Step 8: Time-Based Trend Analysis

# Ensure 'datetime' column exists and convert to datetime format
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')  # Handle invalid date formats
if df['datetime'].isnull().sum() > 0:
    print(f"\n{df['datetime'].isnull().sum()} rows have invalid datetime values and are set to NaT.")
# Extract year, month, and day from 'datetime' and drop the column
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df = df.drop(columns=['datetime'])

# Analyze outcomes across months or years
monthly_outcomes = df.groupby(['year', 'month'])['outcome_type'].value_counts().unstack().fillna(0)

# Plot trends over time (monthly)
monthly_outcomes.plot(kind='line', figsize=(10, 6))
plt.title("Adoption Outcomes Over Time (by Month)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Step 9: Grouped Analysis

# Group by animal_type and outcome_type to find insights
grouped_data = df.groupby(['animal_type', 'outcome_type']).size().reset_index(name='count')

# Visualize the distribution of outcomes by animal type
sns.countplot(x='animal_type', hue='outcome_type', data=df)
plt.title("Outcome Types by Animal Type")
plt.xticks(rotation=45)
plt.show()

# Step 10: Machine Learning - Random Forest Model

# Preprocess the dataset for modeling
# Drop columns that do not provide useful features for modeling
columns_to_drop = ['age_upon_outcome', 'animal_id', 'name', 'MonthYear', 'date_of_birth']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Convert categorical columns to dummy variables
categorical_columns = ['animal_type', 'sex_upon_outcome', 'breed', 'color', 'outcome_subtype']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Fill missing values:
# For numeric columns, fill with the mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# For categorical columns, fill with the mode (most frequent value)
for col in df.select_dtypes(include=[object]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Ensure the target column is separate
X = df.drop(columns=['outcome_type'])  # Drop 'outcome_type' column from features
y = df['outcome_type']  # Set 'outcome_type' as target
X = X.apply(pd.to_numeric, errors='coerce')
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Save the trained model
import joblib
joblib.dump(model, "random_forest_model.pkl")
print("\nModel saved as 'random_forest_model.pkl'.")

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy}")

# Display Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 11: Correlation Matrix and Histogram
# Correlation of numeric features
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# Step 12: Export the cleaned dataset
df.to_csv("cleaned_animal_outcomes.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_animal_outcomes.csv'.")
