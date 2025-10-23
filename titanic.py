import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb 

sb.set(style="whitegrid")

# Reading CSV file
df = pd.read_csv("titanic (4).csv")

# Basic exploration
print(df.head())
print(df.info())
print(df.isnull().sum())

# Handle missing values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

print(df.isnull().sum())
print(df.describe())
print(df['Pclass'].unique())
print(df['Embarked'].value_counts())

# Create FamilySize column
df["FamilySize"] = df['Parch'] + df['SibSp']

# Create DataFrame for passengers older than 30
df30 = df[df['Age'] > 30]

# Sort by Fare descending
sort = df.sort_values(by='Fare', ascending=False)

# Survival rate by Pclass
survival_rate = df.groupby('Pclass')['Survived'].mean()
print(survival_rate)

# Bar chart for survival rate
plt.bar(survival_rate.index, survival_rate, color='skyblue')
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate")
plt.show()

# Passenger(s) with highest Fare
print(df[df['Fare'] == df["Fare"].max()])

# Count of survivors and non-survivors
sb.countplot(x='Survived', data=df)
plt.title("Count of Survivors vs Non-Survivors")
plt.show()

# Histogram of Ages
plt.hist(df["Age"], bins=20, color='green', edgecolor='black')
plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Count plot for passenger class
sb.countplot(x='Pclass', data=df, palette='Set2')
plt.title("Number of Passengers in Each Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# Box plot for Fare distribution
sb.boxplot(y=df["Fare"], color='skyblue')
plt.title("Box Plot of Passenger Fares")
plt.ylabel("Fare")
plt.show()

# Scatter plot for Age vs Pclass
plt.scatter(df["Age"], df["Pclass"], color='green')
plt.title("Passenger Age vs Class")
plt.xlabel("Age")
plt.ylabel("Pclass")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# conclusion of the titanic dataset 
print("""
🧾 Conclusion:
1. Higher classes (Pclass 1) had higher survival rates.
2. Fare distribution shows a few high-value outliers from 1st class passengers.
3. Most passengers were from 3rd class.
4. Older passengers are more common in higher classes.
5. Data cleaning improved the dataset by handling missing values in Age and Embarked columns.
""")