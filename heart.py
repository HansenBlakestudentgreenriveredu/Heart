import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Optional: for better-looking plots
sns.set(style="whitegrid")

# Load the dataset
file_path = os.path.join("Archive", "2022", "heart_2022_with_nans.csv")
df = pd.read_csv(file_path)

# Preview data
print("First 5 rows of the dataset:")
print(df.head())

# Clean column names
df.columns = df.columns.str.strip()
print("\nAvailable columns:")
print(df.columns.tolist())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# --- 📊 VISUALIZATION SECTION ---

# ✅ Bar Plot - Race
if 'Race' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Race')
    plt.title('Race Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Column 'Race' not found. Skipping bar plot.")

# ✅ Histogram - BMI
if 'BMI' in df.columns:
    plt.figure(figsize=(8, 5))
    df['BMI'].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Column 'BMI' not found. Skipping histogram.")

# ✅ Pie Chart - Heart Disease
if 'HeartDisease' in df.columns:
    plt.figure(figsize=(6, 6))
    df['HeartDisease'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=['lightgreen', 'lightcoral']
    )
    plt.title('Heart Disease Proportion')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Column 'HeartDisease' not found. Skipping pie chart.")

# ✅ Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# ✅ Heart Disease by Sex (Bar Plot)
if {'Sex', 'HeartDisease'}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    df.groupby('Sex')['HeartDisease'].value_counts(normalize=True).unstack().plot(
        kind='bar', stacked=True
    )
    plt.title('Heart Disease by Sex')
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Columns 'Sex' or 'HeartDisease' not found. Skipping bar plot by sex.")

# ✅ Boxplot - BMI vs Heart Disease
if {'BMI', 'HeartDisease'}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='HeartDisease', y='BMI', data=df)
    plt.title('BMI vs Heart Disease')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Columns 'BMI' or 'HeartDisease' not found. Skipping boxplot.")

# ✅ Histogram - BMI by Smoking Status
if {'BMI', 'Smoking'}.issubset(df.columns):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='BMI', hue='Smoking', element='step', kde=True)
    plt.title('BMI by Smoking Status')
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Columns 'BMI' or 'Smoking' not found. Skipping histogram by smoking.")

# Pause if run in terminal
input("✅ Done! Press Enter to exit...")