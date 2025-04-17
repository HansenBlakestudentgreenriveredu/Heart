import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Style for plots
sns.set(style="whitegrid")

# Load the dataset
file_path = os.path.join("Archive", "2022", "heart_2022_with_nans.csv")
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip()
print("Available columns:", df.columns.tolist())

# Focus only on the columns you're interested in
columns_of_interest = [
    'State', 'Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
    'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth',
    'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan',
    'RaceEthnicityCategory', 'AgeCategory', 'HeightInMeters', 'WeightInKilograms',
    'BMI', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos'
]

# Keep only those columns in the dataframe
df = df[[col for col in columns_of_interest if col in df.columns]]

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# --- CATEGORICAL VARIABLES (Bar Plots) ---
categorical_cols = ['Sex', 'GeneralHealth', 'LastCheckupTime', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory']
for col in categorical_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# --- BINARY VARIABLES (Pie Charts) ---
binary_cols = [
    'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer',
    'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
    'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',
    'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'ChestScan', 'AlcoholDrinkers', 'CovidPos'
]

for col in binary_cols:
    if col in df.columns:
        plt.figure(figsize=(6, 6))
        df[col].value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightblue', 'salmon']
        )
        plt.title(f'{col} Distribution')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()

# --- NUMERIC VARIABLES (Histograms) ---
numeric_cols = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'HeightInMeters', 'WeightInKilograms', 'BMI']
for col in numeric_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), bins=30, kde=True, color='skyblue')
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

# --- CORRELATION HEATMAP ---
numeric_df = df[numeric_cols].copy()
numeric_df = numeric_df.dropna()
if not numeric_df.empty:
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.show()

# --- BOX PLOT: BMI vs GeneralHealth ---
if 'BMI' in df.columns and 'GeneralHealth' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='GeneralHealth', y='BMI', data=df)
    plt.title('BMI by General Health')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Bar Plot: CovidPos by AgeCategory ---
if 'CovidPos' in df.columns and 'AgeCategory' in df.columns:
    plt.figure(figsize=(8, 5))
    df.groupby('AgeCategory')['CovidPos'].value_counts(normalize=True).unstack().plot(
        kind='bar', stacked=True
    )
    plt.title('COVID-19 Positive Rate by Age Category')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Optional pause for terminal
input("✅ EDA complete! Press Enter to exit...")