import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Optional but helpful

heart_data_2020 = pd.read_csv("./Archive/2020/heart_2020_cleaned.csv")
heart_data_2022_nan = pd.read_csv("./Archive/2022/heart_2022_with_nans.csv")
heart_data_2022 = pd.read_csv("./Archive/2022/heart_2022_no_nans.csv")

print(heart_data_2022_nan.shape)
print(heart_data_2022_nan.head())
print(heart_data_2022_nan.isna().sum())

heart_data_2022_nan_hist = heart_data_2022_nan["State"].sum().hist()

heart_data_2022_nan_hist.show()