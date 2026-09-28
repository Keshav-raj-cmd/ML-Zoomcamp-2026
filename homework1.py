import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

print(pd.__version__)

print(df.shape)

print(df.head())

#finding all the unique values in the column
print(f"Number of unique fuel types: {df['fuel_type'].nunique()}")

print(df.isnull().sum())

max_fuel_efficiency_asia = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
print(f"Maximum fuel efficiency for cars from Asia: {max_fuel_efficiency_asia} MPG")

print(f"The median of horsepower is: {df['horsepower'].median()}")

print(f"The mode of horsepower is: {df['horsepower'].mode()[0]}")

#finding the value count of all the values in horsepower column
print(df['horsepower'].value_counts())

df['horsepower']=df['horsepower'].fillna(252.0)

print(df.isnull().sum())

print(f"The new median of horsepower is: {df['horsepower'].median()}")

#Selecting all cars from Asia
df_asia=df[df['origin']=='Asia']

print(df_asia)

df_asia=df_asia[['vehicle_weight','model_year']]
print(df_asia)

df_asia=df_asia[:7]

print(df_asia)

X=df_asia.to_numpy()

# Calculate the transpose of X and XTX
XT = X.T
XTX = np.dot(XT, X)
print(f"XTX:\n{XTX}")

# Invert XTX
XTX_inverse = np.linalg.inv(XTX)
print(f"XTX inverse:\n{XTX_inverse}")

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = np.dot(np.dot(XTX_inverse, XT), y)
print(f"The weights are: {w}")

sum_w = np.sum(w)
print(f"The sum of the weights is: {sum_w}")

