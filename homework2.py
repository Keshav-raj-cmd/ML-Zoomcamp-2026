import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------
# 1. Load Data & Filter Columns
# --------------------------------------------------
df_raw = pd.read_csv("/content/car_fuel_efficiency_2026.csv")

# Filter for the requested columns
columns_to_keep = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg",
]
df = df_raw[columns_to_keep].copy()

print(df.head())
print(df)

# --------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# --------------------------------------------------
sns.histplot(df["fuel_efficiency_mpg"], kde=True)
plt.title("Distribution of fuel_efficiency_mpg")
plt.show()

# Question 1: Check missing values
print("Missing values per column:")
print(df.isnull().sum())

print(df.info())
print(df.head())

print("Q1 Missing values:\n", df.isnull().sum())

# Question 2: Median for variable 'horsepower'
print("\nQ2 Horsepower Median:", df["horsepower"].median())


# --------------------------------------------------
# 3. Model Functions & Data Splitting
# --------------------------------------------------
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)

    return w[0], w[1:]


def rmse(y, y_pred):
    error = y_pred - y
    mse = (error**2).mean()
    return np.sqrt(mse)


def split_data(df, seed=42):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_shuffled = df.iloc[idx].copy()

    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train : n_train + n_val].copy()
    df_test = df_shuffled.iloc[n_train + n_val :].copy()

    return df_train, df_val, df_test


# --------------------------------------------------
# 4. Question 3: Comparing Imputation with 0 vs Mean
# --------------------------------------------------
df_train, df_val, df_test = split_data(df, seed=42)

y_train = df_train["fuel_efficiency_mpg"].values
y_val = df_val["fuel_efficiency_mpg"].values

features = ["engine_displacement", "horsepower", "vehicle_weight", "model_year"]

# Option 1: Impute with 0
X_train_0 = df_train[features].fillna(0).values
X_val_0 = df_val[features].fillna(0).values

w_0, w = train_linear_regression(X_train_0, y_train)
y_pred_0 = w_0 + X_val_0.dot(w)
rmse_0 = rmse(y_val, y_pred_0)

# Option 2: Impute with Mean
mean_val = df_train["horsepower"].mean()
X_train_mean = df_train[features].fillna(mean_val).values
X_val_mean = df_val[features].fillna(mean_val).values

w_0, w = train_linear_regression(X_train_mean, y_train)
y_pred_mean = w_0 + X_val_mean.dot(w)
rmse_mean = rmse(y_val, y_pred_mean)

print(f"Q3 - RMSE with 0: {round(rmse_0, 3)}")
print(f"Q3 - RMSE with Mean: {round(rmse_mean, 3)}")

# --------------------------------------------------
# 5. Question 4: Regularization with fillna(0)
# --------------------------------------------------
r_values = [0, 0.01, 0.1, 1, 5, 10, 100]

for r in r_values:
    w_0, w = train_linear_regression_reg(X_train_0, y_train, r=r)
    y_pred = w_0 + X_val_0.dot(w)
    score = rmse(y_val, y_pred)
    print(f"Q4 - r={r:4}: RMSE={round(score, 4)}")

# --------------------------------------------------
# 6. Question 5: Stability Over Different Split Seeds
# --------------------------------------------------
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = []

for s in seeds:
    df_train_s, df_val_s, _ = split_data(df, seed=s)

    y_train_s = df_train_s["fuel_efficiency_mpg"].values
    y_val_s = df_val_s["fuel_efficiency_mpg"].values

    X_train_s = df_train_s[features].fillna(0).values
    X_val_s = df_val_s[features].fillna(0).values

    w_0, w = train_linear_regression(X_train_s, y_train_s)
    y_pred_s = w_0 + X_val_s.dot(w)
    score_s = rmse(y_val_s, y_pred_s)
    scores.append(score_s)

std_dev = np.std(scores)
print(f"Q5 - Standard Deviation of RMSEs: {round(std_dev, 3)}")

# --------------------------------------------------
# 7. Question 6: Train on Train+Val with Seed 9, Eval on Test
# --------------------------------------------------
df_train_9, df_val_9, df_test_9 = split_data(df, seed=9)

df_full_train = pd.concat([df_train_9, df_val_9]).reset_index(drop=True)

y_full_train = df_full_train["fuel_efficiency_mpg"].values
y_test = df_test_9["fuel_efficiency_mpg"].values

X_full_train = df_full_train[features].fillna(0).values
X_test = df_test_9[features].fillna(0).values

w_0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)
y_pred_test = w_0 + X_test.dot(w)
test_rmse = rmse(y_test, y_pred_test)

print(f"Q6 - Test RMSE: {round(test_rmse, 3)}")