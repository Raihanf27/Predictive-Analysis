# -*- coding: utf-8 -*-
"""FIX predictive analysis rumah.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Qp6dxMxpsmb8d_FaaSd_ohMWGn8tyEW4

# Data Loading
"""

# Commented out IPython magic to ensure Python compatibility.
#Import Library yang dibutuhkan

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Membaca data excel

df = pd.read_excel("DATA RUMAH.xlsx")
df

"""# Data Cleaning"""

# Melihat Info dataset

df.info()

# Melihat statistik deskriptif dari dataset
df.describe()

#Drop Kolom NO dan NAMA RUMAH
df = df.drop(columns = ['NO', 'NAMA RUMAH'])

df.head()

harga_column = df.pop('HARGA')
df['HARGA'] = harga_column

df.head()

# Memeriksa kolom apakah ada nilai null

df.isna().sum()

# Memeriksa kolom LB, LT, KT, KM apakah ada nilai null

LB = (df.LB == 0).sum()
LT = (df.LT == 0).sum()
KT = (df.KT == 0).sum()
KM = (df.KM == 0).sum()

print("Nilai 0 di kolom LB ada: ", LB)
print("Nilai 0 di kolom LT ada: ", LT)
print("Nilai 0 di kolom KT ada: ", KT)
print("Nilai 0 di kolom KM ada: ", KM)

"""# Exploratory Data Analysis"""

# Univariate analysis

df.hist(figsize=(10, 8))
plt.show()

#Multivariate Analysis
categorical_cols = ['KT', 'KM', 'GRS']

for col in categorical_cols:
  plt.figure(figsize=(8,6))
  sns.barplot(x=col, y='HARGA', data=df)
  plt.title(f'Rata-rata Harga terhadap {col}')
  plt.xlabel(col)
  plt.ylabel('Rata-rata Harga')
  plt.show()

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(df, diag_kind = 'kde')

# Menghitung matriks korelasi dan menampilkan matriks korelasi menggunakan heatmap
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi')
plt.show()

sns.pairplot(df[['LB', 'LT']], plot_kws={"s": 3});

# Membagi data test dan data training

X = df[['LB','LT','KT','KM','GRS']].values #Feature
y = df['HARGA'].values #Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f'Total of sample in whole dataset: {len(X)}')
print(f'Total of sample in train dataset: {len(X_train)}')
print(f'Total of sample in test dataset: {len(X_test)}')

"""# Modelling

### XGBOOST
"""

# Melatih data dengan XGboost

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Inisialisasi model XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Latih model pada data pelatihan
xgb_model.fit(X_train, y_train)

# Prediksi pada data pelatihan dan pengujian
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Inisialisasi model XGBoost dengan beberapa parameter dasar
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_estimators=100,  # Jumlah pohon
    max_depth=3,       # Kedalaman maksimum pohon
    learning_rate=0.1  # Laju pembelajaran
)

# Definisikan grid parameter untuk pencarian hyperparameter
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [2, 3, 4],
    'n_estimators': [50, 100, 150]
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Gunakan MSE negatif untuk mencari nilai terkecil
    cv=5,  # 5-fold cross-validation
    verbose=1  # Tampilkan progres pencarian
)

# Lakukan pencarian hyperparameter pada data pelatihan
grid_search.fit(X_train, y_train)

# Dapatkan model terbaik dari hasil pencarian
best_xgb_model = grid_search.best_estimator_

# Prediksi pada data pelatihan dan pengujian menggunakan model terbaik
y_train_pred = best_xgb_model.predict(X_train)
y_test_pred = best_xgb_model.predict(X_test)

"""### Random Forest"""

# Melatih data dengan Random Forest

from sklearn.ensemble import RandomForestRegressor

# Inisialisasi model Random Forest
rf_model = RandomForestRegressor(random_state=123)

# Latih model dengan data pelatihan
rf_model.fit(X_train, y_train)

# Prediksi harga rumah untuk data pelatihan dan pengujian
y_train_pred_rf = rf_model.predict(X_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluasi model menggunakan MAE, MSE, dan R-squared
mae_train_rf = mean_absolute_error(y_train, y_train_pred_rf)
mse_train_rf = mean_squared_error(y_train, y_train_pred_rf)
r2_train_rf = r2_score(y_train, y_train_pred_rf)

mae_test_rf = mean_absolute_error(y_test, y_pred_rf)
mse_test_rf = mean_squared_error(y_test, y_pred_rf)
r2_test_rf = r2_score(y_test, y_pred_rf)

"""# Evaluasi Model antara XGboost dan Random Forest"""

# Hasil evaluasi untuk XGBoost dengan hyperparameter tuning
print("Evaluasi Model XGBoost pada Data Pelatihan:")
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("R2 Score:", r2_score(y_train, y_train_pred))

print("\Evaluasi Model XGBoost pada Data Pengujian:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R2 Score:", r2_score(y_test, y_test_pred))

# Hasil evaluasi untuk Random Forest
print("\nEvaluasi Model Random Forest pada Data Pelatihan:")
print(f'MAE: {mae_train_rf}')
print(f'MSE: {mse_train_rf}')
print(f'R2 Score: {r2_train_rf}')

print("\nEvaluasi Model Random Forest pada Data Pengujian:")
print(f'MAE: {mae_test_rf}')
print(f'MSE: {mse_test_rf}')
print(f'R2 Score: {r2_test_rf}')

# Membandingkan y_true dengan prediksi XGBoost dan Random Forest
df_comparison = pd.DataFrame({
    'y_true': y_test,
    'XGBoost Prediction': y_test_pred,
    'Random Forest Prediction': y_pred_rf
})

print(df_comparison.head(10))