# -*- coding: utf-8 -*-
"""Predictive Analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1me9Ez4nnYAW8V-VCqae2YJKpRVpZG2IH

# Data Loading

Import Library yang dibutuhkan
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

"""Membaca data excel menggunakan library pandas"""

#Membaca data excel

df = pd.read_excel("DATA RUMAH.xlsx")
df

"""# Data Cleaning

Melihat Info dataset
"""

df.info()

"""Melihat statistik deskriptif dari dataset"""

df.describe()

"""Drop Kolom NO dan NAMA RUMAH karena tidak diperluhkan"""

df = df.drop(columns = ['NO', 'NAMA RUMAH'])

"""Melihat dataset"""

df.head()

"""Memindahkan kolom harga ke paling kanan"""

harga_column = df.pop('HARGA')
df['HARGA'] = harga_column

df.head()

"""Kode dibawah untuk memeriksa apakah ada nilai null"""

# Memeriksa kolom apakah ada nilai null
df.isna().sum()

""" Memeriksa kolom LB, LT, KT, KM apakah ada nilai nol"""

# Memeriksa kolom LB, LT, KT, KM apakah ada nilai nol

LB = (df.LB == 0).sum()
LT = (df.LT == 0).sum()
KT = (df.KT == 0).sum()
KM = (df.KM == 0).sum()

print("Nilai 0 di kolom LB ada: ", LB)
print("Nilai 0 di kolom LT ada: ", LT)
print("Nilai 0 di kolom KT ada: ", KT)
print("Nilai 0 di kolom KM ada: ", KM)

"""Kode dibawah untuk mengecek outlier pada kolom LB, terdapat 53 outlier pada kolom LB. Outlier tidak di imputasi atau didrop karena akan mempengaruhi korelasi data nantinya"""

sns.boxplot(x=df['LB'])

Q1 = df['LB'].quantile(0.25)
Q3 = df['LB'].quantile(0.75)
IQR = Q3 - Q1
lower_bound2 = Q1 - 1.5 * IQR
upper_bound2 = Q3 + 1.5 * IQR

outliers = df[(df['LB'] < lower_bound2) | (df['LB'] > upper_bound2)]
print("Jumlah outlier:", outliers.shape[0])

"""Kode dibawah untuk mengecek outlier pada kolom LT, terdapat 53 outlier pada kolom LT. Outlier tidak di imputasi atau didrop karena akan mempengaruhi korelasi data nantinya"""

sns.boxplot(x=df['LT'])

Q1 = df['LT'].quantile(0.25)
Q3 = df['LT'].quantile(0.75)
IQR = Q3 - Q1
lower_bound3 = Q1 - 1.5 * IQR
upper_bound3 = Q3 + 1.5 * IQR

outliers3 = df[(df['LT'] < lower_bound3) | (df['LT'] > upper_bound3)]
print("Jumlah outlier:", outliers3.shape[0])

"""# Exploratory Data Analysis

Gambar dibawah ini menunjukkan histogram dari beberapa variabel properti: luas bangunan (LB), luas tanah (LT), jumlah kamar tidur (KT), jumlah kamar mandi (KM), jumlah garasi (GRS), dan harga (HARGA). Mayoritas data untuk semua variabel terkonsentrasi pada nilai rendah, dengan distribusi yang miring ke kanan dan beberapa outlier di sisi kanan. Luas bangunan dan luas tanah kebanyakan di bawah 200. Rumah umumnya memiliki sekitar 4 kamar tidur, 3 kamar mandi, dan 2 garasi, dengan beberapa rumah memiliki jumlah yang jauh lebih tinggi hingga 10.
"""

# Univariate analysis

df.hist(figsize=(10, 8))
plt.show()

"""Gambar dibawahini menunjukkan rata-rata harga properti terhadap jumlah kamar tidur (KT), kamar mandi (KM), dan garasi (GRS). Dari grafik pertama, terlihat bahwa harga rata-rata cenderung meningkat dengan bertambahnya jumlah kamar tidur, dengan lonjakan signifikan pada rumah dengan 10 kamar tidur. Grafik kedua menunjukkan bahwa harga rata-rata juga meningkat seiring bertambahnya jumlah kamar mandi, dengan kenaikan yang lebih signifikan pada rumah dengan 7 kamar mandi. Grafik ketiga memperlihatkan bahwa harga rata-rata meningkat secara konsisten dengan bertambahnya jumlah garasi, dengan puncak pada rumah dengan 8 garasi. Secara keseluruhan, ada kecenderungan harga properti meningkat seiring dengan bertambahnya jumlah fasilitas seperti kamar tidur, kamar mandi, dan garasi."""

#Multivariate Analysis
categorical_cols = ['KT', 'KM', 'GRS']

for col in categorical_cols:
  plt.figure(figsize=(8,6))
  sns.barplot(x=col, y='HARGA', data=df)
  plt.title(f'Rata-rata Harga terhadap {col}')
  plt.xlabel(col)
  plt.ylabel('Rata-rata Harga')
  plt.show()

"""Bisa dilihat pada pola sebaran data grafik pairplot dibawah LT dan LB memiliki korelasi dengan fitur HARGA"""

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(df, diag_kind = 'kde')

"""Menghitung matriks korelasi pada setiap fitur terdapat bahwa fitur LT memiliki nilai terbesar dengan fitur HARGA"""

# Menghitung matriks korelasi dan menampilkan matriks korelasi menggunakan heatmap
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi')
plt.show()

"""Bisa dilihat pada pairplot dibawah bahwa LB dan LT juga memiliki korelasi yang tinggi"""

sns.pairplot(df[['LB', 'LT']], plot_kws={"s": 3})

"""Pada kode dibawah ini bertujuan untuk membagi data test dan data training"""

# Membagi data test dan data training

X = df[['LB','LT','KT','KM','GRS']].values #Feature
y = df['HARGA'].values #Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Pada kode dibawah ini bertujuan untuk melihat jumlah dataset, data training dan data test"""

print(f'Total of sample in whole dataset: {len(X)}')
print(f'Total of sample in train dataset: {len(X_train)}')
print(f'Total of sample in test dataset: {len(X_test)}')

"""# Modelling

### XGBOOST

Kode ini digunakan untuk melatih dan membuat prediksi dengan model XGBoost pada dataset.
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

"""Kode ini menginisialisasi dan melakukan pencarian hyperparameter untuk model XGBoost, kemudian melatih model dengan parameter terbaik yang ditemukan."""

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Inisialisasi model XGBoost
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
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

"""### Random Forest

Kode ini digunakan untuk melatih dan membuat prediksi dengan model Random Forest pada dataset.
"""

# Melatih data dengan Random Forest
from sklearn.ensemble import RandomForestRegressor

# Inisialisasi model Random Forest
rf_model = RandomForestRegressor(random_state=123)

# Latih model dengan data pelatihan
rf_model.fit(X_train, y_train)

# Prediksi harga rumah untuk data pelatihan dan pengujian
y_train_pred_rf = rf_model.predict(X_train)
y_pred_rf = rf_model.predict(X_test)

# Inisialisasi model Random Forest tanpa parameter khusus
rf_model = RandomForestRegressor(random_state=123)

# Definisikan grid parameter untuk pencarian hyperparameter dengan ruang pencarian yang lebih kecil
param_grid = {
    'n_estimators': [50, 100],  # Jumlah pohon dalam hutan
    'max_depth': [None, 10],  # Kedalaman maksimum pohon
    'min_samples_split': [2, 5],  # Jumlah minimum sampel untuk membagi simpul
    'min_samples_leaf': [1, 2]    # Jumlah minimum sampel pada daun
}

# Inisialisasi GridSearchCV dengan n_jobs=-1 untuk memanfaatkan semua core
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # Gunakan MSE negatif untuk mencari nilai terkecil
    cv=3,  # 3-fold cross-validation untuk mempercepat proses
    verbose=1,  # Tampilkan progres pencarian
    n_jobs=-1  # Gunakan semua core prosesor yang tersedia
)

# Lakukan pencarian hyperparameter pada data pelatihan
grid_search.fit(X_train, y_train)

# Dapatkan model terbaik dari hasil pencarian
best_rf_model = grid_search.best_estimator_

# Prediksi pada data pelatihan dan pengujian menggunakan model terbaik
y_train_pred_rf = best_rf_model.predict(X_train)
y_test_pred_rf = best_rf_model.predict(X_test)

"""# Evaluasi Model antara XGboost dan Random Forest

Kode dibawah bertujuan untuk membandingkan nilai MAE, MSE, R-Squared pada model XGBoost dan Model Random Forest pada data Pelatihan dan Data Pengujian
"""

# Hasil evaluasi untuk XGBoost dengan hyperparameter tuning
print("Evaluasi Model XGBoost pada Data Pelatihan:")
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("R2 Score:", r2_score(y_train, y_train_pred))

print("\nEvaluasi Model XGBoost pada Data Pengujian:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R2 Score:", r2_score(y_test, y_test_pred))

# Hasil evaluasi untuk Random Forest dengan hyperparameter tuning
print("\nEvaluasi Model Random Forest pada Data Pelatihan:")
print("MAE:", mean_absolute_error(y_train, y_train_pred_rf))
print("MSE:", mean_squared_error(y_train, y_train_pred_rf))
print("R2 Score:", r2_score(y_train, y_train_pred_rf))

print("\nEvaluasi Model Random Forest pada Data Pengujian:")
print("MAE:", mean_absolute_error(y_test, y_test_pred_rf))
print("MSE:", mean_squared_error(y_test, y_test_pred_rf))
print("R2 Score:", r2_score(y_test, y_test_pred_rf))

"""Kode dibawah bertujuan untuk membandingkan nilai y_true dengan model prediksi XGBoost dan Random Forest"""

# Membuat dataframe perbandingan
df_comparison = pd.DataFrame({
    'y_true': y_test,
    'XGBoost Prediction': y_test_pred,
    'Random Forest Prediction': y_test_pred_rf
})

# Menampilkan 10 baris pertama dari dataframe perbandingan
print(df_comparison.head(10))