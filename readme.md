# Laporan Proyek Machine Learning - Raihan Fahlevi

## Domain Proyek

Industri real estate merupakan salah satu sektor Industri real estate adalah sektor ekonomi yang sangat signifikan dan dinamis, dengan harga rumah dipengaruhi oleh berbagai faktor ekonomi, sosial, dan lingkungan. Memiliki model prediksi harga rumah yang akurat sangat penting bagi pembeli rumah untuk menentukan apakah harga yang ditawarkan sesuai dengan nilai pasar sebenarnya, membantu dalam perencanaan keuangan yang lebih efektif, dan mengatur pembiayaan dengan lebih baik. Bagi penjual rumah, prediksi harga membantu dalam menetapkan harga jual yang kompetitif dan strategi penjualan yang tepat. Proyek ini bertujuan untuk mengembangkan model prediksi harga rumah menggunakan pendekatan machine learning, khususnya regresi dengan Random Forest dan XGBoost, yang mampu menangani data kompleks dan memberikan prediksi yang akurat, sehingga membantu berbagai pihak dalam pengambilan keputusan yang lebih baik dan strategis di pasar penjualan rumah.

## Business Understanding

### Problem Statements

Bagaimana kita dapat memperkirakan harga rumah yang wajar mengingat banyaknya faktor yang memengaruhi harga dan variasi yang sulit diprediksi hanya dengan melihatnya?
Bagaimana cara mengatasi proses penentuan harga rumah secara manual yang membutuhkan waktu dan tenaga signifikan serta mahal dengan metode tradisional seperti survei fisik, analisis pasar komparatif?

### Goal

- Kita dapat memperkirakan harga rumah yang wajar dengan mengembangkan model prediktif berbasis Machine Learning. Model ini akan memperhitungkan berbagai faktor yang memengaruhi harga rumah, seperti lokasi, ukuran, jumlah kamar, kondisi rumah, dan lain-lain. Dengan menggunakan data historis dan algoritma yang tepat, model ini dapat menghasilkan estimasi harga yang akurat, membantu pembeli dan penjual dalam membuat keputusan yang lebih tepat mengenai harga rumah.
- Kita dapat mengatasi proses penentuan harga rumah secara manual yang memakan waktu dan mahal dengan mengotomatisasi proses penilaian rumah menggunakan Machine Learning. Dengan memanfaatkan teknologi ini, kita dapat menggantikan metode tradisional seperti survei fisik, analisis pasar komparatif dengan sistem otomatis yang dapat memberikan hasil penilaian dengan cepat dan efisien. Otomatisasi ini tidak hanya menghemat waktu dan sumber daya, tetapi juga meningkatkan konsistensi dan akurasi dalam penilaian harga rumah.

## Data Understanding

**Jumlah Data**: Dataset yang digunakan hanya dataset DATA RUMAH. Dataset ini terdiri dari 1010 baris dan 6 kolom, masing-masing mewakili rumah yang berbeda dengan berbagai karakteristik yang mempengaruhi harganya.

**Kondisi Data**: Data yang digunakan pada proyek ini sudah bersih hanya perlu drop kolom NO dan NAMA RUMAH saja untuk melakukan modelling

**Sumber Data**
Data ini diperoleh dari kaggle yang dapat diakses melalui link berikut: https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah/data

### Variabel-variabel pada Dataset rumah sebagai berikut:

Dataset ini berisi informasi mengenai karakteristik rumah dan harga jualnya. Berikut adalah variabel-variabel yang ada:

- `LB`: Luas bangunan rumah dalam meter persegi.
- `LT`: Luas tanah rumah dalam meter persegi.
- `KT`: Jumlah kamar tidur dalam rumah.
- `KM`: Jumlah kamar mandi dalam rumah.
- `GRS`: Apakah rumah memiliki garasi atau tidak.
- `HARGA`: Harga jual rumah dalam Rupiah.

### Pemeriksaan data

Langkah pertama adalah memeriksa dataset untuk mendapatkan pemahaman tentang karakteristiknya. Ini termasuk:

- `df.info()` : Untuk melihat informasi umum tentang dataset, seperti jumlah baris, kolom, tipe data, dan nilai yang hilang.
- `df.describe()`: Untuk mendapatkan statistik deskriptif dari variabel numerik, seperti rata-rata, standar deviasi, nilai minimum, dan maksimum.

Berikut gambar Statistik Deskriptifnya:

![Screenshot 2024-07-23 225052](https://github.com/user-attachments/assets/bea024da-1f48-491f-a88f-18d3d13d45a5)

Berdasarkan deskripsi statistik yang diberikan untuk dataset properti, berikut adalah penjelasan dari setiap fitur:

- `HARGA`:

Rata-rata (mean): Luas bangunan rata-rata adalah sekitar 7,63 miliar.

Standar Deviasi (std): Standar deviasi adalah 7,34 miliar.

Nilai Minimum (min): Luas bangunan terkecil adalah 430 juta.

Kuartil Pertama (25%): 3,26 miliar.

Median (50%): 5 miliar.

Kuartil Ketiga (75%): 9 miliar.

Nilai Maksimum (max): Luas bangunan terbesar adalah 65 miliar.

- `Luas Bangunan (LB)`:

Rata-rata (mean): Luas Bangunan rata-rata adalah 276,54 m².

Standar Deviasi (std): Standar deviasi adalah 177,86 m².

Nilai Minimum (min): Luas Bangunan terkecil adalah 40 m².

Kuartil Pertama (25%): 150 m².

Median (50%): 216,5 m².

Kuartil Ketiga (75%): 350 m².

Nilai Maksimum (max): Luas tanah terbesar adalah 1126 m².

- `Luas Tanah (LT)`:

Rata-rata (mean): Luas tanah rata-rata adalah 237 m².

Standar Deviasi (std): Standar deviasi adalah 179 m².

Nilai Minimum (min): Luas tanah terkecil adalah 25 m².

Kuartil Pertama (25%): 130 m².

Median (50%): 165 m².

Kuartil Ketiga (75%): 290 m².

Nilai Maksimum (max): Luas tanah terbesar adalah 1400 m².

- `Kamar Tidur (KT)`:

Rata-rata (mean): Rata-rata jumlah kamar tidur adalah 4,6.

Standar Deviasi (std): Standar deviasi adalah 1,57.

Nilai Minimum (min): Jumlah kamar tidur terkecil adalah 2.

Kuartil Pertama (25%): 4 kamar tidur.

Median (50%): 4 kamar tidur.

Kuartil Ketiga (75%): 5 kamar tidur.

Nilai Maksimum (max): Jumlah kamar tidur terbanyak adalah 10.

- `Kamar Mandi (KM)`:

Rata-rata (mean): Rata-rata jumlah kamar mandi adalah 3,60.

Standar Deviasi (std): Standar deviasi adalah 1,42.

Nilai Minimum (min): Jumlah kamar mandi terkecil adalah 1.

Kuartil Pertama (25%): 3 kamar mandi.

Median (50%): 3 kamar mandi.

Kuartil Ketiga (75%): 4 kamar mandi.

Nilai Maksimum (max): Jumlah kamar mandi terbanyak adalah 10.

- `Garasi (GRS)` :

Rata-rata (mean): Rata-rata jumlah garasi adalah 1,92.

Standar Deviasi (std): Standar deviasi adalah 1,51.
Nilai Minimum (min): Tidak ada garasi (0).

Kuartil Pertama (25%): 1 garasi.

Median (50%): 2 garasi.

Kuartil Ketiga (75%): 2 garasi.

Nilai Maksimum (max): Jumlah garasi terbanyak adalah 10.

### Pengecekan Outlier

Pada dataset ini terdapat outlier dikolom LT dan LB

Bisa dilihat pada boxplot dibawah menunjukkan adanya outlier. Titik-titik yang berada di luar whiskers di sebelah kanan menunjukkan adanya outlier dalam data LB. Ini adalah titik-titik data yang nilainya jauh lebih tinggi dari kuartil atas ditambah 1.5 kali interquartile range (IQR).

![Outlier LB](https://github.com/user-attachments/assets/47c08bbc-d39d-42f2-ae56-fcebf1dbfc80)

Bisa dilihat pada boxplot dibawah menunjukkan adanya outlier. Pada gambar diatas, titik-titik yang berada di sebelah kanan whiskers menunjukkan adanya outlier dalam data LT. Ini adalah titik-titik data yang nilainya jauh lebih tinggi dari kuartil atas ditambah 1.5 kali interquartile range (IQR).

![Outlier LT](https://github.com/user-attachments/assets/05481315-4e43-4583-992a-a100c085cbfd)


Jadi pada dataset terdapat outlier tetapi tidak dihapus karena itu akan berpengaruh korelasi data bisa dilihat pada gambar dibawah jika outlier dihapus dan tidak dihapus:

**Gambar data jika outlier tidak dihapus:**
![outlier](https://github.com/user-attachments/assets/666c2621-4357-4026-809d-7606bc415f5b)

### Gambar data jika outlier dihapus:

![nooutlier](https://github.com/user-attachments/assets/c913abe6-dcff-4a7e-90f6-5e3319045565)

### Exploratory Data Analysis

- **Univariate analysis**
  ![univariate](https://github.com/user-attachments/assets/c5e64dbc-a394-483b-9c71-107379246630)

Gambar diatas ini menunjukkan histogram dari beberapa variabel properti: luas bangunan (LB), luas tanah (LT), jumlah kamar tidur (KT), jumlah kamar mandi (KM), jumlah garasi (GRS), dan harga (HARGA). Mayoritas data untuk semua variabel terkonsentrasi pada nilai rendah, dengan distribusi yang miring ke kanan dan beberapa outlier di sisi kanan. Luas bangunan dan luas tanah kebanyakan di bawah 200. Rumah umumnya memiliki sekitar 4 kamar tidur, 3 kamar mandi, dan 2 garasi, dengan beberapa rumah memiliki jumlah yang jauh lebih tinggi hingga 10.

- **Multivariate Analysis**
  ![Multivariate](https://github.com/user-attachments/assets/08c0c4ee-71d7-41ea-8c55-ab7a61602621)

Gambar diatas ini menunjukkan rata-rata harga properti terhadap jumlah kamar tidur (KT), kamar mandi (KM), dan garasi (GRS). Dari grafik pertama, terlihat bahwa harga rata-rata cenderung meningkat dengan bertambahnya jumlah kamar tidur, dengan lonjakan signifikan pada rumah dengan 10 kamar tidur. Grafik kedua menunjukkan bahwa harga rata-rata juga meningkat seiring bertambahnya jumlah kamar mandi, dengan kenaikan yang lebih signifikan pada rumah dengan 7 kamar mandi. Grafik ketiga memperlihatkan bahwa harga rata-rata meningkat secara konsisten dengan bertambahnya jumlah garasi, dengan puncak pada rumah dengan 8 garasi. Secara keseluruhan, ada kecenderungan harga properti meningkat seiring dengan bertambahnya jumlah fasilitas seperti kamar tidur, kamar mandi, dan garasi.

- **Numerical Features**
  ![hubungan antar fitur numerik](https://github.com/user-attachments/assets/17175ef6-e80a-4de7-ae6d-13ed05c3c7d4)

Bisa dilihat pada pola sebaran data grafik pairplot dibawah LT dan LB memiliki korelasi dengan fitur HARGA

## Data Preparation

### 1. Memindah kolom harga ke paling kanan
Memindahkan kolom harga ke paling kanan agar mudah
untuk membaca dataset.

### 2. Penanganan Nilai yang Hilang

Dataset diperiksa untuk nilai yang hilang menggunakan df.isna().sum(). Tidak ditemukan nilai yang hilang dalam dataset ini.

### 3. Drop Kolom yang tidak digunakan

Pada dataset yang digunakan ini saya hanya memakai 6 kolom saja yakni LB, LT, KT, KM, GRS, HARGA. Saya tidak memakai kolom NO dan NAMA karena kolom itu tidak dibutuhkan untuk project ini.

### 4. Penanganan Nilai Nol pada Variabel Tertentu

Variabel LB, LT, KT, dan KM diperiksa untuk nilai nol menggunakan (df.LB == 0).sum() dan seterusnya. Meskipun nilai nol mungkin valid dalam beberapa kasus, penting untuk memastikan bahwa mereka tidak mewakili kesalahan entri data. Jika nilai nol dianggap tidak valid, mereka dapat ditangani dengan cara yang sama seperti nilai yang hilang.

### 5. Pemisahan Data

Dataset dibagi menjadi set pelatihan dan pengujian menggunakan train_test_split dengan rasio 80:20. Ini memungkinkan evaluasi kinerja model pada data yang tidak terlihat.

## Modeling

### 1. XGBoost

Pada bagian ini, kita akan melatih model prediksi harga rumah menggunakan algoritma XGBoost (Extreme Gradient Boosting). Proses pelatihan ini melibatkan beberapa tahapan utama, yaitu inisialisasi model, pelatihan model pada data, prediksi hasil, dan pencarian hyperparameter untuk mendapatkan model terbaik. Berikut adalah penjelasan lebih rinci mengenai tahapan-tahapan tersebut:

A. Inisialisasi Model XGBoost
Model XGBoost: Menggunakan xgboost.XGBRegressor yang diinisialisasi dengan parameter dasar.

```
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
```

random_state=42: Menetapkan seed acak untuk memastikan hasil yang konsisten setiap kali kode dijalankan.

B. Melatih Model
Latih Model: Model dilatih menggunakan data pelatihan

```
xgb_model.fit(X_train, y_train)
```

C. Prediksi
Prediksi Data Latih dan Uji: Model digunakan untuk memprediksi nilai pada data pelatihan dan data uji.

```
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)
```

D. Hyperparameter Tuning

#### Parameter Dasar Model XGBoost

- objective='reg': Fungsi objektif ini digunakan untuk regresi dengan tujuan meminimalkan kesalahan kuadrat antara nilai prediksi dan nilai aktual. Dalam konteks prediksi harga rumah, ini berarti model berusaha untuk mengurangi perbedaan antara harga rumah yang diprediksi dan harga sebenarnya.

- random_state=42: Seed untuk generator angka acak digunakan untuk memastikan hasil eksperimen dapat direproduksi. Dengan menetapkan random_state, hasil pelatihan model menjadi konsisten setiap kali dijalankan.

#### Grid Search untuk Hyperparameter Tuning

- learning_rate: Parameter ini menentukan kecepatan model dalam belajar. Nilai yang dicoba adalah 0.01, 0.1, dan 0.2. Learning rate yang lebih kecil membuat model belajar lebih lambat tetapi lebih teliti, sementara learning rate yang lebih besar mempercepat proses belajar tetapi berisiko overfitting.
- max_depth: Parameter ini membatasi kedalaman setiap pohon dalam model. Nilai yang dicoba adalah 2, 3, dan 4. Kedalaman pohon yang lebih besar dapat menangkap lebih banyak informasi tetapi juga berisiko overfitting.
- n_estimators: Parameter ini menentukan jumlah pohon yang akan dibangun dalam model. Nilai yang dicoba adalah 50, 100, dan 150. Jumlah pohon yang lebih banyak dapat meningkatkan akurasi model tetapi juga meningkatkan risiko overfitting dan waktu komputasi.

#### Proses Pencarian Hyperparameter Terbaik dengan GridSearchCV

- GridSearchCV digunakan untuk menguji berbagai kombinasi hyperparameter. Proses ini dilakukan dengan cara membagi data pelatihan menjadi beberapa bagian, melatih model pada kombinasi parameter yang berbeda, dan mengevaluasi performa model pada bagian yang tidak digunakan dalam pelatihan.
- Evaluasi Performansi: Performansi model dievaluasi berdasarkan metrik kesalahan kuadrat rata-rata (Mean Squared Error, MSE). Kombinasi hyperparameter yang menghasilkan MSE terendah dianggap sebagai yang terbaik.
- Kombinasi Hyperparameter Terbaik
Setelah melakukan GridSearchCV, kombinasi hyperparameter yang menghasilkan performa terbaik adalah:

      learning_rate=0.1

      max_depth=3

      n_estimators=100

Kombinasi ini memberikan keseimbangan yang baik antara akurasi prediksi dan kemampuan model untuk generalisasi pada data yang belum pernah dilihat.

Pada data ini XGBoost bekerja seperti dengan cara XGBoost membangun banyak pohon keputusan secara berurutan. Setiap pohon baru memperbaiki kesalahan dari pohon sebelumnya. Model baru dibangun untuk mengurangi gradient loss dari model sebelumnya, sehingga model belajar untuk memprediksi kesalahan.
XGBoost menggunakan teknik regularisasi (L1 dan L2 regularization) untuk mengurangi overfitting dan membuat model lebih generalis.XGBoost secara otomatis menangani nilai yang hilang, yang membuatnya robust terhadap data yang tidak lengkap.

### Random Forest

Pada bagian ini, saya akan melatih model prediksi harga rumah menggunakan algoritma Random Forest. Proses pelatihan ini melibatkan beberapa tahapan utama, yaitu inisialisasi model, pelatihan model pada data, prediksi hasil. Berikut adalah penjelasan lebih rinci mengenai tahapan-tahapan tersebut:

A. Inisialisasi Model:
`RandomForestRegressor(random_state=123)`: Membuat model Random Forest dengan parameter random_state untuk memastikan hasil yang konsisten setiap kali kode dijalankan.

B. Pelatihan Model:
`rf_model.fit(X_train, y_train)`: Melatih model menggunakan data pelatihan (`X_train` dan `y_train`).

C. Prediksi:
`rf_model.predict(X_train)`: Membuat prediksi untuk data pelatihan.
`rf_model.predict(X_test)`: Membuat prediksi untuk data pengujian.

Pada data ini Random Forest bekerja dengan cara Random Forest membentuk banyak pohon keputusan secara acak, di mana setiap pohon dilatih pada subset acak dari data pelatihan (bagging). Hal ini membantu dalam mengurangi varians dan mengatasi overfitting. Setiap pohon keputusan memberikan prediksi independen. Untuk regresi, prediksi akhir diperoleh dengan mengambil rata-rata prediksi dari semua pohon. Random Forest juga dapat memberikan estimasi pentingnya setiap fitur dalam dataset, yang berguna untuk memahami fitur mana yang paling berpengaruh dalam membuat prediksi.

D. Hyperparameter Tuning

- Inisialisasi Model Random Forest

```
rf_model = RandomForestRegressor(random_state=123)
```

- Mendefinisikan Grid Parameter untuk Pencarian Hyperparameter
```
param_grid = {
    'n_estimators': [50, 100],  
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

```
`n_estimators`: Jumlah pohon dalam hutan. Dicoba dengan 50 dan 100.

`max_depth`: Kedalaman maksimum pohon. None berarti pohon akan tumbuh sampai semua daun murni atau memiliki kurang dari min_samples_split sampel.

`min_samples_split`: Jumlah minimum sampel yang diperlukan untuk membagi simpul internal. Dicoba dengan 2 dan 5.
`min_samples_leaf`: Jumlah minimum sampel yang harus ada di daun. Dicoba dengan 1 dan 2.

- Inisialisasi GridSearchCV
```
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=-1
)

```

- Lakukan Pencarian Hyperparameter pada Data Pelatihan

```
grid_search.fit(X_train, y_train)
```

- Prediksi pada Data Pelatihan dan Pengujian Menggunakan Model Terbaik
```
y_train_pred_rf = best_rf_model.predict(X_train)
y_test_pred_rf = best_rf_model.predict(X_test)
```

## Kelebihan dan Kekurangan Algoritma:

### XGBoost
- Kelebihan: Akurasi tinggi, penanganan data yang hilang, regularisasi untuk mencegah overfitting, dan kecepatan yang baik.
- Kekurangan: Dapat menjadi kompleks untuk hyperparameter tuning, rentan terhadap overfitting jika tidak dikonfigurasi dengan benar.

### Random Forest

- Kelebihan: Robust terhadap overfitting, dapat menangani data non-linear, dan dapat digunakan untuk feature importance selection.
- Kekurangan: Kurang akurat dibandingkan XGBoost dalam beberapa kasus, dapat menjadi lambat untuk dataset besar.

## Evaluation

Metrik yang digunakan pada project ini adalah
**Mean Absolue Error (MAE), Mean Squared Error (MSE), dan R-squared**

**Mean Absolute Error (MAE)**: Mengukur rata-rata selisih absolut antara nilai prediksi dan nilai sebenarnya. MAE memberikan gambaran umum tentang seberapa jauh prediksi dari nilai sebenarnya. Semakin kecil nilai MAE, semakin baik kinerja model.

**Mean Squared Error (MSE)**: Mengukur rata-rata kuadrat selisih antara nilai prediksi dan nilai sebenarnya. MSE memberikan penalti yang lebih besar untuk kesalahan yang lebih besar. Semakin kecil nilai MSE, semakin baik kinerja model.

**R2 Score**: Mengukur seberapa baik model cocok dengan data. Nilai R2 Score berkisar antara 0 hingga 1, di mana 1 menunjukkan kesesuaian yang sempurna. Semakin tinggi nilai R2 Score, semakin baik model dalam menjelaskan variasi data.

### Model XGBoost

**Evaluasi pada Data Pelatihan**:

MAE: 1,469,339,139.4591584

MSE: 6.934707093962935e+18

R2 Score: 0.8752894282801635

**Evaluasi pada Data Pengujian**:

MAE: 1,775,984,023.5

MSE: 8.609931393182666e+18

R2 Score: 0.8155652739886136

Hasil:

**MAE** yang tinggi menunjukkan bahwa prediksi rata-rata model memiliki kesalahan absolut sebesar sekitar 1.47 miliar pada data pelatihan dan 1.78 miliar pada data pengujian.

**MSE** yang tinggi menunjukkan adanya beberapa prediksi dengan kesalahan besar, karena MSE memperbesar pengaruh dari kesalahan besar.

**R2 Score** pada data pelatihan (0.875) dan pengujian (0.816) menunjukkan bahwa model ini dapat menjelaskan sekitar 87.5% variabilitas data pada pelatihan dan 81.6% pada pengujian. Meskipun ada penurunan performa pada data pengujian, penurunan ini tidak terlalu drastis, menunjukkan model yang cukup robust.

### Model Random Forest

**Evaluasi Model Random Forest pada Data Pelatihan:**

MAE: 1102919539.8623614

MSE: 6.043296672889429e+18

R2 Score: 0.8913201418694765

**Evaluasi Model Random Forest pada Data Pengujian:**

MAE: 1738359197.5723364

MSE: 9.081039075018135e+18

R2 Score: 0.8054736005183717

Hasil:

**MAE** yang lebih rendah pada data pelatihan (sekitar 1.10 miliar) menunjukkan bahwa model ini memiliki kesalahan rata-rata yang lebih kecil pada data pelatihan dibandingkan dengan data pengujian (sekitar 1.74 miliar). Hal ini mengindikasikan bahwa model lebih akurat dalam memprediksi data yang sudah dilatih dibandingkan dengan data baru yang belum pernah dilihat sebelumnya.

**MSE** yang lebih rendah pada data pelatihan (6.04e+18) menunjukkan bahwa model ini lebih baik dalam mengurangi pengaruh dari kesalahan besar pada data pelatihan. Namun, pada data pengujian, MSE lebih tinggi (9.08e+18), yang menunjukkan bahwa model mengalami kesulitan dalam menangani beberapa kesalahan besar pada data pengujian dibandingkan pada data pelatihan.

**R2 Score** yang tinggi pada data pelatihan (0.891) menunjukkan bahwa model ini dapat menjelaskan 89.1% variabilitas data pada pelatihan, yang menunjukkan performa yang sangat baik pada data pelatihan. Namun, R² Score yang menurun pada data pengujian (0.805) menunjukkan bahwa model kehilangan beberapa kemampuan untuk menjelaskan variabilitas pada data pengujian. Penurunan ini mengindikasikan adanya overfitting, di mana model sangat baik dalam mempelajari data pelatihan tetapi kurang mampu menggeneralisasi pola pada data yang belum pernah dilihat sebelumnya.

Model machine learning bekerja dengan baik dan dapat memprediksi harga rumah dengan cepat dan akurat. Dengan mempertimbangkan berbagai faktor yang memengaruhi harga rumah, model ini mampu memberikan estimasi yang wajar dan dapat diandalkan. Hal ini mengatasi tantangan utama dalam memperkirakan harga rumah yang sangat bervariasi dan sulit diprediksi hanya dengan pengamatan langsung.

Goals berhasil dicapai dengan membuat model machine learning yang mampu memprediksi harga rumah. Model ini tidak hanya membantu dalam menentukan harga yang wajar bagi pembeli dan penjual, tetapi juga mengotomatiskan proses penilaian yang sebelumnya memerlukan waktu dan biaya yang signifikan. Dengan demikian, model ini mendukung keputusan yang lebih baik dan cepat di pasar real estate.

Penggunaan model XGBoost dan Random Forest memberikan dampak positif dengan menyediakan alat prediksi harga rumah yang lebih akurat dan stabil. Model ini memanfaatkan berbagai fitur dan data yang relevan untuk menghasilkan prediksi yang lebih dapat diandalkan dibandingkan metode tradisional. Hal ini membantu pembeli dan penjual membuat keputusan yang lebih baik dan strategis di pasar real estate. Dengan alat prediksi yang efisien, para pelaku pasar dapat menghemat waktu dan sumber daya, serta meningkatkan kepuasan dan kepercayaan dalam transaksi properti.

## Kesimpulan

**Random Forest** menunjukkan performa yang lebih baik pada data pelatihan dengan MAE dan MSE yang lebih rendah serta R² Score yang lebih tinggi. Namun, hal ini juga dapat menunjukkan bahwa Random Forest mungkin mengalami overfitting karena performanya menurun lebih drastis pada data pengujian dibandingkan XGBoost.
**XGBoost** menunjukkan performa yang lebih konsisten antara data pelatihan dan pengujian dengan perbedaan yang lebih kecil dalam R² Score, MAE, dan MSE, yang menunjukkan kemampuan generalisasi yang lebih baik pada data baru.

Berdasarkan hasil evaluasi ini, XGBoost lebih direkomendasikan untuk digunakan dalam prediksi harga rumah pada dataset ini karena kemampuannya dalam generalisasi yang lebih baik dibandingkan dengan Random Forest. Meskipun Random Forest menunjukkan performa yang sangat baik pada data pelatihan, overfitting yang terjadi membuatnya kurang ideal untuk digunakan pada data baru yang tidak terlihat sebelumnya.
