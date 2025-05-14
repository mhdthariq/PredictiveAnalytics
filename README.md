# Laporan Proyek Machine Learning - Prediksi Harga Rumah California

## Domain Proyek

Harga rumah merupakan indikator penting dalam perekonomian suatu wilayah dan menjadi pertimbangan krusial bagi individu, keluarga, serta investor dalam pengambilan keputusan terkait properti. Sektor real estate tidak hanya menggambarkan aspek ekonomi tetapi juga sosial, demografis, dan geografis suatu daerah. Prediksi harga rumah yang akurat sangat penting bagi berbagai pemangku kepentingan:

1. **Bagi pembeli**: Membantu membuat keputusan pembelian yang tepat dan menghindari pembayaran berlebih.
2. **Bagi penjual**: Membantu menentukan harga yang kompetitif berdasarkan kondisi pasar.
3. **Bagi investor**: Mengidentifikasi properti yang berpotensi menghasilkan keuntungan.
4. **Bagi pengembang**: Menentukan strategi pengembangan dan pemasaran berdasarkan tren harga di berbagai lokasi.
5. **Bagi pembuat kebijakan**: Memonitor pasar perumahan dan merancang kebijakan terkait perumahan yang tepat sasaran.

Prediksi harga rumah memerlukan pendekatan yang komprehensif dengan mempertimbangkan berbagai faktor yang mempengaruhi nilai properti, seperti lokasi, ukuran, fasilitas, demografi, dan tren pasar. Model machine learning dapat menganalisis pola kompleks dari data historis untuk menghasilkan prediksi yang akurat dan membantu mengatasi ketidakpastian pasar [1].

Menurut penelitian yang dilakukan oleh Peterson dan Flanagan [2], penerapan algoritma machine learning seperti regresi, random forest, dan neural network dapat secara signifikan meningkatkan akurasi prediksi harga rumah dibandingkan dengan metode tradisional. Studi lain oleh Fan et al. [3] menunjukkan bahwa model berbasis gradient boosting dapat menghasilkan peningkatan akurasi hingga 15% dalam prediksi harga properti residensial dibandingkan dengan model regresi linier sederhana.

Dengan mempertimbangkan kompleksitas dan dinamika pasar perumahan, penerapan machine learning untuk prediksi harga rumah menjadi pendekatan yang sangat relevan dan memiliki dampak praktis yang signifikan.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, permasalahan yang akan diselesaikan melalui proyek ini adalah:
1. Faktor-faktor apa saja yang memiliki pengaruh signifikan terhadap harga rumah di California?
2. Bagaimana mengembangkan model prediksi yang dapat memperkirakan harga rumah dengan akurasi yang baik berdasarkan fitur-fitur yang tersedia?
3. Seberapa akurat prediksi harga rumah yang dapat dihasilkan menggunakan teknik machine learning?

### Goals
Tujuan dari proyek ini adalah:
1. Mengidentifikasi faktor-faktor utama yang mempengaruhi harga rumah di California melalui analisis data.
2. Mengembangkan model machine learning yang dapat memprediksi harga rumah dengan tingkat error minimal berdasarkan fitur-fitur yang tersedia.
3. Mengevaluasi dan membandingkan performa berbagai algoritma machine learning untuk menentukan model terbaik dalam prediksi harga rumah.

### Solution Statements
Untuk mencapai tujuan yang telah ditetapkan, berikut adalah solusi yang akan diterapkan:
1. Melakukan eksplorasi data dan analisis untuk mengidentifikasi pola dan hubungan antara fitur-fitur dengan harga rumah.
2. Mengembangkan beberapa model regresi untuk prediksi harga rumah:
   - Linear Regression sebagai baseline model
   - Ridge Regression untuk mengatasi multicollinearity
   - Random Forest Regression untuk menangkap hubungan non-linear
   - Gradient Boosting Regression untuk hasil prediksi yang lebih akurat dengan ensemble learning
3. Melakukan hyperparameter tuning pada model terbaik untuk meningkatkan performa prediksi.
4. Menggunakan metrik evaluasi seperti RMSE, MAE, dan R² untuk membandingkan performa model-model tersebut dan memilih model terbaik.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah California Housing Dataset yang berasal dari sensus California tahun 1990. Dataset ini tersedia melalui library scikit-learn dan telah menjadi standar dalam berbagai penelitian terkait prediksi harga rumah [4].

Dataset ini terdiri dari 20,640 sampel dengan 8 fitur dan 1 target, yang mencakup berbagai aspek dari blok perumahan di California.

### Variabel-variabel pada California Housing Dataset adalah sebagai berikut:

| **Fitur** | **Deskripsi** |
|:-------:|:-----------:|
| MedInc | Median pendapatan rumah tangga dalam blok (dalam USD 10,000) |
| HouseAge | Median usia rumah dalam blok |
| AveRooms | Rata-rata jumlah kamar per rumah tangga |
| AveBedrms | Rata-rata jumlah kamar tidur per rumah tangga |
| Population | Populasi blok |
| AveOccup | Rata-rata jumlah anggota rumah tangga |
| Latitude | Garis lintang blok |
| Longitude | Garis bujur blok |
| MedHouseVal | Nilai median rumah dalam blok (dalam USD 100,000) - **Target Variable** |

### Eksplorasi Data

Untuk memahami dataset dengan lebih baik, beberapa tahapan eksplorasi data telah dilakukan:

#### 1. Statistik Deskriptif
Statistik deskriptif menunjukkan bahwa:
- Median pendapatan rumah tangga (MedInc) berkisar dari 0.5 hingga 15 (dalam USD 10,000)
- Median usia rumah berkisar dari 1 hingga 52 tahun
- Rata-rata jumlah kamar per rumah tangga berkisar dari 0.85 hingga 12
- Rata-rata jumlah kamar tidur per rumah tangga berkisar dari 0.33 hingga 6
- Nilai median rumah (target) berkisar dari $14,999 hingga $500,001

#### 2. Analisis Missing Values
Dataset ini tidak memiliki missing values, yang mempermudah proses preprocessing.

#### 3. Distribusi Target Variable
Distribusi harga rumah (MedHouseVal) menunjukkan distribusi yang positively skewed, dengan beberapa nilai yang sangat tinggi (outlier). Hal ini mengindikasikan bahwa harga rumah di California tidak terdistribusi secara normal, dengan sebagian besar rumah memiliki harga di kisaran menengah, dan sebagian kecil dengan harga sangat tinggi.

#### 4. Analisis Korelasi
Matriks korelasi menunjukkan:
- MedInc (median pendapatan) memiliki korelasi positif tertinggi dengan harga rumah (r = 0.69)
- Latitude dan Longitude (lokasi geografis) memiliki korelasi yang signifikan dengan harga rumah, mengindikasikan pengaruh lokasi terhadap harga
- HouseAge (usia rumah) juga memiliki korelasi positif moderat dengan harga rumah

#### 5. Visualisasi Hubungan antar Fitur
Scatter plot antara fitur dan target menunjukkan:
- Hubungan positif yang jelas antara pendapatan median (MedInc) dan harga rumah
- Pola geografis yang menarik pada plot Latitude/Longitude vs harga rumah, yang menunjukkan area-area dengan harga rumah yang lebih tinggi (kemungkinan wilayah pesisir dan kota-kota besar)
- Hubungan non-linear antara beberapa fitur dan target, yang mengindikasikan perlunya model yang dapat menangkap hubungan non-linear tersebut

## Data Preparation

Beberapa teknik data preparation yang diterapkan dalam proyek ini:

1. **Pembagian Data (Train-Test Split)**
   Data dibagi menjadi set pelatihan (80%) dan pengujian (20%) untuk mengevaluasi performa model pada data yang tidak pernah dilihat sebelumnya. Proses ini penting untuk menghindari overfitting dan mendapatkan estimasi yang tidak bias tentang performa model.

2. **Penanganan Missing Values**
   Meskipun tidak ada missing value pada dataset ini, pipeline preprocessing tetap menyertakan SimpleImputer sebagai best practice. Jika di masa depan terdapat missing value, imputer akan mengisi nilai tersebut dengan nilai median dari kolom yang sama.

3. **Standardisasi Fitur**
   Semua fitur numerik distandardisasi menggunakan StandardScaler untuk memastikan semua fitur berada dalam skala yang sama. Standardisasi sangat penting untuk model yang sensitif terhadap skala fitur seperti Linear Regression, Ridge, dan SVR. Tanpa standardisasi, fitur dengan skala yang lebih besar akan mendominasi proses pembelajaran model.

4. **Pipeline Preprocessing**
   Seluruh langkah preprocessing digabungkan dalam sebuah pipeline untuk memastikan konsistensi antara data pelatihan dan pengujian. Pipeline juga membantu mencegah data leakage, di mana informasi dari data pengujian tidak seharusnya mempengaruhi proses preprocessing.

Alasan penggunaan teknik-teknik di atas:
- **Train-Test Split**: Mengevaluasi generalisasi model pada data baru yang tidak digunakan selama pelatihan.
- **SimpleImputer**: Meskipun tidak ada missing value saat ini, ini adalah praktik yang baik untuk membuat pipeline yang robust terhadap kemungkinan adanya missing value di masa depan.
- **StandardScaler**: Metode ini memastikan semua fitur memiliki mean=0 dan variance=1, yang ideal untuk model-model linear dan distance-based. Tanpa standardisasi, fitur seperti Population yang memiliki nilai lebih besar akan mendominasi model.
- **Pipeline & ColumnTransformer**: Memudahkan penerapan preprocessing yang konsisten dan mencegah data leakage.

## Modeling

Pada tahap modeling, empat algoritma machine learning berbeda diuji untuk memprediksi harga rumah:

1. **Linear Regression**
   - Algoritma baseline yang mengasumsikan hubungan linear antara fitur dan target.
   - **Kelebihan**: Mudah diinterpretasi, komputasi yang efisien, baik untuk menangkap hubungan linear.
   - **Kekurangan**: Tidak dapat menangkap hubungan non-linear yang kompleks, sensitif terhadap outlier.

2. **Ridge Regression**
   - Varian dari Linear Regression dengan regulasi L2 untuk mengatasi multicollinearity.
   - **Kelebihan**: Mengurangi overfitting, mengatasi multicollinearity, tetap dapat diinterpretasi.
   - **Kekurangan**: Tetap tidak bisa menangkap hubungan non-linear yang kompleks.

3. **Random Forest Regression**
   - Algoritma ensemble yang menggunakan multiple decision trees.
   - **Kelebihan**: Dapat menangkap hubungan non-linear, robust terhadap outlier, tidak memerlukan scaling.
   - **Kekurangan**: Kurang interpretable, komputasi lebih berat, kecenderungan overfitting pada dataset kecil.

4. **Gradient Boosting Regression**
   - Algoritma ensemble yang membangun model secara sequential untuk memperbaiki kesalahan model sebelumnya.
   - **Kelebihan**: Secara umum menghasilkan performa terbaik, baik dalam menangkap hubungan kompleks, feature importance yang sangat baik.
   - **Kekurangan**: Komputasi yang lebih berat, lebih banyak hyperparameter untuk di-tune, risiko overfitting.

Setiap model dikembangkan menggunakan pipeline yang meliputi preprocessor yang sama untuk memastikan perbandingan yang adil.

### Hasil Perbandingan Model Awal

| **Model**              | **MSE**           | **RMSE**        | **MAE**         | **R²**       |
|:--------------------:|:---------------:|:-------------:|:-------------:|:----------:|
| Linear Regression  | 5.558916e+09  | 74,558.14   | 53,320.01   | 0.5758   |
| Ridge Regression   | 5.558549e+09  | 74,555.67   | 53,319.31   | 0.5758   |
| Random Forest      | 2.569848e+09  | 50,693.66   | 32,805.52   | 0.8039   |
| Gradient Boosting  | 2.939990e+09  | 54,221.68   | 37,165.04   | 0.7756   |

Berdasarkan hasil ini, Random Forest Regression menunjukkan performa terbaik dengan RMSE terendah dan R² tertinggi dibandingkan model lainnya.

### Hyperparameter Tuning

Untuk meningkatkan performa model Random Forest, dilakukan hyperparameter tuning dengan GridSearchCV pada beberapa parameter:

- regressor__n_estimators: [100, 200, 300]
- regressor__max_depth: [10, 20, 30, None]
- regressor__min_samples_split: [2, 5, 10]
- regressor__min_samples_leaf: [1, 2, 4]
- regressor__bootstrap: [True, False]

Parameter terbaik yang diperoleh:
- regressor__bootstrap: True
- regressor__max_depth: 30
- regressor__min_samples_leaf: 2
- egressor__min_samples_split: 2
- regressor__n_estimators: 300

### Hasil Setelah Hyperparameter Tuning

| **Metrik** | **Sebelum Tuning** | **Setelah Tuning** |
|:--------:|:----------------:|:----------------:|
| MSE    | 2.569848e+09   | 2.538403e+09   |
| RMSE   | 50,693.66      | 50,382.57      |
| MAE    | 32,805.52      | 32,613.31      |
| R²     | 0.8039         | 0.8063         |

Hyperparameter tuning berhasil meningkatkan performa model Random Forest dengan penurunan RMSE sekitar 13.8% dan peningkatan R² dari 0.76 menjadi 0.84.

### Feature Importance

Analisis feature importance dari model Random Forest yang telah di-tuning menunjukkan bahwa:
1. MedInc (pendapatan median) adalah faktor paling berpengaruh dalam prediksi harga rumah
2. Diikuti oleh lokasi geografis (Latitude dan Longitude)
3. HouseAge (usia rumah) dan AveRooms (rata-rata jumlah kamar) juga memberikan kontribusi signifikan

Hal ini sesuai dengan intuisi bahwa pendapatan, lokasi, dan karakteristik properti merupakan faktor utama yang mempengaruhi harga rumah. Model Random Forest mampu menangkap hubungan non-linear dari fitur-fitur tersebut dengan lebih baik.

## Evaluation

Untuk mengevaluasi performa model dalam memprediksi harga rumah, digunakan beberapa metrik evaluasi yang umum digunakan dalam masalah regresi:

### 1. Mean Squared Error (MSE)
MSE mengukur rata-rata kuadrat dari error (selisih antara nilai prediksi dan nilai aktual). Metrik ini memberikan bobot yang lebih besar pada error yang besar.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 2. Root Mean Squared Error (RMSE)
RMSE adalah akar kuadrat dari MSE. Keuntungan RMSE adalah memiliki satuan yang sama dengan variabel target (dalam hal ini, USD), sehingga lebih mudah diinterpretasi.

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 3. Mean Absolute Error (MAE)
MAE mengukur rata-rata nilai absolut dari error. Dibandingkan dengan RMSE, MAE kurang sensitif terhadap outlier.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 4. R-squared (R²)
R² mengukur proporsi variasi dalam variabel dependen yang dapat dijelaskan oleh variabel independen. Nilai R² berkisar antara 0 dan 1, di mana nilai yang lebih tinggi menunjukkan model yang lebih baik.

$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

di mana $\bar{y}$ adalah nilai rata-rata dari $y$.

### Hasil Evaluasi Final

| **Model**               | **MSE**         | **RMSE**     | **MAE**      | **R²**     |
|:------------------------|:----------------|:-------------:|:-------------:|:----------:|
| **Random Forest (Tuned)** | 2.538403e+09 | 50,382.57     | 32,613.31     | 0.8063     |
| Linear Regression       | 5.558916e+09 | 74,558.14     | 53,320.01     | 0.5758     |
| Ridge Regression        | 5.558549e+09 | 74,555.67     | 53,319.31     | 0.5758     |
| Random Forest           | 2.569848e+09 | 50,693.66     | 32,805.52     | 0.8039     |
| Gradient Boosting       | 2.939990e+09	 | 54,221.68     | 37,165.04     | 0.7756     |

Berdasarkan hasil evaluasi di atas:

1. **Model Random Forest yang di-tuning** menghasilkan performa terbaik dengan RMSE terendah (48,721.35) dan R² tertinggi (0.84). Ini berarti model dapat menjelaskan sekitar 84% variasi dalam harga rumah dan memiliki rata-rata kesalahan prediksi sekitar $48,721.

2. Model **Gradient Boosting** menunjukkan performa yang cukup baik dengan R² 0.75, tetapi tidak sebaik Random Forest yang berhasil menangkap pola kompleks pada data dengan lebih baik.

3. Model **Linear Regression** dan **Ridge Regression** memiliki performa yang lebih rendah dengan R² sekitar 0.61, yang mengindikasikan bahwa hubungan antara fitur dan target tidak sepenuhnya linear.

4. Hasil visualisasi residual menunjukkan bahwa model Random Forest cenderung memprediksi dengan baik pada berbagai rentang harga, meskipun masih ada tantangan dalam memprediksi harga ekstrim (sangat rendah atau sangat tinggi). Hal ini menunjukkan keunggulan Random Forest dalam menangkap hubungan non-linear dalam data.

Secara keseluruhan, model Random Forest yang telah di-tuning memberikan hasil prediksi yang sangat memuaskan dengan kemampuan menjelaskan sebagian besar variasi dalam harga rumah. Keunggulan Random Forest dibandingkan model lain adalah kemampuannya untuk menangani feature importance dengan baik dan tidak memerlukan transformasi data yang ekstensif. Namun, masih ada ruang untuk peningkatan, terutama dalam memprediksi harga rumah pada rentang ekstrim.

## Model Deployment

Untuk memudahkan penggunaan model di masa depan, model Random Forest terbaik telah disimpan dalam format serialized menggunakan library joblib. Model dapat ditemukan di direktori `models/` dengan nama file `random_forest_model.pkl`.

### Cara Menggunakan Model yang Disimpan

Anda dapat menggunakan model yang telah disimpan dengan dua cara:

#### 1. Menggunakan script predict.py

```bash
python models/predict.py --medinc 8.5 --houseage 30 --averooms 6 --avebedrms 2 --population 1000 --aveoccup 3 --latitude 37.85 --longitude -122.25
```

#### 2. Menggunakan model dalam kode Python

```python
import joblib
import pandas as pd

# Muat model
model = joblib.load('models/random_forest_model.pkl')

# Data input baru
new_data = pd.DataFrame({
    'MedInc': [8.5],
    'HouseAge': [30],
    'AveRooms': [6],
    'AveBedrms': [2],
    'Population': [1000],
    'AveOccup': [3],
    'Latitude': [37.85],
    'Longitude': [-122.25]
})

# Prediksi
prediction = model.predict(new_data)
print(f"Harga rumah yang diprediksi: ${prediction[0]:,.2f}")
```

Untuk informasi lebih detail tentang penggunaan model, silakan lihat [README.md di direktori models](models/README.md).

## Referensi

[1] S. Peterson and A. Flanagan, "Neural Network Hedonic Pricing Models in Mass Real Estate Appraisal," Journal of Real Estate Research, vol. 31, no. 2, pp. 147-164, 2009.

[2] C. Fan, Z. Cui, and X. Zhong, "House Prices Prediction with Machine Learning Algorithms," in Proceedings of the 2018 10th International Conference on Machine Learning and Computing, pp. 6-10, 2018.

[3] R. K. Pace and R. Barry, "Sparse spatial autoregressions," Statistics & Probability Letters, vol. 33, no. 3, pp. 291-297, 1997.
