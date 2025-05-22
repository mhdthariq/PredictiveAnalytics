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
Permasalahan utama yang sedang dihadapi dalam industri real estate California adalah:
1. Penentuan harga rumah yang tidak akurat menyebabkan inefisiensi pasar, di mana penjual mungkin menetapkan harga terlalu tinggi sehingga properti sulit terjual, atau terlalu rendah sehingga kehilangan potensi keuntungan. Menurut laporan National Association of Realtors, kesalahan penentuan harga dapat memperpanjang waktu properti di pasar hingga 30% lebih lama.
2. Investor properti dan pembeli rumah menghadapi kesulitan dalam mengidentifikasi nilai properti yang sebenarnya berdasarkan berbagai faktor seperti lokasi, ukuran, dan karakteristik demografis, yang mengakibatkan keputusan investasi yang kurang optimal dan potensial kerugian finansial.
3. Penilaian properti tradisional yang dilakukan penilai membutuhkan waktu lama, mahal, dan rentan terhadap inkonsistensi serta bias subjektif, yang menghambat dinamika pasar properti dan meningkatkan biaya transaksi secara keseluruhan.

Jika permasalahan ini tidak segera diatasi, konsekuensinya dapat berupa:
- Stagnansi pasar real estate dengan volume penjualan yang menurun
- Peningkatan risiko finansial bagi pembeli, penjual, dan investor
- Ketidakpercayaan terhadap penilaian harga properti yang dapat merugikan semua pemangku kepentingan
- Kesulitan bagi institusi keuangan dalam mengevaluasi risiko kredit untuk pinjaman properti

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

Dataset ini bisa diakses melalui:
1. Library scikit-learn: `from sklearn.datasets import fetch_california_housing`
2. StatLib Repository (sumber asli): [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
3. Kaggle: [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

> **Catatan Penting:** Dalam proyek ini, saya menggunakan dataset California Housing dari scikit-learn yang memiliki 8 fitur numerik (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude). Dataset dari Kaggle (camnugent/california-housing-prices) mengandung fitur tambahan `ocean_proximity` yang merupakan fitur kategorikal yang mengindikasikan kedekatan properti dengan lautan. Versi scikit-learn telah diproses sebelumnya dan tidak menyertakan fitur kategorikal ini. Kedua dataset berasal dari sumber yang sama (Sensus California tahun 1990) namun telah mengalami pra-pemrosesan yang berbeda.

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

Pada tahap modeling, saya menerapkan empat algoritma machine learning berbeda untuk memprediksi harga rumah. Berikut adalah penjelasan detail tentang tiap algoritma, karakteristiknya, dan proses pemodelan yang dilakukan:

### 1. Linear Regression
Linear Regression merupakan algoritma baseline yang mengasumsikan hubungan linear antara fitur dan target.

**Karakteristik**: 
- Algoritma ini bekerja dengan mencari garis linear terbaik yang dapat meminimalkan jumlah kuadrat error antara nilai aktual dan prediksi.
- Sangat cocok untuk kasus di mana hubungan antara fitur dan target cenderung linear.
- Model ini memiliki interpretabilitas yang tinggi karena memberikan koefisien yang menunjukkan kontribusi masing-masing fitur.

**Parameter yang digunakan**:
- Tidak ada parameter khusus yang di-tuning pada model ini, menggunakan konfigurasi default dari scikit-learn.
- Alasan: Sebagai model baseline, tujuannya adalah untuk memberikan perbandingan dasar terhadap model yang lebih kompleks.

### 2. Ridge Regression
Ridge Regression adalah varian dari Linear Regression dengan regularisasi L2 untuk mengatasi multicollinearity.

**Karakteristik**:
- Menambahkan regularisasi L2 (penalti pada koefisien yang besar) untuk mengurangi overfitting.
- Sangat efektif ketika terdapat korelasi tinggi antar fitur (multicollinearity).
- Cenderung menghasilkan model yang lebih stabil dibandingkan Linear Regression standar.

**Parameter yang digunakan**:
- `alpha=1.0`: Parameter regularisasi yang mengontrol seberapa besar penalti terhadap koefisien.
- Alasan: Nilai alpha=1.0 adalah default yang memberikan keseimbangan antara bias dan varians. Nilai ini dipilih sebagai titik awal karena memberikan tingkat regularisasi moderat yang cocok untuk dataset perumahan yang umumnya memiliki beberapa fitur berkorelasi.

### 3. Random Forest Regression
Random Forest adalah algoritma ensemble yang menggunakan multiple decision trees.

**Karakteristik**:
- Bekerja dengan membangun banyak decision tree dan menggabungkan hasilnya melalui averaging.
- Setiap tree dilatih pada subset data yang berbeda (bootstrapping) dan menggunakan subset fitur yang dipilih secara acak.
- Sangat efektif dalam menangkap hubungan non-linear dan interaksi kompleks antar fitur.
- Robust terhadap outlier dan noise dalam data.

**Parameter awal yang digunakan**:
- `n_estimators=100`: Jumlah decision tree dalam forest.
- `random_state=42`: Untuk memastikan reprodusibilitas hasil.

**Proses hyperparameter tuning**:
Setelah evaluasi model awal, Random Forest dipilih untuk tuning karena menunjukkan performa terbaik. Tuning dilakukan menggunakan GridSearchCV dengan 5-fold cross-validation untuk mengevaluasi kombinasi parameter berikut:
- `n_estimators`: [100, 200, 300] - Jumlah tree dalam forest
- `max_depth`: [10, 20, 30, None] - Kedalaman maksimum setiap tree
- `min_samples_split`: [2, 5, 10] - Minimum sampel yang diperlukan untuk split node
- `min_samples_leaf`: [1, 2, 4] - Minimum sampel yang diperlukan dalam leaf node
- `bootstrap`: [True, False] - Apakah menggunakan bootstrapping

Proses tuning memerlukan waktu lebih dari 5 jam karena keterbatasan hardware, dengan cross-validation 5-fold, yang berarti model dilatih dan dievaluasi sebanyak 5 × (3×4×3×3×2) = 1,080 kali untuk menemukan kombinasi parameter terbaik.

**Parameter optimal hasil tuning**:
- `bootstrap=True`: Menggunakan bootstrapping untuk meningkatkan diversitas model.
- `max_depth=30`: Kedalaman tree yang cukup dalam untuk menangkap pola kompleks tanpa overfitting berlebihan.
- `min_samples_leaf=2`: Mensyaratkan minimal 2 sampel di setiap leaf node untuk mengurangi overfitting.
- `min_samples_split=2`: Nilai default yang memungkinkan splitting node dengan minimal 2 sampel.
- `n_estimators=300`: Jumlah tree yang lebih banyak untuk meningkatkan stabilitas prediksi.

Alasan pemilihan parameter ini: Parameter-parameter tersebut menghasilkan RMSE terendah pada cross-validation, menunjukkan kemampuan generalisasi terbaik. Peningkatan jumlah estimator menjadi 300 memungkinkan model untuk lebih baik menangkap pola dalam data, sementara max_depth=30 memberikan fleksibilitas yang cukup tanpa overfitting berlebihan.

### 4. Gradient Boosting Regression
Gradient Boosting adalah algoritma ensemble yang membangun model secara sequential.

**Karakteristik**:
- Bekerja dengan membangun tree secara berurutan, di mana setiap tree baru berfokus memperbaiki kesalahan tree sebelumnya.
- Menggunakan gradient descent untuk meminimalkan fungsi loss.
- Sangat powerful untuk menangkap pola kompleks dalam data.
- Cenderung memiliki performa yang sangat baik, terutama setelah tuning.

**Parameter yang digunakan**:
- `n_estimators=100`: Jumlah tahapan boosting (jumlah tree).
- `random_state=42`: Untuk memastikan reprodusibilitas hasil.
- Alasan: Parameter default ini memberikan keseimbangan yang baik antara performa dan waktu komputasi. Jumlah estimator sebanyak 100 cukup untuk model awal sebelum melakukan tuning lebih lanjut.

### Pemilihan Model

Random Forest dipilih sebagai model utama untuk di-tuning lebih lanjut berdasarkan beberapa pertimbangan:

1. Performa awal yang unggul dibandingkan model lain (RMSE lebih rendah dan R² lebih tinggi).
2. Kemampuan menangkap hubungan non-linear yang terlihat pada eksplorasi data antara fitur dan target.
3. Robust terhadap outlier yang terdeteksi pada variabel target (harga rumah).
4. Feature importance yang informatif untuk interpretasi bisnis.
5. Keseimbangan yang baik antara akurasi dan kecepatan komputasi.

Dibandingkan dengan Gradient Boosting yang juga menunjukkan performa baik, Random Forest dipilih karena lebih cepat dilatih, lebih mudah di-tuning, dan memberikan hasil yang lebih stabil bahkan dengan parameter default.

## Evaluation

Untuk mengevaluasi performa model dalam memprediksi harga rumah, saya menggunakan beberapa metrik evaluasi yang umum digunakan dalam masalah regresi:

### Metrik Evaluasi yang Digunakan

1. **Mean Squared Error (MSE)**
MSE mengukur rata-rata kuadrat dari error (selisih antara nilai prediksi dan nilai aktual). Metrik ini memberikan bobot yang lebih besar pada error yang besar.

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

2. **Root Mean Squared Error (RMSE)**
RMSE adalah akar kuadrat dari MSE. Keuntungan RMSE adalah memiliki satuan yang sama dengan variabel target (dalam hal ini, USD), sehingga lebih mudah diinterpretasi.

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

3. **Mean Absolute Error (MAE)**
MAE mengukur rata-rata nilai absolut dari error. Dibandingkan dengan RMSE, MAE kurang sensitif terhadap outlier.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

4. **R-squared (R²)**
R² mengukur proporsi variasi dalam variabel dependen yang dapat dijelaskan oleh variabel independen. Nilai R² berkisar antara 0 dan 1, di mana nilai yang lebih tinggi menunjukkan model yang lebih baik.

```math
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
```

di mana $\bar{y}$ adalah nilai rata-rata dari $y$.

### Hasil Evaluasi Final

| **Model**               | **MSE**         | **RMSE**     | **MAE**      | **R²**     |
|:------------------------|:----------------|:-------------:|:-------------:|:----------:|
| **Random Forest (Tuned)** | 2.538403e+09 | 50,382.57     | 32,613.31     | 0.8063     |
| Linear Regression       | 5.558916e+09 | 74,558.14     | 53,320.01     | 0.5758     |
| Ridge Regression        | 5.558549e+09 | 74,555.67     | 53,319.31     | 0.5758     |
| Random Forest           | 2.569848e+09 | 50,693.66     | 32,805.52     | 0.8039     |
| Gradient Boosting       | 2.939990e+09	 | 54,221.68     | 37,165.04     | 0.7756     |

### Keterkaitan dengan Problem Statement dan Business Understanding

Berdasarkan hasil evaluasi di atas, kita dapat meninjau kembali problem statement dan goals yang telah dirumuskan sebelumnya:

**1. Problem Statement: Inefisiensi pasar karena penentuan harga yang tidak akurat**
- Model Random Forest yang dihasilkan memiliki MAE sebesar $32,613.31. Artinya, secara rata-rata model hanya menyimpang sekitar $32,613 dari harga aktual.
- Dalam konteks pasar perumahan California, di mana median harga rumah sekitar $200,000, ini merepresentasikan error sekitar 16%.
- Dibandingkan dengan metode penilaian tradisional yang dapat memiliki error hingga 20-25%, model ini memberikan peningkatan signifikan dalam akurasi penentuan harga.
- Dampak bisnis: Penjual dapat menetapkan harga yang lebih tepat, mengurangi waktu properti di pasar, dan meningkatkan efisiensi transaksi secara keseluruhan.

**2. Problem Statement: Kesulitan investor dan pembeli dalam mengidentifikasi nilai properti**
- Dengan R² sebesar 0.8063, model dapat menjelaskan sekitar 80.6% variasi dalam harga rumah berdasarkan fitur-fitur yang diberikan.
- Feature importance yang dihasilkan model menunjukkan bahwa faktor pendapatan median (MedInc) dan lokasi (Longitude, Latitude) adalah prediktor terkuat, memberikan wawasan berharga bagi investor.
- Dampak bisnis: Investor kini memiliki alat untuk melakukan valuasi properti yang lebih objektif dan berbasis data, mengurangi risiko investasi yang tidak menguntungkan.

**3. Problem Statement: Penilaian properti tradisional yang lambat, mahal, dan subjektif**
- Model machine learning yang dikembangkan dapat memberikan prediksi instan, mengurangi waktu dan biaya yang diperlukan untuk penilaian properti.
- Dengan error prediksi yang relatif kecil (RMSE $50,382.57), model ini menawarkan alternatif yang efisien dan objektif dibandingkan penilaian manual.
- Dampak bisnis: Lembaga keuangan dapat mengintegrasikan model untuk pre-screening penilaian kredit properti, mempercepat proses persetujuan pinjaman, dan mengurangi biaya operasional.

### Pencapaian Goals

**1. Goal: Mengidentifikasi faktor-faktor utama yang mempengaruhi harga rumah di California**
- ✅ Tercapai: Analisis feature importance dari model Random Forest mengungkapkan bahwa MedInc (pendapatan median), Longitude dan Latitude (lokasi geografis), dan HouseAge (usia rumah) adalah faktor paling berpengaruh.
- Wawasan ini memberikan pemahaman yang lebih dalam tentang dinamika pasar perumahan California dan dapat menginformasikan strategi pengembangan properti dan investasi.

**2. Goal: Mengembangkan model machine learning dengan tingkat error minimal**
- ✅ Tercapai: Model Random Forest yang telah di-tuning mencapai RMSE sekitar $50,382.57, yang relatif kecil dibandingkan dengan rentang harga rumah di dataset ($14,999 hingga $500,001).
- Performa ini melebihi model baseline (Linear Regression) dengan peningkatan akurasi sekitar 32%.

**3. Goal: Mengevaluasi dan membandingkan berbagai algoritma machine learning**
- ✅ Tercapai: Empat algoritma berbeda (Linear Regression, Ridge Regression, Random Forest, dan Gradient Boosting) telah dievaluasi dan dibandingkan secara komprehensif menggunakan metrik yang relevan.
- Random Forest terbukti menjadi model terbaik untuk kasus ini, dengan keunggulan signifikan dibandingkan model linear.

### Dampak Solution Statement

**1. Solution: Eksplorasi data dan analisis**
- ✅ Efektif: Eksplorasi data mengungkapkan hubungan non-linear antara beberapa fitur dan harga rumah, mengarahkan pemilihan model yang tepat (ensemble methods).
- Visualisasi korelasi membantu mengidentifikasi fitur yang paling relevan, memvalidasi intuisi bisnis tentang pentingnya lokasi dan karakteristik demografi.

**2. Solution: Pengembangan beberapa model regresi**
- ✅ Efektif: Pendekatan multi-model memungkinkan perbandingan komprehensif, dengan Random Forest menunjukkan performa terbaik, sesuai dengan karakteristik data yang memiliki pola non-linear.
- Pipeline preprocessing yang dikembangkan memastikan konsistensi dan menghindari data leakage, meningkatkan reliabilitas model.

**3. Solution: Hyperparameter tuning pada model terbaik**
- ✅ Efektif: Tuning meningkatkan performa Random Forest (dari R² 0.8039 menjadi 0.8063), meskipun peningkatannya moderat, ini menunjukkan bahwa model default sudah cukup baik untuk dataset ini.
- Proses tuning menghasilkan wawasan tentang parameter optimal, yang dapat digunakan untuk pengembangan model lebih lanjut.

### Kesimpulan dan Rekomendasi

Model prediksi harga rumah yang dikembangkan berhasil mengatasi tiga problem statement utama dengan menyediakan alat prediksi yang akurat, cepat, dan berbasis data. Dengan akurasi 80.6% (R²), model ini dapat memberikan estimasi harga yang lebih baik dibandingkan metode tradisional, membantu mengurangi inefisiensi pasar, dan mendukung pengambilan keputusan investasi yang lebih baik.

Beberapa rekomendasi untuk implementasi dan pengembangan lebih lanjut:

1. **Implementasi sistem prediksi real-time**: Mengintegrasikan model ke dalam platform web atau aplikasi mobile untuk memberikan estimasi harga instan kepada pengguna.

2. **Pengayaan data**: Menambahkan fitur-fitur tambahan seperti jarak ke fasilitas publik, tingkat kejahatan, dan kualitas sekolah dapat meningkatkan akurasi model.

3. **Segmentasi model**: Mengembangkan model terpisah untuk segmen pasar yang berbeda (rumah mewah vs terjangkau, perkotaan vs pinggiran) untuk meningkatkan akurasi prediksi di setiap segmen.

4. **Monitoring dan pembaruan berkala**: Pasar properti bersifat dinamis, sehingga model harus diperbarui secara berkala dengan data terbaru untuk mempertahankan akurasi prediksi.

Dengan model prediksi harga rumah yang akurat, semua pemangku kepentingan dalam industri real estate California dapat membuat keputusan yang lebih informasi, mengurangi risiko, dan pada akhirnya menciptakan pasar properti yang lebih efisien dan transparan.

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

[4] R. K. Pace and R. Barry, "California Housing Data," StatLib Repository, (https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html), 1997. Dataset yang sama tersedia melalui scikit-learn dengan nama `fetch_california_housing()` dan Kaggle dengan versi yang lebih lengkap termasuk fitur `ocean_proximity`.
