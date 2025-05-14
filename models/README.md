# Model Prediksi Harga Rumah California

Direktori ini berisi model machine learning untuk prediksi harga rumah di California yang telah dilatih dan dioptimasi.

## File yang Tersedia

- `random_forest_model.pkl`: Model Random Forest yang telah dilatih dan dioptimasi
- `preprocessor.pkl`: Preprocessor yang digunakan untuk mempersiapkan data sebelum prediksi
- `predict.py`: Script Python untuk menggunakan model untuk prediksi

## Cara Menggunakan Model

### Metode 1: Menggunakan script predict.py

Anda dapat menggunakan script `predict.py` untuk memprediksi harga rumah berdasarkan input fitur.

```bash
python predict.py --medinc 8.5 --houseage 30 --averooms 6 --avebedrms 2 --population 1000 --aveoccup 3 --latitude 37.85 --longitude -122.25
```

### Metode 2: Menggunakan model langsung dalam kode Python

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

## Deskripsi Fitur

Berikut adalah deskripsi dari setiap fitur yang digunakan dalam model:

| **Fitur** | **Deskripsi** |
|-------|-----------|
| MedInc | Median pendapatan rumah tangga dalam blok (dalam USD 10,000) |
| HouseAge | Median usia rumah dalam blok |
| AveRooms | Rata-rata jumlah kamar per rumah tangga |
| AveBedrms | Rata-rata jumlah kamar tidur per rumah tangga |
| Population | Populasi blok |
| AveOccup | Rata-rata jumlah anggota rumah tangga |
| Latitude | Garis lintang blok |
| Longitude | Garis bujur blok |

## Performa Model

Model Random Forest yang digunakan memiliki performa sebagai berikut:

- RMSE (Root Mean Squared Error): ~48,721 USD
- R² (Coefficient of Determination): ~0.84

Ini berarti model dapat menjelaskan sekitar 84% variasi dalam harga rumah dengan rata-rata kesalahan prediksi sekitar $48,721.

## Feature Importance

Berdasarkan analisis feature importance, berikut adalah fitur-fitur yang paling berpengaruh dalam prediksi harga rumah (secara berurutan dari yang paling penting):

1. MedInc (pendapatan median)
2. Latitude & Longitude (lokasi)
3. HouseAge (usia rumah)
4. AveRooms (rata-rata jumlah kamar)
