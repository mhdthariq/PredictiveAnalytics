#!/usr/bin/env python3
"""
Script untuk menggunakan model Random Forest yang telah dilatih
untuk memprediksi harga rumah berdasarkan input pengguna.

Penggunaan:
    python predict.py --medinc 8.5 --houseage 30 --averooms 6 --avebedrms 2 --population 1000 --aveoccup 3 --latitude 37.85 --longitude -122.25

Author: [Muhammad Thariq Arya Putra Sembiring]
"""

import argparse
import joblib
import pandas as pd
import os
import numpy as np
from pathlib import Path

# Mendefinisikan path ke model
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"

def load_model():
    """
    Load model yang telah dilatih
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model

def predict_price(model, features):
    """
    Memprediksi harga rumah berdasarkan fitur

    Args:
        model: Model yang telah dilatih
        features: Dictionary berisi nilai fitur

    Returns:
        float: Harga rumah yang diprediksi
    """
    # Mengkonversi input ke DataFrame
    df = pd.DataFrame([features])

    # Melakukan prediksi
    prediction = model.predict(df)[0]

    return prediction

def parse_arguments():
    """
    Parse argumen command line
    """
    parser = argparse.ArgumentParser(description="Prediksi harga rumah di California")

    parser.add_argument("--medinc", type=float, required=True, help="Median pendapatan rumah tangga (dalam USD 10,000)")
    parser.add_argument("--houseage", type=float, required=True, help="Median usia rumah dalam blok")
    parser.add_argument("--averooms", type=float, required=True, help="Rata-rata jumlah kamar per rumah tangga")
    parser.add_argument("--avebedrms", type=float, required=True, help="Rata-rata jumlah kamar tidur per rumah tangga")
    parser.add_argument("--population", type=float, required=True, help="Populasi blok")
    parser.add_argument("--aveoccup", type=float, required=True, help="Rata-rata jumlah anggota rumah tangga")
    parser.add_argument("--latitude", type=float, required=True, help="Garis lintang blok")
    parser.add_argument("--longitude", type=float, required=True, help="Garis bujur blok")

    return parser.parse_args()

def main():
    """
    Fungsi utama
    """
    # Parse argumen
    args = parse_arguments()

    # Menyiapkan fitur
    features = {
        'MedInc': args.medinc,
        'HouseAge': args.houseage,
        'AveRooms': args.averooms,
        'AveBedrms': args.avebedrms,
        'Population': args.population,
        'AveOccup': args.aveoccup,
        'Latitude': args.latitude,
        'Longitude': args.longitude
    }

    # Memuat model
    try:
        model = load_model()

        # Melakukan prediksi
        predicted_price = predict_price(model, features)

        print(f"\nFitur Input:")
        for k, v in features.items():
            print(f"{k}: {v}")

        print(f"\nHarga Rumah yang Diprediksi: ${predicted_price:,.2f}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
