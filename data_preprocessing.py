import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
# Ganti 'flood_network_data.csv' dengan path ke file dataset Anda
data = pd.read_csv('network_traffic_data_extended.csv')

# Tampilkan informasi awal dataset
print("Informasi dataset:")
print(data.info())
print("\nStatistik deskriptif:")
print(data.describe())

# 1. Mengatasi nilai yang hilang
# Ganti nilai yang hilang dengan median atau metode lainnya
data.fillna(data.median(numeric_only=True), inplace=True)

# 2. Mengatasi nilai duplikat
data = data.drop_duplicates()

# 3. Encoding variabel kategorikal (jika ada)
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

# 4. Feature Scaling (Standardization)
# Identifikasi fitur numerik
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# 5. Membagi data menjadi fitur (X) dan label (y)
# Asumsikan kolom 'Label' adalah target yang akan diprediksi
X = data.drop('Label', axis=1)
y = data['Label']

# 6. Membagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tampilkan informasi data hasil preprocessing
print("\nShape data training:", X_train.shape)
print("Shape data testing:", X_test.shape)

# Simpan data hasil preprocessing jika diperlukan
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train_preprocessed.csv', index=False)
y_test.to_csv('y_test_preprocessed.csv', index=False)

print("\nData preprocessing selesai.")