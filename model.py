# model.py (FINAL VERSION, USE ALL DATA)

import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd
import joblib

# 🔥 Step 1: Set Kaggle API credentials first
os.environ['KAGGLE_USERNAME'] = 'nik2polistico'
os.environ['KAGGLE_KEY'] = '79275680b3fb3ebcb872b386b463f54b'

# 🔥 Step 2: Import Kaggle API AFTER setting credentials
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 3: Authenticate and Download the Dataset
api = KaggleApi()
api.authenticate()

dataset_name = "nirajspatil/indiet-dataset"
download_path = "kaggle_datasets"
os.makedirs(download_path, exist_ok=True)

print("⬇️ Downloading dataset...")
api.dataset_download_files(dataset_name, path=download_path, unzip=True)

path = download_path
print("✅ Dataset downloaded to:", path)

# Step 4: Load Dataset
dataset_path = os.path.join(path, "InDiet_Dataset.csv")  # Use the correct Kaggle file name!
if not os.path.exists(dataset_path):
    print(f"❌ Dataset not found at {dataset_path}")
    exit()

df = pd.read_csv(dataset_path)

print("🔎 Dataset Preview:")
print(df.head())

# Step 5: Prepare Features and Labels
filtered_df = df.dropna(subset=[
    'food_group_nin',
    'energy_category',
    'nutrient_score',
    'health_score',
    'diversity_score',
    'food_name'
])

# Label Encoding
label_encoder_pref = LabelEncoder()
label_encoder_energy = LabelEncoder()
label_encoder_meal = LabelEncoder()

filtered_df['food_group_encoded'] = label_encoder_pref.fit_transform(filtered_df['food_group_nin'])
filtered_df['energy_category_encoded'] = label_encoder_energy.fit_transform(filtered_df['energy_category'])

# Normalize numeric features
scaler = MinMaxScaler()
filtered_df[['nutrient_score', 'health_score', 'diversity_score']] = scaler.fit_transform(
    filtered_df[['nutrient_score', 'health_score', 'diversity_score']]
)

# Step 6: Create Training Data
X = filtered_df[['food_group_encoded', 'energy_category_encoded', 'nutrient_score', 'health_score', 'diversity_score']].values
y = label_encoder_meal.fit_transform(filtered_df['food_name'])

print("✅ Features and Labels ready:")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Step 7: Build the Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(5,)),  # 5 input features
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder_meal.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the Model
print("🏋️‍♂️ Training the model...")
model.fit(X, y, epochs=100, verbose=2)

# Step 9: Save the Model and Encoders
model.save('meal_recommender.h5')
print("✅ Model saved as meal_recommender.h5")

joblib.dump(label_encoder_pref, 'label_encoder_pref.pkl')
joblib.dump(label_encoder_energy, 'label_encoder_energy.pkl')
joblib.dump(label_encoder_meal, 'label_encoder_meal.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
print("✅ LabelEncoders and Scaler saved successfully!")
