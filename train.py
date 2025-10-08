import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

# Load and preprocess data
def load_data():
    df = pd.read_csv("daily_food_nutrition_dataset.csv")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    return df

df = load_data()

# Define nutrient columns
nutrient_cols = ["Calories (kcal)", "Protein (g)", "Carbohydrates (g)", "Fat (g)", 
                 "Fiber (g)", "Sugars (g)", "Sodium (mg)", "Cholesterol (mg)"]

# Scale the nutritional features
scaler = StandardScaler()
X = scaler.fit_transform(df[nutrient_cols])

# Train a Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
knn_model.fit(X)

# Save the model and scaler
joblib.dump(knn_model, "diet_knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Nearest Neighbor Model Saved Successfully!")
