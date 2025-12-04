from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="House Price Prediction System")

# Permissions for Frontend access (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Model
model = joblib.load("turkiye_house_price_model.pkl")
print("Model loaded successfully.") if model else print("Failed to load model.")  

# 3. Load Scaler
scaler = joblib.load("standard_scaler.pkl")
print("Scaler loaded successfully.") if scaler else print("Failed to load scaler.")
 
# 4. Input Data Schema
class HouseFeatures(BaseModel):
    m2: float
    room_count: int
    bathroom_count: int
    total_floors: int
    building_age: int
    floor_no: int

# 5. Prediction Endpoint
@app.post("/predict")
def predict_price(data: HouseFeatures):
    
    if not model:
        return {"error": "Model could not be loaded, cannot predict."}

    # A. Prepare data (Order must match training data)
    features = [
        data.m2,
        data.oda_sayisi,
        data.bulundugu_kat,
        data.eşya_durumu,
        data.binanın_yaşı,
        data.ısıtma_tipi,
        data.sehir,
        data.binanın_kat_sayısı,
        data.kullanım_durumu,
        data.yatırıma_uygunluk,
        data.takas,
        data.tapu_durumu,
        data.banyo_sayısı
    ]

    # B. Convert to NumPy Array
    input_array = np.array([features])

    # C. Scale Data (If scaler exists)
    if scaler:
        input_array = scaler.transform(input_array)

    # D. Make Prediction
    try:
        log_prediction = model.predict(input_array)
        actual_price = np.expm1(log_prediction[0]) # Reverse log transformation
        
        return {
            "estimated_price": round(float(actual_price), 2),
            "currency": "TL"
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# To run: uvicorn main:app --reload