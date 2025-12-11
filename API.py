from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import re

# 1. Kaydedilen Modelleri Yükle
# Loads models
model = joblib.load("turkiye_house_price_model.pkl")
scaler = joblib.load("standard_scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

app = FastAPI()
origins = [
    
    "https://house-pricing-prediction-xggx.onrender.com",
]

# CORS uyumluluğunun eklenmesi
# Adding CORS compatiability
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              # Allows the frontend to access the API
    allow_credentials=True,             # Allows cookies/authorization headers
    allow_methods=["*"],                # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],                # Allows all headers
)

# 2. Veri Modeli
# Data models
class HouseData(BaseModel):
    Net_Metrekare: float
    Brut_Metrekare: float
    Oda_Sayisi: float
    Bulundugu_Kat: str
    Esya_Durumu: str
    Binanin_Yasi: str
    Isitma_Tipi: str
    Sehir: str
    Binanin_Kat_Sayisi: float
    Kullanim_Durumu: str
    Yatirima_Uygunluk: str
    Takas: str
    Tapu_Durumu: str
    Banyo_Sayisi: float

# --- Temizlik Fonksiyonları ---
# --- Cleaning functions ---
def clean_age(veri):
    sayilar = [float(s) for s in re.findall(r'\d+', str(veri))]
    if len(sayilar) == 2:
        return sum(sayilar)/2
    elif len(sayilar) == 1:
        return sayilar[0]
    return 0.0

def clean_floor(data):
    yazi = str(data).lower()
    if any(k in yazi for k in ["giriş","zemin","bahçe"]):
        return 0.0
    sayilar = re.findall(r'\d+', yazi)
    return float(sayilar[0]) if sayilar else 0.0

# Root uç noktası
# Root endpoint
@app.get("/")
def read_root():
    return {"status": "ok", "message": "House Price Prediction API is operational."}
# --- Tahmin Endpoint ---
# --- Prediction endpoint ---
@app.post("/predict")
def predict_price(data: HouseData):
    try:
        # DF'e çevir
        # Changing to dataframe
        df = pd.DataFrame([data.dict()])
        
        # Sütun adlarını model ile uyumlu yap
        # Making column names compatiable with model's expected column names
        df.rename(columns={
            'Brut_Metrekare': 'Brüt_Metrekare',
            'Oda_Sayisi': 'Oda_Sayısı',
            'Bulundugu_Kat': 'Bulunduğu_Kat',
            'Esya_Durumu': 'Eşya_Durumu',
            'Binanin_Yasi': 'Binanın_Yaşı',
            'Isitma_Tipi': 'Isıtma_Tipi',
            'Sehir': 'Şehir',
            'Binanin_Kat_Sayisi': 'Binanın_Kat_Sayısı',
            'Kullanim_Durumu': 'Kullanım_Durumu',
            'Yatirima_Uygunluk': 'Yatırıma_Uygunluk',
            'Takas': 'Takas',
            'Tapu_Durumu': 'Tapu_Durumu',
            'Banyo_Sayisi': 'Banyo_Sayısı'
        }, inplace=True)

        # --- Özellik Mühendisliği ---
        # --- Feature Engineering ---
        df['Oda_Buyuklugu'] = df['Net_Metrekare'] / df['Oda_Sayısı']
        df['Banyo_Orani'] = df['Banyo_Sayısı'] / df['Oda_Sayısı']
        
        # --- Temizlik ---
        # --- Cleaning ---
        df['Binanın_Yaşı'] = df['Binanın_Yaşı'].apply(clean_age)
        df['Bulunduğu_Kat'] = df['Bulunduğu_Kat'].apply(clean_floor)
        df.loc[df['Banyo_Sayısı'] > 5, 'Banyo_Sayısı'] = 5
        
        # --- Label Encoding ---
        label_cols = ['Eşya_Durumu','Takas','Yatırıma_Uygunluk']
        for col in label_cols:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except:
                df[col] = 0
        
        # --- One-hot Encoding ---
        ohe_cols = ['Isıtma_Tipi','Kullanım_Durumu','Tapu_Durumu']
        df = pd.get_dummies(df, columns=ohe_cols)
        
        # Özelliklerin sayısal olmasını sağlamak
        # Ensuring features are numeric
                
        # 1. Label Encoded (Should be int/float)
        label_cols = ['Eşya_Durumu','Takas','Yatırıma_Uygunluk']
        for col in label_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # 2. Cleaned Features (Should be float)
        cleaned_cols = ['Binanın_Yaşı', 'Bulunduğu_Kat']
        for col in cleaned_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
        # 3. Target Encoded (Already done, but keeping the check)
        df['Şehir'] = pd.to_numeric(df['Şehir'], errors='coerce')
                
        # Filling any NaN values created by 'coerce' (from failed conversion) with 0
        df.fillna(0, inplace=True)
        
        # --- Model kolonları ile hizala ---
        # --- Aligning with model columns ---
                
        cols_to_drop = [
            'Bulunduğu_Kat', # Zaten clean_floor ile numeric oldu
            'Eşya_Durumu',
            'Binanın_Yaşı',
            'Isıtma_Tipi',
            'Kullanım_Durumu',
            'Yatırıma_Uygunluk',
            'Takas',
            'Tapu_Durumu'
        ]
                
                
        original_input_cols = [
            'Bulunduğu_Kat', 'Eşya_Durumu', 'Binanın_Yaşı', 'Isıtma_Tipi', 'Kullanım_Durumu',
            'Yatırıma_Uygunluk', 'Takas', 'Tapu_Durumu'
        ]
                
        # Dropping columns that are still objects, which can interfere with reindex and model:
        # The safest approach is to ensure all columns in the final DF are included in model_columns.

        # Kodlanmış olan orijinal kategorik sütunların kaldırılması
        # Dropping the original categorical columns that were encoded:
        df.drop(columns=['Isıtma_Tipi', 'Kullanım_Durumu', 'Tapu_Durumu'], errors='ignore', inplace=True)
                        
        # BEFORE reindex:
        print("DF columns BEFORE reindex:", df.columns.tolist())
        df = df.reindex(columns=model_columns, fill_value=0)
        
        # AFTER reindex:
        print("DF columns AFTER reindex:", df.columns.tolist()) # Should match model_coSlumns

        # Puanların NumPy dizisini çıkarmak için .values icin kullanılıyor
        # Using .values to extract the NumPy array of scores
        df['Şehir'] = target_encoder.transform(df)['Şehir'].values 
        df['Şehir'] = df['Şehir'].astype(float)

        # --- ölçeklendirme
        # --- Scaling ---
        num_cols = ['Net_Metrekare','Brüt_Metrekare','Oda_Sayısı',
                    'Binanın_Kat_Sayısı','Banyo_Sayısı',
                    'Oda_Buyuklugu','Banyo_Orani']
        df[num_cols] = scaler.transform(df[num_cols])
        
        # --- Tahmin ---
        # --- Predicting ---
        log_pred = model.predict(df)
        price = np.expm1(log_pred)[0]
        
        return {"tahmini_fiyat": f"{price:,.0f} TL"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

import joblib
cols = joblib.load("model_columns.pkl")
print(cols)

#python -m uvicorn API:app --reload