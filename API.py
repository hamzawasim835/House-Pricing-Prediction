from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import re

# 1. Kaydedilen Modelleri Yükle
model = joblib.load("turkiye_house_price_model.pkl")
scaler = joblib.load("standard_scaler.pkl")
target_encoder = joblib.load("target_encoder.pkl")
label_encoders = joblib.load("label_encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

app = FastAPI()

# 2. Veri Modeli (Giriş Değerleri)
class HouseData(BaseModel):
    Net_Metrekare: float
    Brut_Metrekare: float
    Oda_Sayisi: float
    Bulundugu_Kat: str  # Regex yapılacak
    Esya_Durumu: str
    Binanin_Yasi: str   # Regex yapılacak
    Isitma_Tipi: str
    Sehir: str
    Binanin_Kat_Sayisi: int
    Kullanim_Durumu: str
    Yatirima_Uygunluk: str
    Takas: str
    Tapu_Durumu: str
    Banyo_Sayisi: float

# 3. Temizlik Fonksiyonları (Notebook'tan alındı)
def clean_age(veri):
    sayilar = [float(s) for s in re.findall(r'\d+', str(veri))]
    if len(sayilar) == 2:
        return sum(sayilar) / 2
    elif len(sayilar) == 1:
        return sayilar[0]
    return 0.0

def clean_floor(data):
    yazi = str(data).lower()
    if "giriş" in yazi or "zemin" in yazi or "bahçe" in yazi:
        return 0.0
    sayilar = re.findall(r'\d+', yazi)
    if sayilar:
        return float(sayilar[0])
    return 0.0

# 4. Tahmin Endpoint'i
@app.post("/predict")
def predict_price(data: HouseData):
    try:
        # Veriyi DataFrame'e çevir
        df = pd.DataFrame([data.dict()])
        
        # Sütun isimlerini Notebook ile eşle (Türkçe karakter vb.)
        df.rename(columns={
            'Brut_Metrekare': 'Brüt_Metrekare',
            'Oda_Sayisi': 'Oda_Sayısı',
            'Bulundugu_Kat': 'Bulunduğu_Kat',
            'Eşya_Durumu': 'Eşya_Durumu',
            'Binanin_Yasi': 'Binanın_Yaşı',
            'Isitma_Tipi': 'Isıtma_Tipi',
            'Sehir': 'Şehir',
            'Binanin_Kat_Sayisi': 'Binanın_Kat_Sayısı',
            'Kullanim_Durumu': 'Kullanım_Durumu',
            'Yatirima_Uygunluk': 'Yatırıma_Uygunluk',
            'Tapu_Durumu': 'Tapu_Durumu',
            'Banyo_Sayisi': 'Banyo_Sayısı'
        }, inplace=True)

        # --- FEATURE ENGINEERING ---
        df['Oda_Buyuklugu'] = df['Net_Metrekare'] / df['Oda_Sayısı']
        df['Banyo_Orani'] = df['Banyo_Sayısı'] / df['Oda_Sayısı']
        
        # --- TEMİZLİK (REGEX) ---
        df['Binanın_Yaşı'] = df['Binanın_Yaşı'].apply(clean_age)
        df['Bulunduğu_Kat'] = df['Bulunduğu_Kat'].apply(clean_floor)
        
        # Banyo sayısı outlier capping
        df.loc[df['Banyo_Sayısı'] > 5, 'Banyo_Sayısı'] = 5

        # --- ENCODING ---
        # 1. Label Encoding
        for col in ['Takas', 'Eşya_Durumu', 'Yatırıma_Uygunluk']:
            # Bilinmeyen değer gelirse '0' veya varsayılan ata
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except:
                df[col] = 0 # Fallback

        # 2. Target Encoding (Şehir, vb.)
        # Target encoder transform sadece X ister (y yok)
        df = target_encoder.transform(df)

        # 3. One-Hot Encoding
        cols_to_ohe = ['Isıtma_Tipi', 'Kullanım_Durumu', 'Tapu_Durumu']
        df = pd.get_dummies(df, columns=cols_to_ohe, drop_first=True)

        # --- KOLON HİZALAMA (Çok Önemli) ---
        # API'ye gelen tek satırda tüm one-hot kolonları oluşmaz.
        # Modelin beklediği kolonlara göre reindex yapıp eksikleri 0 ile dolduruyoruz.
        df = df.reindex(columns=model_columns, fill_value=0)

        # --- SCALING ---
        num_cols = ['Net_Metrekare','Brüt_Metrekare','Oda_Sayısı','Binanın_Kat_Sayısı','Banyo_Sayısı','Oda_Buyuklugu','Banyo_Orani']
        df[num_cols] = scaler.transform(df[num_cols])

        # --- TAHMİN ---
        log_pred = model.predict(df)
        price = np.expm1(log_pred)[0] # Logaritmik tahmini TL'ye çevir

        return {"tahmini_fiyat": f"{price:,.0f} TL"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# python -m uvicorn API:app --reload