from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="House Price Prediction System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model & Scaler
model = joblib.load("turkiye_house_price_model.pkl")
scaler = joblib.load("standard_scaler.pkl")

# === MODEL FEATURES ===
MODEL_FEATURES = [
    'Net_Metrekare', 'Brüt_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 'Eşya_Durumu',
    'Binanın_Yaşı', 'Şehir', 'Binanın_Kat_Sayısı', 'Yatırıma_Uygunluk', 'Takas',
    'Banyo_Sayısı', 'Oda_Buyuklugu', 'Banyo_Orani',

    # Isıtma Tipi
    'Isıtma_Tipi_Doğalgaz Sobalı', 'Isıtma_Tipi_Güneş Enerjisi', 'Isıtma_Tipi_Isıtma Yok',
    'Isıtma_Tipi_Jeotermal', 'Isıtma_Tipi_Kat Kaloriferi', 'Isıtma_Tipi_Klimalı',
    'Isıtma_Tipi_Kombi Doğalgaz', 'Isıtma_Tipi_Merkezi (Pay Ölçer)',
    'Isıtma_Tipi_Merkezi Doğalgaz', 'Isıtma_Tipi_Merkezi Kömür',
    'Isıtma_Tipi_Sobalı', 'Isıtma_Tipi_Yerden Isıtma', 'Isıtma_Tipi_Bilinmiyor',

    # Kullanım Durumu
    'Kullanım_Durumu_Kiracı Oturuyor',
    'Kullanım_Durumu_Mülk Sahibi Oturuyor',
    'Kullanım_Durumu_Bilinmiyor',

    # Tapu Durumu
    'Tapu_Durumu_Bilinmiyor',
    'Tapu_Durumu_Hisseli Tapu',
    'Tapu_Durumu_Kat Mülkiyeti',
    'Tapu_Durumu_Kat İrtifakı',
    'Tapu_Durumu_Müstakil Tapulu',
    'Tapu_Durumu_nan'
]

# === USER INPUT MODEL ===
class HouseInput(BaseModel):
    Net_Metrekare: float
    Brüt_Metrekare: float
    Oda_Sayısı: int
    Bulunduğu_Kat: int
    Eşya_Durumu: int
    Binanın_Yaşı: int
    Şehir: float
    Binanın_Kat_Sayısı: int
    Yatırıma_Uygunluk: int
    Takas: int
    Banyo_Sayısı: int
    Oda_Buyuklugu: float
    Banyo_Orani: float

    # kategorik ham girişler
    Isıtma_Tipi: str
    Kullanım_Durumu: str
    Tapu_Durumu: str


def encode_categoricals(data: HouseInput):
    """
    Kullanıcıdan gelen kategorik değerleri modelin kolonlarına çeviren fonksiyon
    """
    row = {col: 0 for col in MODEL_FEATURES}

    # numeric values
    row.update({
        "Net_Metrekare": data.Net_Metrekare,
        "Brüt_Metrekare": data.Brüt_Metrekare,
        "Oda_Sayısı": data.Oda_Sayısı,
        "Bulunduğu_Kat": data.Bulunduğu_Kat,
        "Eşya_Durumu": data.Eşya_Durumu,
        "Binanın_Yaşı": data.Binanın_Yaşı,
        "Şehir": data.Şehir,
        "Binanın_Kat_Sayısı": data.Binanın_Kat_Sayısı,
        "Yatırıma_Uygunluk": data.Yatırıma_Uygunluk,
        "Takas": data.Takas,
        "Banyo_Sayısı": data.Banyo_Sayısı,
        "Oda_Buyuklugu": data.Oda_Buyuklugu,
        "Banyo_Orani": data.Banyo_Orani,
    })

    # Isıtma
    heat_col = f"Isıtma_Tipi_{data.Isıtma_Tipi}"
    if heat_col in row:
        row[heat_col] = 1

    # Kullanım
    use_col = f"Kullanım_Durumu_{data.Kullanım_Durumu}"
    if use_col in row:
        row[use_col] = 1

    # Tapu
    deed_col = f"Tapu_Durumu_{data.Tapu_Durumu}"
    if deed_col in row:
        row[deed_col] = 1

    return [row[col] for col in MODEL_FEATURES]


@app.post("/predict")
def predict_price(data: HouseInput):
    row = encode_categoricals(data)
    row = np.array([row])

    # scale
    row_scaled = scaler.transform(row)

    # predict
    log_price = model.predict(row_scaled)[0]
    price = float(np.expm1(log_price))

    return {"estimated_price": round(price, 2), "currency": "TL"}

# To run: python -m uvicorn API:app --reload