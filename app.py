
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "xgboost-fraud-v1.joblib"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

modelo = load(MODEL_PATH)

app = FastAPI(
    title="API de detección de fraude",
    description="Predicción de transacciones fraudulentas mediante XGBoost",
    version="1.0.0"
)


class Transaccion(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.get("/")
def inicio():
    return {"mensaje": "API de detección de fraude activa"}


@app.post("/score")
def evaluar_transaccion(transaccion: Transaccion):
    datos = transaccion.model_dump()
    X = pd.DataFrame([datos], columns=FEATURES)

    probabilidad = float(modelo.predict_proba(X)[0, 1])
    prediccion = int(modelo.predict(X)[0])

    return {
        "prediccion": prediccion,
        "resultado": "Fraude" if prediccion == 1 else "Transaccion normal",
        "puntuacion_fraude": round(probabilidad, 4)
    }
