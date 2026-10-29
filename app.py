from pathlib import Path
import pandas as pd
import os
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

# 30 features que espera tu modelo XGBoost
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Diccionario para cargar dinámicamente los modelos según el dataset elegido
MODELOS = {
    "100": load(MODEL_DIR / "xgboost-fraud-v1.joblib"),
    "50": load(MODEL_DIR / "xgboost-fraud-v50.joblib"),
    "25": load(MODEL_DIR / "xgboost-fraud-v25.joblib")
}

app = FastAPI(
    title="API de detección de fraude",
    description="Predicción de transacciones fraudulentas mediante XGBoost con múltiples datasets",
    version="1.0.0"
)

# Habilitar CORS para permitir peticiones desde Angular (local y producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Transaccion(BaseModel):
    dataset: str = "100"  # Recibe la versión elegida desde el frontend ('100', '50', '25')
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


@app.post("/predict")
def evaluar_transaccion(transaccion: Transaccion):
    datos = transaccion.model_dump()
    
    # Extraemos la versión del dataset enviada por Angular (por defecto "100")
    dataset_version = datos.pop("dataset", "100")
    
    # Seleccionamos el modelo dinámicamente; si la versión no existe, usa el del 100% por seguridad
    modelo_activo = MODELOS.get(dataset_version, MODELOS["100"])
    
    # Creamos el DataFrame con las 30 características exactas para XGBoost
    X = pd.DataFrame([datos], columns=FEATURES)

    probabilidad = float(modelo_activo.predict_proba(X)[0, 1])
    prediccion = int(modelo_activo.predict(X)[0])

    return {
        "dataset_usado": dataset_version,
        "prediccion": prediccion,
        "is_fraud": bool(prediccion == 1),
        "message": "Transacción Fraudulenta" if prediccion == 1 else "Transacción Legítima",
        "probability": round(probabilidad, 4)
    }


@app.post("/retrain")
def trigger_retrain():
    github_token = os.environ.get("GITHUB_TOKEN")
    owner = "dianapimen"  # Tu usuario real
    repo = "heart-model-pimin"    # Tu repo real del backend
    
    url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"
    
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    payload = {
    "event_type": "ml_ci_cd",
        "client_payload": {
            "dataseturl": "https://huggingface.co/datasets/dianizz/CreditCard/resolve/main/creditcard_100.csv?download=true",
            "sha": "latest" # (Y asegúrate de enviar también el sha si tu docker build lo usa)
        }
    }

    
    response = requests.post(url, json=payload, headers=headers)
    
    # Imprimir en los logs de Render lo que responde GitHub
    print(f"STATUS CODE DE GITHUB: {response.status_code}")
    print(f"RESPUESTA DE GITHUB: {response.text}")
    
    if response.status_code == 204:
        return {"status": "success", "message": "Reentrenamiento iniciado"}
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Error de GitHub ({response.status_code}): {response.text}"
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
