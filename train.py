from pathlib import Path
import pandas as pd
from joblib import dump
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Cargar el dataset
print("Cargando dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(axis=0)

# Definimos los porcentajes que queremos entrenar
porcentajes = {
    "100": 1.0,
    "50": 0.5,
    "25": 0.25
}

for version, frac in porcentajes.items():
    print(f"\n--- Entrenando modelo con el {version}% de los datos ---")
    
    # Si es menor al 100%, tomamos una muestra aleatoria manteniendo la proporción
    if frac < 1.0:
        df_subset = df.sample(frac=frac, random_state=0)
    else:
        df_subset = df.copy()

    # Separar la variable objetivo y las características
    y = df_subset["Class"]
    X = df_subset.drop(columns=["Class"])

    # Dividir los datos: 80% entrenamiento y 20% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    # Compensar el desequilibrio entre transacciones normales y fraudulentas
    negativas = (y_train == 0).sum()
    positivas = (y_train == 1).sum()
    peso_fraude = negativas / positivas if positivas > 0 else 1

    # Crear el clasificador XGBoost
    modelo = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=peso_fraude,
        eval_metric="logloss",
        random_state=0,
        n_jobs=-1
    )

    # Entrenar
    print(f"Entrenando XGBoost (Versión {version}%)...")
    modelo.fit(X_train, y_train)

    # Evaluar
    predicciones = modelo.predict(X_test)
    print(f"\nREPORTE DE CLASIFICACIÓN ({version}%)")
    print(classification_report(y_test, predicciones, zero_division=0))

    # Definir la ruta de guardado según la versión
    if version == "100":
        model_path = MODEL_DIR / "xgboost-fraud-v1.joblib"
    elif version == "50":
        model_path = MODEL_DIR / "xgboost-fraud-v50.joblib"
    else:
        model_path = MODEL_DIR / "xgboost-fraud-v25.joblib"

    # Guardar el modelo
    dump(modelo, model_path)
    print(f"Modelo {version}% guardado en: {model_path}")

print("\n¡Entrenamiento de todos los modelos completado con éxito!")
