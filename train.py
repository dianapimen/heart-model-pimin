import pandas as pd
from pathlib import Path
from joblib import dump
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "creditcard.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgboost-fraud-v1.joblib"

# Cargar el dataset
df = pd.read_csv(DATA_PATH)
df = df.dropna(axis=0)

# Separar la variable objetivo y las características
y = df["Class"]
X = df.drop(columns=["Class"])

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
peso_fraude = negativas / positivas

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
print("Entrenando modelo XGBoost...")
modelo.fit(X_train, y_train)

# Evaluar
predicciones = modelo.predict(X_test)

print("\nREPORTE DE CLASIFICACIÓN")
print(classification_report(y_test, predicciones, zero_division=0))

print("MATRIZ DE CONFUSIÓN")
print(confusion_matrix(y_test, predicciones))

# Guardar el modelo
MODEL_DIR.mkdir(parents=True, exist_ok=True)
dump(modelo, MODEL_PATH)

print(f"\nModelo guardado en: {MODEL_PATH}")