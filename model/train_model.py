import os
import psycopg2
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Connexion à PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)

# Requête SQL
query = "SELECT sepal_length, petal_length, petal_length_dup, sepal_width FROM iris"
df = pd.read_sql_query(query, conn)
conn.close()

# Définition des features et target
X = df[["sepal_width", "petal_length", "petal_length_dup"]]
y = df["sepal_length"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"📊 Mean Squared Error: {mse}")

# Tracking avec MLflow
mlflow.set_experiment("sepal_length_prediction")
with mlflow.start_run():
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="SepalLengthPredictor")

# Sauvegarde locale
os.makedirs("model/artifacts", exist_ok=True)
joblib.dump(model, "model/artifacts/sepal_length_predictor.joblib")
print("✅ Modèle sauvegardé avec succès")

# Génération automatique du graphique après entraînement
import subprocess
subprocess.run(["python", "model/visualize.py"])
