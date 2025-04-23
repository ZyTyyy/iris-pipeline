import os
import joblib
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

# Connexion à la base de données
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)

# Chargement des données
query = "SELECT sepal_length, petal_length, petal_length_dup, sepal_width FROM iris"
df = pd.read_sql_query(query, conn)
conn.close()

# Chargement du modèle
model = joblib.load("model/artifacts/sepal_length_predictor.joblib")

# Prédictions
X = df[["sepal_width", "petal_length", "petal_length_dup"]]
y_true = df["sepal_length"]
y_pred = model.predict(X)

# Visualisation
plt.figure(figsize=(8, 6))
plt.scatter(X["sepal_width"], y_true, color="blue", label="Valeurs réelles")
plt.scatter(X["sepal_width"], y_pred, color="red", label="Prédictions", alpha=0.6)
plt.xlabel("Largeur des sépales (sepal_width)")
plt.ylabel("Longueur des sépales (sepal_length)")
plt.title("Comparaison des valeurs réelles et prédites")
plt.legend()
plt.grid(True)

# Sauvegarde du graphique
output_path = "model/artifacts/visualisation_reelle_vs_predite.png"
plt.savefig(output_path)
print(f"✅ Visualisation sauvegardée dans {output_path}")
