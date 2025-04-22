import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

# Connexion à la base PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)

# Lecture des données depuis la base
query = "SELECT * FROM iris;"
df = pd.read_sql_query(query, conn)

# Préparation des features et de la target
X = df.drop("species", axis=1)
y = df["species"]

# Encodage de la variable cible
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Découpage train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print("Classification Report :")
print(classification_report(y_test, y_pred, target_names=le.classes_))

conn.close()

import joblib
import os

# Créer un dossier de sauvegarde s'il n'existe pas
os.makedirs("model/artifacts", exist_ok=True)

# Sauvegarde du modèle
joblib.dump(model, "model/artifacts/random_forest_model.joblib")

# Sauvegarde de l’encodeur de labels
joblib.dump(le, "model/artifacts/label_encoder.joblib")

print("✅ Modèle et encodeur sauvegardés dans model/artifacts/")
