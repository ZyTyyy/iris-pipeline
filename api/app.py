from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Chargement du modèle et de l'encodeur
model = joblib.load("model/artifacts/random_forest_model.joblib")
label_encoder = joblib.load("model/artifacts/label_encoder.joblib")

# Création de l'app FastAPI
app = FastAPI()

# Définition du format d'entrée attendu
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_length_dup: float

# Route principale
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction Iris 🌸"}

# Route /predict
@app.post("/predict")
def predict_species(input_data: IrisInput):
    # Créer un array numpy avec les données
    input_array = np.array([[input_data.sepal_length, input_data.sepal_width,
                             input_data.petal_length, input_data.petal_length_dup]])

    # Prédire la classe
    prediction = model.predict(input_array)
    species = label_encoder.inverse_transform(prediction)[0]

    return {"predicted_species": species}
