from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import mlflow.pyfunc

app = FastAPI()

# Chargement du modèle MLflow depuis l'alias de production
model_uri = "models:/SepalLengthPredictor@production"
model = mlflow.pyfunc.load_model(model_uri)

# Modèle de données attendu dans la requête POST
class SepalFeatures(BaseModel):
    sepal_width: float
    petal_length: float
    petal_width: float

# Route de prédiction
@app.post("/predict")
def predict_sepal_length(features: SepalFeatures):
    # Crée un tableau avec les features dans le bon ordre attendu par le modèle
    input_array = np.array([[features.sepal_width, features.petal_length, features.petal_width]])
    
    # Effectue la prédiction
    prediction = model.predict(input_array)
    
    # Retourne le résultat
    return {"predicted_sepal_length": float(prediction[0])}
