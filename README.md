# 🌸 Iris ML Pipeline - Dockerized Machine Learning Project

Ce projet met en place une pipeline complète de Machine Learning, de l'ingestion des données jusqu'à l'API de prédiction, en passant par le prétraitement, l'entraînement et le suivi avec MLflow — le tout orchestré avec Docker Compose.

---

## 🚀 Objectifs du projet

- Importer, nettoyer et stocker les données Iris dans PostgreSQL
- Entraîner un modèle de régression prédictif (Scikit-learn)
- Suivre les métriques et les versions du modèle via MLflow
- Fournir une API FastAPI pour effectuer des prédictions

---

## 🧱 Architecture des services (Docker Compose)

```bash
.
├── api/               → FastAPI app pour exposer le modèle
├── model/             → Entraînement + logging MLflow
├── preprocess/        → Nettoyage des données + insertion PostgreSQL
├── iris.csv      → Données brutes
├── init.sql           → Script SQL pour créer la table iris
├── docker-compose.yml → Orchestration des services
└── .env               → Variables d'environnement (DB, ports)
```

---

## 🐳 Services inclus

| Service       | Port local        | Description                                      |
|--------------|-------------------|--------------------------------------------------|
| PostgreSQL    | 5432              | Base de données relationnelle                   |
| pgAdmin       | 5050              | Interface web pour gérer la base de données     |
| MLflow        | 5000              | Interface de suivi des modèles ML               |
| FastAPI       | 8000              | API REST pour prédictions                       |
| Preprocessing | (lancé au start)  | Nettoyage + insertion des données               |
| Training      | (lancé au start)  | Entraînement + enregistrement du modèle         |

---

## ⚙️ Lancer le projet

Assure-toi d’avoir Docker et Docker Compose installés, puis lance :

```bash
docker-compose down -v  # Pour repartir propre
docker-compose up --build
```

---

## 📊 Accès aux interfaces

- Swagger (FastAPI) : http://localhost:8000/docs

- MLflow UI : http://localhost:5000/

- PostgreSQL : localhost:5432 (user: postgres, mdp: selon .env)

---

## 🧪 Exemple d’appel API

```http
POST /predict
Content-Type: application/json

{
  "SepalWidthCm": 3.0,
  "PetalLengthCm": 4.5,
  "PetalWidthCm": 1.2
}
```

Réponse :

```json
{
  "prediction": 5.87
}
```
## Aperçus utiles

- Le modèle est sauvegardé dans : model/artifacts/sepal_length_predictor.joblib

- La base de données s'appelle iris et contient une table iris

- Les logs MLflow se trouvent dans le dossier mlruns/