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
├── data/iris.csv      → Données brutes
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

- pgAdmin : http://localhost:5050
  - Login: admin@admin.com / admin
  - DB: iris_db > Table: iris
- MLflow : http://localhost:5000
- Swagger UI : http://localhost:8000/docs

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

---

## 📁 Artéfacts

- 📈 `model/artifacts/plot.png` : visualisation réelle vs prédite
- ✅ Modèle enregistré : `SepalLengthPredictor` (MLflow Tracking)

---
