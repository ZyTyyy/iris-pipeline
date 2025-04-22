from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import psycopg2
import os

# Chargement du CSV
df = pd.read_csv("data/iris.csv")

# Nettoyage rapide (remplacer les points dans les noms de colonnes, mettre en minuscules)
df.columns = df.columns.str.strip().str.lower().str.replace('.', '_')

# Connexion à PostgreSQL
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS")
)

# Création de la table
cur = conn.cursor()
cur.execute("DROP TABLE IF EXISTS iris;")
cur.execute("""
    CREATE TABLE iris (
        sepal_length FLOAT,
        sepal_width FLOAT,
        petal_length FLOAT,
        petal_length_dup FLOAT,
        species VARCHAR
    );
""")

# Insertion des données
for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO iris (sepal_length, sepal_width, petal_length, petal_length_dup, species)
        VALUES (%s, %s, %s, %s, %s);
    """, tuple(row))

conn.commit()
cur.close()
conn.close()
