# Utiliser une image légère de Python 3.10
FROM python:3.10-slim

# Configuration recommandée par Hugging Face Spaces : création d'un utilisateur non-root
# Hugging Face exécute les conteneurs avec un utilisateur au lieu de 'root'
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances
COPY --chown=user app/api/requirements.txt /app/

# Installer les dépendances Python
# --no-cache-dir pour réduire la taille de l'image
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement ce qui est nécessaire pour l'API (On ignore le dossier ui et les données de test)
COPY --chown=user app/api /app/app/api
COPY --chown=user app/model /app/app/model

# Exposer le port que FastAPI utilisera (Hugging Face EXIGE le port 7860)
EXPOSE 7860

# Commande de lancement attendue
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
