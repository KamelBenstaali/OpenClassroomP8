# 🚀 Vision Transport API - Segmentation Sémantique

Cette API fournit un service de segmentation d'images en temps réel pour le projet de véhicule autonome. Elle est construite avec **FastAPI** et utilise un modèle de Deep Learning (U-Net / MobileNet) entraîné sur Cityscapes.

## 🛠 Fonctionnalités
*   **Performance Asynchrone** : Basée sur ASGI pour traiter plusieurs requêtes sans bloquer.
*   **Chargement Optimisé** : Le modèle TensorFlow est chargé une seule fois au démarrage (Singleton) pour une latence d'inférence minimale.
*   **Swagger UI** : Documentation interactive générée automatiquement.

## 📦 Installation et Lancement

### 1. Pré-requis
Assurez-vous d'avoir Python 3.10+ installé.

### 2. Installation des dépendances
Placez-vous dans le dossier `app/api` :
```bash
cd app/api
pip install -r requirements.txt
```

*Note : Si vous êtes sur Mac M1/M2, assurez-vous d'avoir installé `tensorflow` (et non tensorflow-cpu).*

### 3. Configuration du Modèle
L'API attend un modèle `.keras` valide.
Par défaut, elle cherche dans : `../../models/checkpoints/UNet_Light_NoAug/best_model.keras`
*Vous pouvez modifier ce chemin dans `main.py` (variable `MODEL_PATH`).*

### 4. Démarrage du Serveur
Lancez le serveur avec Uvicorn (rechargement automatique activé pour le dev) :
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
L'API sera accessible sur : `http://localhost:8000`

## 🔌 Endpoints

### `GET /` (Health Check)
Vérifie que l'API tourne et que le modèle est bien chargé en mémoire.
*   **Réponse** : `{"status": "API is running", "model_loaded": true}`

### `POST /predict` (Inférence)
Envoie une image pour obtenir son masque de segmentation.
*   **Input** : Fichier image (Multipart form data, key=`file`).
*   **Process** :
    1.  Resize automatique en **224x224**.
    2.  Normalisation [0-1].
    3.  Inférence Modèle.
*   **Output** : JSON contenant :
    *   `filename` : Nom du fichier source.
    *   `shape` : Dimensions du masque (224, 224).
    *   `mask` : Matrice 2D des classes prédites (0-7) sous forme de liste de listes.

## 📚 Documentation Interactive
Une fois le serveur lancé, accédez à la documentation Swagger pour tester l'API directement depuis votre navigateur :
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**
