
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Configuration ---
# Taille attendue par le modèle (doit correspondre à l'entraînement)
IMG_HEIGHT = 224
IMG_WIDTH = 224
# Chemin absolue vers le modèle pour éviter les erreurs de chemin relatif
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "Experiences", "Models", "UNet_Light_WithAug", "final_model.keras")

# --- Initialisation de l'App ---
app = FastAPI(
    title="Segmentation API - P8",
    description="API de segmentation d'images pour véhicules autonomes (Cityscapes)",
    version="1.0.0"
)

# CORS pour autoriser les requêtes depuis le frontend (Streamlit ou React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variable Globale pour le Modèle ---
model = None

# --- Chargement du Modèle au Démarrage ---
@app.on_event("startup")
async def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Chargement du modèle depuis {MODEL_PATH}...")
            # compile=False car on n'a pas besoin de la fonction de perte pour l'inférence
            # cela évite les erreurs avec les custom losses (Combo Loss) non définies
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Modèle chargé avec succès.")
        else:
            print(f"⚠️ ATTENTION : Modèle introuvable à {MODEL_PATH}")
            print("Veuillez vérifier le chemin ou uploader un modèle.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")

# --- Palette de Couleurs (Cityscapes 8 classes) ---
PALETTE = [
    [128, 64, 128],  # flat (Road) - Violet
    [220, 20, 60],   # human - Rouge
    [0, 0, 142],     # vehicle - Bleu
    [70, 70, 70],    # construction - Gris
    [220, 220, 0],   # object - Jaune
    [107, 142, 35],  # nature - Vert
    [70, 130, 180],  # sky - Ciel
    [0, 0, 0]        # void - Noir
]

# --- Fonctions Utilitaires ---
def preprocess_image(image_bytes):
    """
    Convertit les bytes en tenseur prêt pour le modèle
    """
    # Ouverture avec PIL
    img = Image.open(io.BytesIO(image_bytes))
    
    # Conversion RGB (au cas où on reçoit du RGBA ou Grayscale)
    img = img.convert('RGB')
    
    # Redimensionnement
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Conversion en Array et Normalisation [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Ajout de la dimension Batch (1, 224, 224, 3)
    img_tensor = np.expand_dims(img_array, axis=0)
    
    return img_tensor

def postprocess_mask(pred_tensor):
    """
    Convertit la sortie probabiliste (Softmax) en masque de classes (Argmax)
    """
    # pred_tensor shape: (1, 224, 224, 8)
    # Argmax sur le dernier axe -> (1, 224, 224)
    mask = np.argmax(pred_tensor, axis=-1)
    
    # On retire la dimension batch -> (224, 224)
    mask = mask[0]
    
    return mask.astype(np.uint8)

def colorize_mask(mask_array):
    """
    Applique la palette de couleurs sur un masque 2D (H, W)
    Retourne une image PIL
    """
    h, w = mask_array.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, color in enumerate(PALETTE):
        # i correspond à la classe (0 à 7)
        colored_mask[mask_array == i] = color
        
    return Image.fromarray(colored_mask)

# --- Endpoints ---
@app.get("/")
def read_root():
    return {"status": "API is running", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit une image, renvoie le masque de segmentation au format JSON (matrice brute).
    Idéal pour les applications clientes (Streamlit, React...).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")
    
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        # 1. Lecture
        contents = await file.read()
        
        # 2. Prétraitement
        input_tensor = preprocess_image(contents)
        
        # 3. Inférence
        predictions = model.predict(input_tensor)
        
        # 4. Post-traitement
        mask = postprocess_mask(predictions)
        
        # 5. Réponse
        return {
            "filename": file.filename,
            "mask": mask.tolist(), # Conversion numpy -> list pour JSON
            "shape": mask.shape
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import Response

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """
    Reçoit une image, renvoie l'image du masque colorisé directement (Format PNG).
    Idéal pour tester visuellement dans le navigateur ou Swagger UI.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")
    
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    try:
        # 1. Lecture
        contents = await file.read()
        
        # 2. Prétraitement
        input_tensor = preprocess_image(contents)
        
        # 3. Inférence
        predictions = model.predict(input_tensor)
        
        # 4. Post-traitement
        mask = postprocess_mask(predictions)
        
        # 5. Colorisation
        colored_img = colorize_mask(mask)
        
        # 6. Conversion en bytes pour la réponse
        img_byte_arr = io.BytesIO()
        colored_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(content=img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Pour lancer localement : python app/api/main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
