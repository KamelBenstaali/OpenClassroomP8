
import streamlit as st
import requests
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os
import io
import matplotlib.pyplot as plt

# --- Configuration ---
API_URL = "http://localhost:8000/predict"
# Dossier contenant quelques exemples d'images/masques pour la démo
# Structure attendue:
# app/data/images/frankfurt_...leftImg8bit.png
# app/data/masks/frankfurt_...gtFine_labelIds.png
DATA_DIR = "../data/test_samples" 
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")

# Palette de couleurs Cityscapes (8 classes)
# 0:flat, 1:human, 2:vehicle, 3:construction, 4:object, 5:nature, 6:sky, 7:void
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

LABELS = ['Flat', 'Human', 'Vehicle', 'Construction', 'Object', 'Nature', 'Sky', 'Void']

# --- Fonctions Utilitaires ---

def colorize_mask(mask_array):
    """ Applique la palette de couleurs sur un masque 2D (H, W) """
    # Création d'une image vide RGB
    h, w = mask_array.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i, color in enumerate(PALETTE):
        colored_mask[mask_array == i] = color
        
    return Image.fromarray(colored_mask)

def load_local_images():
    """ Scanne le dossier local pour trouver les IDs disponibles """
    if not os.path.exists(IMG_DIR):
        return []
    files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
    # On extrait juste l'ID unique (ex: frankfurt_000000_012345)
    # Hypothèse nom: ville_seq_id_leftImg8bit.png
    ids = [f.replace("_leftImg8bit.png", "") for f in files]
    return sorted(ids)

def apply_transforms(image, brightness, contrast, flip):
    """ Applique les transformations en temps réel """
    if flip:
        image = ImageOps.mirror(image)
        
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    return image

# --- Interface Streamlit ---

st.set_page_config(layout="wide", page_title="Segmentation Demo")

st.title("🚗 Segmentation Sémantique - Véhicule Autonome")
st.markdown("### Interface de Démonstration & Test de Robustesse")

# 1. Sidebar : Sélection et Contrôles
st.sidebar.header("1. Sélection de l'Image")

available_ids = load_local_images()
if not available_ids:
    st.sidebar.warning(f"Aucune image trouvée dans {IMG_DIR}. Veuillez ajouter des images de test.")
    selected_id = None
else:
    selected_id = st.sidebar.selectbox("Choisir une image ID :", available_ids)

st.sidebar.markdown("---")
st.sidebar.header("2. Perturbations (Test Robustesse)")

brightness = st.sidebar.slider("Luminosité", 0.1, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contraste", 0.1, 2.0, 1.0, 0.1)
flip = st.sidebar.checkbox("Flip Horizontal (Miroir)")

# 2. Chargement de l'image sélectionnée
if selected_id:
    img_path = os.path.join(IMG_DIR, f"{selected_id}_leftImg8bit.png")
    mask_path = os.path.join(MASK_DIR, f"{selected_id}_gtFine_labelIds.png")
    
    try:
        original_image = Image.open(img_path).convert('RGB')
        
        # Application des transformations
        transformed_image = apply_transforms(original_image, brightness, contrast, flip)
        
        # Chargement du masque réel (si dispo)
        if os.path.exists(mask_path):
            real_mask = Image.open(mask_path)
            if flip: # Si on flip l'image, il faut flipper le masque réel aussi pour comparer !
                real_mask = ImageOps.mirror(real_mask)
            real_mask_np = np.array(real_mask)
            # Mapping 34 -> 8 classes (Simplified logic for visualization if needed, or assume pre-mapped)
            # Pour l'affichage direct, on suppose que le masque est déjà mappé ou on utilise une palette large
            # Ici on va assumer que ce sont les labels IDs bruts et on fera au mieux, 
            # ou idéalement il faudrait une fonction de mapping ici aussi.
            # Pour la démo, on affiche souvent le raw
        else:
            real_mask = None
            
    except Exception as e:
        st.error(f"Erreur chargement image: {e}")
        st.stop()

    # 3. Zone d'affichage principale
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Image (Input)")
        st.image(transformed_image, use_container_width=True)
        st.caption(f"ID: {selected_id}")

    with col2:
        st.subheader("Vérité Terrain (Masque Réel)")
        if real_mask:
            # Note: Pour un affichage parfait, il faudrait mapper les ids Cityscapes vers nos couleurs
            # Ici on affiche en niveau de gris ou brut si pas traité
            st.image(real_mask, use_container_width=True) 
        else:
            st.info("Pas de masque réel disponible.")

    # 4. Prédiction via API
    with col3:
        st.subheader("Prédiction Modèle")
        
        if st.button("Lancer la Prédiction 🚀", type="primary"):
            with st.spinner("Interrogation de l'API..."):
                # Préparation du fichier
                buf = io.BytesIO()
                transformed_image.save(buf, format="PNG")
                buf.seek(0)
                
                try:
                    files = {"file": ("image.png", buf, "image/png")}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        mask_pred = np.array(data["mask"], dtype=np.uint8)
                        
                        # Colorisation
                        colored_pred = colorize_mask(mask_pred)
                        
                        st.image(colored_pred, use_container_width=True)
                        
                        # Légende
                        st.markdown("**Légende :**")
                        legend_cols = st.columns(4)
                        for i, label in enumerate(LABELS):
                            color_hex = '#%02x%02x%02x' % tuple(PALETTE[i])
                            legend_cols[i%4].markdown(f":blue_square: <span style='color:{color_hex}'>■</span> {label}", unsafe_allow_html=True)

                    else:
                        st.error(f"Erreur API : {response.status_code} - {response.text}")
                        
                except Exception as e:
                    st.error(f"Erreur connexion : {e}")
                    st.warning("Vérifiez que l'API tourne bien sur localhost:8000")

