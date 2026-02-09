
import streamlit as st
import requests
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import os
import io

# --- Configuration ---
API_URL = "http://localhost:8000/predict"
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

def inject_custom_css():
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* App Background */
        .stApp {
            background: #f8f9fa;
        }

        /* Header Styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            color: #1e1e1e;
            margin-bottom: 0.5rem;
            text-align: center;
            background: -webkit-linear-gradient(45deg, #1A2980, #26D0CE);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .sub-header {
            font-size: 1.2rem;
            font-weight: 400;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #eaeaea;
        }

        /* Card-like containers for images */
        .image-card {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            text-align: center;
            border: 1px solid #f0f0f0;
        }
        
        .image-card h4 {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }

        /* Legend Styling */
        .legend-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 15px;
            padding: 10px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 0.85rem;
            color: #444;
            background: #f1f3f5;
            padding: 4px 8px;
            border-radius: 6px;
        }
        
        .color-box {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 6px;
        }

        /* Custom Button */
        div.stButton > button {
            background: linear-gradient(90deg, #1A2980 0%, #26D0CE 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }

        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(38, 208, 206, 0.4);
            color: white;
        }
        
        /* Hide Streamlit Default Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
    </style>
    """, unsafe_allow_html=True)

# --- Interface Streamlit ---

st.set_page_config(layout="wide", page_title="Segmentation Demo", page_icon="🚗")

inject_custom_css()

# Header Area
st.markdown('<div class="main-header">🚗 Segmentation Sémantique - Véhicule Autonome</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interface de Démonstration & Test de Robustesse</div>', unsafe_allow_html=True)

# 1. Sidebar : Sélection et Contrôles
st.sidebar.markdown("## ⚙️ Panneau de Contrôle")
st.sidebar.markdown("---")
st.sidebar.subheader("1. Sélection de l'Image")

available_ids = load_local_images()
if not available_ids:
    st.sidebar.error(f"Aucune image trouvée dans {IMG_DIR}")
    selected_id = None
else:
    selected_id = st.sidebar.selectbox("Choisir une image ID :", available_ids)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Perturbations (Test)")

brightness = st.sidebar.slider("☀️ Luminosité", 0.1, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider("🌓 Contraste", 0.1, 2.0, 1.0, 0.1)
flip = st.sidebar.checkbox("↔️ Flip Horizontal")


# 2. Logique Principale
# Initialisation de l'état pour la persistance
if 'pred_mask' not in st.session_state:
    st.session_state['pred_mask'] = None

if selected_id:
    img_path = os.path.join(IMG_DIR, f"{selected_id}_leftImg8bit.png")
    mask_path = os.path.join(MASK_DIR, f"{selected_id}_gtFine_labelIds.png")
    
    try:
        # Chargement & Transform
        original_image = Image.open(img_path).convert('RGB')
        transformed_image = apply_transforms(original_image, brightness, contrast, flip)
        
        # Masque Réel
        real_mask_img = None
        if os.path.exists(mask_path):
            real_mask = Image.open(mask_path)
            if flip:
                real_mask = ImageOps.mirror(real_mask)
            real_mask_img = real_mask 
            
    except Exception as e:
        st.error(f"Erreur: {e}")
        st.stop()

    # Bouton de prédiction dans la sidebar pour ne pas casser l'alignement
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Lancer la Prédiction", type="primary"):
        with st.spinner("Analyse en cours..."):
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
                    colored_mask_img = colorize_mask(mask_pred)
                    
                    # Redimensionnement pour correspondre à l'image d'origine (et à l'affichage)
                    # Le modèle sort du 224x224, mais on veut l'afficher aligné avec l'input
                    colore_mask_resized = colored_mask_img.resize(transformed_image.size, resample=Image.NEAREST)
                    
                    st.session_state['pred_mask'] = colore_mask_resized
                else:
                    st.error(f"Erreur API: {response.status_code}")
            except Exception as e:
                st.error("API non disponible")

    # Layout avec Colonnes
    col_input, col_truth, col_pred = st.columns(3)
    
    with col_input:
        st.markdown('<div class="image-card"><h4>📷 Image (Input)</h4>', unsafe_allow_html=True)
        st.image(transformed_image, use_container_width=True)
        st.caption(f"ID: {selected_id}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_truth:
        st.markdown('<div class="image-card"><h4>🎯 Vérité Terrain</h4>', unsafe_allow_html=True)
        if real_mask_img:
            st.image(real_mask_img, use_container_width=True)
        else:
            # Placeholder simple
            st.info("Non disponible")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="image-card"><h4>🤖 Prédiction Modèle</h4>', unsafe_allow_html=True)
        
        if st.session_state['pred_mask'] is not None:
            st.image(st.session_state['pred_mask'], use_container_width=True)
        else:
            # Placeholder pour aligner visuellement
            # On crée une image grise de la MEME TAILLE que l'input pour garantir l'alignement parfait
            placeholder = Image.new('RGB', transformed_image.size, (240, 240, 240))
            st.image(placeholder, use_container_width=True, caption="En attente...")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Légende (Full Width en bas)
    if st.session_state['pred_mask'] is not None:
        st.markdown("### Légende des Classes")
        
        # Construction propre du HTML sans indentation excessive
        items_html = ""
        for i, label in enumerate(LABELS):
            color = PALETTE[i]
            rgb_val = f"rgb({color[0]}, {color[1]}, {color[2]})"
            items_html += f'<div class="legend-item"><div class="color-box" style="background-color: {rgb_val};"></div>{label}</div>'
            
        legend_html = f'<div class="legend-container">{items_html}</div>'
        st.markdown(legend_html, unsafe_allow_html=True)



