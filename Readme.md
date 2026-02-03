# Future Vision Transport - Segmentation d'Images pour Véhicule Autonome (Projet P8)

Ce projet vise à concevoir un système de **segmentation sémantique embarqué** capable d'identifier les zones navigables (route) et les obstacles (piétons, véhicules) à partir d'images caméra.

## 🎯 Objectifs
1.  **Concevoir un modèle de Deep Learning** performant et léger.
2.  **Tracking des expérimentations** via MLflow.
3.  **Développer une API de prédiction** pour l'intégration.
4.  **Créer une interface de démonstration** pour valider la robustesse.

## 🏗 Architecture du Projet
```bash
P8/
├── app/
│   ├── api/           # Micro-service FastAPI (Inférence)
│   └── ui/            # Interface de Démo Streamlit
├── data/              # (Non tracké) Images brutes Cityscapes
├── Documentation/     # Note Technique, Slides, Plan
├── Mes_notebooks/     # Notebooks d'entraînement (Colab)
├── Experiences/            # Checkpoints des modèles entraînés (.keras), modeles, artefacts, et autres 
│   │                        # metriques sauvegardées
│   ├── Models/
│.  └── checkpoints/
└── requirements.txt   # Dépendances globales (Dev local)
```

## 🧠 Modèles Implémentés
Nous avons comparé 3 architectures pour trouver le meilleur compromis Précision / Légèreté :
| Modèle | Description | Avantage Clé |
| :--- | :--- | :--- |
| **U-Net Light** | Architecture "Maison" from scratch | Baseline ultra-légère |
| **MobileNetV2 U-Net** | Transfer Learning (ImageNet) | Convergence rapide & Robustesse |
| **DeepLabV3+** | Convolutions à trous (ASPP) | Meilleure gestion du contexte multi-échelle |

## 🚀 Guide de Démarrage Rapide

### 1. Installation de l'environnement
Il est recommandé d'utiliser **Python 3.10**.
```bash
# Création venv
python3.10 -m venv venv
source venv/bin/activate

# Installation
pip install -r requirements.txt
```

### 2. Entraînement (Notebook)
Ouvrez `Mes_notebooks/Notebook_1.ipynb` (idéalement sur Google Colab avec GPU).
Le notebook gère :
*   Le téléchargement du dataset Cityscapes.
*   L'entraînement des 3 modèles avec **Combo Loss** (Dice + CrossEntropy).
*   Le tracking MLflow.
*   La sauvegarde du meilleur modèle dans `models/checkpoints/`.

### 3. Lancement de la Démo (Local)
Une fois le modèle entraîné récupéré :

**Terminal 1 : API**
```bash
cd app/api
uvicorn main:app --reload
```

**Terminal 2 : Interface**
```bash
cd app/ui
streamlit run app.py
```

## 📊 Résultats Attendus
*   **Mean IoU** cible : > 60%
*   **Consommation RAM** : < 1 Go (Contrainte embarquée)
*   **Latence** : < 200ms / image

---
*Projet réalisé dans le cadre du parcours "AI Engineer" d'OpenClassrooms.*
