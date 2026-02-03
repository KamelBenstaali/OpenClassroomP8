# 🚗 Interface de Démonstration - Segmentation Sémantique

Cette application **Streamlit** sert d'interface graphique pour démontrer les capacités du modèle de segmentation. Elle permet aux équipes métiers (Laura) de tester la robustesse du modèle via des scénarios interactifs.

## 🌟 Fonctionnalités Clés
1.  **Sélecteur d'Images** : Choix parmi une liste d'images de test pré-chargées (provenant de Cityscapes).
2.  **Laboratoire de Robustesse** :
    *   🌞 **Luminosité** : Simuler des conditions de jour/nuit (Slider 0.1x à 2.0x).
    *   🌗 **Contraste** : Simuler du brouillard ou des conditions difficiles.
    *   🪞 **Flip Horizontal** : Vérifier si le modèle reconnait la route dans un miroir.
3.  **Visualisation Comparative** :
    *   Affichage côte à côte : *Input Modifié* vs *Vérité Terrain* vs *Prédiction API*.
    *   Application automatique de la **palette de couleurs Cityscapes** sur le masque brut renvoyé par l'API.

## 🚀 Installation et Lancement

### 1. Pré-requis
*   L'API (`app/api`) doit être lancée et accessible sur `http://localhost:8000`.
*   Python 3.10+ installé.

### 2. Installation des dépendances
Placez-vous dans le dossier `app/ui` :
```bash
cd app/ui
pip install -r requirements.txt
```

### 3. Préparation des Données de Test
L'application s'attend à trouver des images dans `../data/test_samples`.
*   *Images* : `../data/test_samples/images/*.png`
*   *Masques* : `../data/test_samples/masks/*.png`
*(Assurez-vous d'avoir exécuté le script `setup_demo_data.py` ou copié manuellement quelques images Cityscapes ici).*

### 4. Démarrage de l'Application
Lancez Streamlit :
```bash
streamlit run app.py
```
L'interface s'ouvrira automatiquement dans votre navigateur (URL par défaut : `http://localhost:8501`).

## 🖌️ Légende des Couleurs
L'application utilise la nomenclature Cityscapes simplifiée (8 classes) :
*   🟣 **Flat (Route)** : Violet
*   🔴 **Human** : Rouge
*   🔵 **Vehicle** : Bleu
*   ⚫ **Void** : Noir
*   (Voir la légende interactive dans l'app pour le reste)
