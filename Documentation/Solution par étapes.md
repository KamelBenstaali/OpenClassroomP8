
# Solution par Étapes - Projet de Segmentation pour Véhicule Autonome

## 1. Préparation des Données et Pipeline d'Ingestion
- **Conversion en Pipeline tf.data Optimisé** : Remplacement du chargement via `cv2` par un pipeline TensorFlow natif (`tf.io.read_file`, `tf.image.decode_png`) pour maximiser la vitesse et l'efficacité mémoire.
- **Transformation et Mapping** : Intégration du mapping des 8 catégories directement dans le pipeline.
- **Augmentation des Données** : Implémentation de transformations aléatoires (flip horizontal, luminosité, contraste, zoom léger) appliquées simultanément à l'image et au masque pour améliorer la robustesse du modèle.
- **Optimisation de la Performance** :
    *   Utilisation de `.map()`, `.cache()`, `.shuffle()`, `.batch()`, et `.prefetch(tf.data.AUTOTUNE)`.
    *   Activation de la **Mixed Precision** (`mixed_float16`) pour accélérer l'entraînement et réduire la consommation VRAM.
    *   **Taille d'entrée standard** : Fixée à **256x512** (ou 224x224 selon contraintes MobileNet) pour garantir une comparaison équitable de la consommation mémoire entre tous les modèles.

## 2. Modélisation et Comparaison de Modèles
L'objectif est de trouver le meilleur compromis performance/légèreté pour un système embarqué.
- **Contrainte Critique** : Consommation RAM < 1 Go en inférence.
- **Modèles à Tester** :
    Chaque modèle sera testé dans deux configurations :
    *   **Sans Data Augmentation**
    *   **Avec Data Augmentation** (Flip horizontal, luminosité, contraste, zoom léger) afin d'évaluer l'impact sur la robustesse et la généralisation.

    1.  **U-Net Standard (Baseline)** :
        *   *Pourquoi ?* Architecture historique de référence pour la segmentation sémantique.
        *   *Objectif :* Servir de point de comparaison (Baseline) pour évaluer les gains des architectures plus modernes. Construite en version "Light" (canaux réduits) pour tenter de respecter les contraintes mémoire.

    2.  **U-Net avec Backbone MobileNetV2 (Transfer Learning)** :
        *   *Pourquoi ?* Le standard industriel pour l'embarqué. Combine la puissance d'extraction de MobileNetV2 (pré-entraîné ImageNet) avec la précision de reconstruction de U-Net.
        *   *Avantage :* Excellent compromis vitesse/précision et convergence rapide grâce au Transfer Learning.

    3.  **DeepLabV3+ (Backbone MobileNetV2)** :
        *   *Pourquoi ?* État de l'art en segmentation. Utilise des convolutions à trous (Atrous Spatial Pyramid Pooling - ASPP) pour capturer le contexte à plusieurs échelles (ex: grands bus vs petits panneaux).
        *   *Avantage :* Meilleure gestion des objets de tailles variables tout en restant léger grâce au backbone MobileNet.

## 3. Métriques d'Évaluation
Pour évaluer la qualité de la segmentation de manière exhaustive :
- **IoU par Classe (Intersection over Union)** : Pour identifier les performances sur chaque catégorie spécifique (route, piéton, véhicule, etc.).
- **Mean IoU** : La métrique globale de référence.
- **Pixel Accuracy** : Pourcentage global de pixels correctement classés.
- **Dice Coefficient (F1-Score)** : Particulièrement utile pour les classes déséquilibrées.

## 4. Entraînement et Suivi
- **Fonction de Perte (Loss)** : Validation et implémentation de la **Combo Loss** (`SparseCategoricalCrossentropy + Dice Loss`).
    *   Optimisation jointe de la convergence globale et de la précision géométrique (classes rares).
- **Experiment Tracking (MLflow)** :
    *   Mise en place de MLflow pour tracker chaque expérimentation.
    *   **Organisation** : Structure par dossiers (`Experiences/`) et noms de runs clairs.
    *   **Métriques Métier** : Calcul post-training de l'**IoU par Classe** et du **Dice Score**, sauvegardés en artefact CSV.
- **Callbacks Keras** :
    *   `ModelCheckpoint` : Sauvegarde unique du meilleur modèle (`best_model.keras`) par expérience.
    *   `EarlyStopping` : Arrêt automatique sur stagnation de la `val_loss`.

## 5. Déploiement de l'API et Interface (En cours d'implémentation)
- **API (FastAPI)** :
    *   Choix technologique : **FastAPI** pour sa performance (async) et sa légèreté.
    *   **Endpoint `/predict`** : Accepte une image, resize en 224x224, infère avec le modèle chargé au démarrage, renvoie un masque brut (JSON).
    *   **Optimisation** : Chargement unique du modèle au startup (`compile=False` pour éviter les dépendances lourdes).

- **Application Web de Démo (Streamlit)** :
    *   **Sélection d'Images** : Menu déroulant pour choisir parmi un set d'images de test pré-chargées (pas d'upload utilisateur, conformément à la demande métier).
    *   **Test de Robustesse** : Ajout de contrôles interactifs (Sliders Luminosité/Contraste, Flip Miroir) pour modifier l'image avant l'envoi à l'API. Cela permet de démontrer la généralisation du modèle en direct.
    *   **Visualisation** : Affichage côte à côte : Image Modifiée | Masque Réel | Prédiction Colorisée.

- **Déploiement Cloud** : Prévu sur Azure ou Heroku.
