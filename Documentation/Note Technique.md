# Note Technique : Conception d'un Système de Segmentation d'Images pour Véhicule Autonome

## 1. Introduction et Définition du Problème
Dans le cadre du développement des véhicules autonomes chez Future Vision Transport, la perception de l'environnement est une brique critique. L'objectif est de concevoir un module de **segmentation sémantique** capable de classer chaque pixel d'une image caméra en 8 catégories (Route, Véhicule, Piéton, etc.).

### 1.1 Contraintes Fortes
*   **Environnement Embarqué** : Le modèle doit tourner sur un hardware limité. La contrainte majeure est une consommation mémoire (RAM) inférieure à **1 Go** lors de l'inférence.
*   **Sécurité** : La précision de la segmentation est vitale. Confondre une route et un trottoir peut mener à un accident. Nous utiliserons des métriques robustes (Dice Score, IoU) pour garantir cette fiabilité, particulièrement sur les classes vulnérables (Piétons).

## 2. État de l'Art et Choix d'Architectures
Pour répondre au compromis Précision/Légèreté, nous avons étudié et implémenté trois architectures de Réseaux de Neurones Convolutionnels (CNN).

### 2.1 Théorie : Encodeur-Décodeur
La majorité des réseaux de segmentation suivent cette structure :
*   **Encodeur (Backbone)** : Réduit la dimension spatiale pour extraire des caractéristiques sémantiques abstraites (Quoi ?).
*   **Décodeur** : Reconstruction spatiale de l'image pour localiser précisément les classes (Où ?).

### 2.2 Modèle 1 : U-Net Light (Baseline)
*   **Concept** : Architecture historique introduisant les "Skip Connections" pour une reconstruction fine.
*   **Adaptations "Light" vs U-Net Officiel (Ronneberger 2015)** :
    1.  **Réduction des Filtres** : Division par 4 du nombre de canaux (Start: 16 vs 64). L'original (30M params) est inexploitable sur embarqué. Notre version (<2M params) tient largement dans 1 Go RAM.
    2.  **Padding "Same"** : L'officiel utilise `valid` (réduction de taille progressive), obligeant à rogner les features (cropping). Nous utilisons `same` pour maintenir les dimensions (224x224) constantes, simplifiant l'architecture.
    3.  **Modernisation** : Ajout de **Batch Normalization** (absent en 2015) après chaque convolution pour accélérer la convergence.

### 2.3 Modèle 2 : U-Net avec Backbone MobileNetV2
*   **Innovation : Convolutions Séparables** : MobileNetV2 utilise des *Depthwise Separable Convolutions*.
    *   *Principe* : Au lieu de faire une convolution 3D lourde, on sépare le spatial (Depthwise) du canal (Pointwise).
    *   *Gain* : Réduit le coût de calcul et le nombre de paramètres d'un facteur 8 à 9.
*   **Transfer Learning** : L'encodeur est pré-entraîné sur ImageNet (1.4M images). Cela apporte une capacité de généralisation visuelle immédiate, cruciale vu la taille modeste de notre dataset d'entraînement (Cityscapes).

### 2.4 Modèle 3 : DeepLabV3+ (État de l'Art)
*   **Problème adressé** : La perte de résolution et de contexte. Un réseau classique "oublie" le contexte global en zoomant.
*   **Solution : ASPP (Atrous Spatial Pyramid Pooling)** : Utilise des convolutions "dilatées" (à trous) avec différents taux (rates 6, 12, 18). Cela permet au réseau de regarder la scène avec plusieurs "niveaux de zoom" simultanés sans ajouter de paramètres. C'est idéal pour détecter à la fois un gros bus et un petit feu tricolore.

## 3. Stratégie de Données et Augmentation

### 3.1 Pipeline d'Ingestion Optimisé (tf.data)
Pour garantir la performance industrielle :
*   **Parallélisme** : Utilisation de `num_parallel_calls=tf.data.AUTOTUNE` pour charger les images sur le CPU pendant que le GPU s'entraîne.
*   **Mixed Precision** : Activation de `mixed_float16` pour réduire l'empreinte VRAM de 50% et accélérer les calculs tensoriels.

### 3.2 Stratégie de Data Augmentation
Le dataset Cityscapes est limité (2975 images train). Pour éviter le surapprentissage :
*   **Transformations** : Flip Horizontal (miroir), Ajustement aléatoire de Luminosité (+/- 10%) et Contraste.
*   **Validation** : Chaque modèle est entraîné deux fois (Avec et Sans Augmentation) pour quantifier scientifiquement le gain de robustesse apporté par cette stratégie.

### 3.3 Fonction de Perte Hybride (Combo Loss)
Face au déséquilibre des classes (Route > 80% des pixels vs Piéton < 1%), l'Accuracy classique est trompeuse.
Nous avons implémenté une fonction de coût combinée :
$$ \mathcal{L}_{Total} = \mathcal{L}_{CE} + \mathcal{L}_{Dice} $$
*   La **Cross-Entropy** assure une convergence stable.
*   La **Dice Loss** force le modèle à maximiser le recouvrement géométrique des petites objets.

## 4. Résultats et Analyse Comparative
*(Section à compléter avec vos tableaux MLflow et graphiques)*

### 4.1 Protocole Expérimental
*   Tableau des hyperparamètres (Learning Rate, Batch Size, Optimizer Adam).

### 4.2 Comparaison des Performances
*   [Insérer Tableau Comparatif : Modèle | Mean IoU | Dice Score | RAM Usage]
*   [Insérer Graphiques MLflow : Courbes Loss Train/Val]

### 4.3 Analyse par Classe
*   Quelle classe est la mieux prédite ? (Probablement 'Flat' ou 'Sky').
*   Quelle classe pose problème ? (Probablement 'Object' ou 'Human'). Analyse des matrices de confusion.

## 5. Mise en Production : Architecture API & UI

### 5.1 API de Prédiction (Micro-service)
Développée avec **FastAPI** pour sa performance asynchrone (ASGI).
*   **Workflow** :
    1.  Réception Image (POST multipart).
    2.  Prétraitement (Resize 224x224 + Normalisation) identique à l'entraînement.
    3.  Inférence (Modèle chargé en Singleton au démarrage).
    4.  Post-traitement (Argmax) -> Renvoi Masque JSON.
*   **Optimisation** : Chargement du modèle hors du cycle de requête pour une latence minimale (< 200ms).

### 5.2 Interface de Démonstration (Streamlit)
Outil conçu pour les équipes métier ("Laura").
*   **Test de Robustesse Interactif** : Intègre des sliders (Luminosité, Contraste) permettant de dégrader l'image en direct pour tester les limites du modèle.
*   **Visualisation** : Affichage comparatif (Image Altérée vs Ground Truth vs Prédiction Colorisée).

## 6. Conclusion et Perspectives
Ce travail a permis de valider une architecture légère (**[Nom du modèle gagnant]**) capable de tourner sous la contrainte de 1 Go tout en maintenant une précision acceptable (> 60% IoU).
**Pistes d'amélioration :**
*   **Quantization (INT8)** : Convertir le modèle en entiers 8-bits via TensorFlow Lite pour diviser sa taille par 4 et accélérer l'inférence sur CPU ARM.
*   **Pruning** : Élaguer les connexions neuronales faibles pour alléger le modèle.
