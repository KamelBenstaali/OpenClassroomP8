# Analyse Détaillée des Architectures de Modélisation

Ce document détaille la structure interne couche par couche des trois modèles implémentés pour le projet de segmentation "Future Vision Transport".

---

## Modèle 1 : U-Net "Light" (Custom)
Ce modèle est un auto-encodeur convolutionnel symétrique. Il a été conçu "from scratch" pour être ultra-léger (< 2M paramètres).

### 1. Encodeur (Contracting Path)
Son rôle est d'analyser l'image pour comprendre "QUOI" s'y trouve, en sacrifiant la précision spatiale ("OÙ").
*   **Conv2D (16, 32, 64 filtres)** : Applique des filtres apprenants (kernels 3x3) pour détecter des motifs simples (bords) puis complexes (formes).
    *   *Padding='same'* : Préserve la taille de l'image (224x224) pour éviter les calculs de cropping.
*   **BatchNormalization** : Normalise les activations pour stabiliser et accélérer l'apprentissage (évite l'explosion des gradients).
*   **ReLU (Activation)** : Introduit la non-linéarité (permet de comprendre des fonctions complexes).
*   **MaxPooling2D (2x2)** : Réduit la taille de l'image par 2 (Downsampling). Force le modèle à généraliser et augmente le champ réceptif.

### 2. Bottleneck (Goulot d'étranglement)
C'est le point de compression maximale (Image 28x28, 128 canaux).
*   **Dropout (0.2)** : Désactive aléatoirement 20% des neurones pendant l'entraînement. Empêche le modèle d'apprendre par cœur (Overfitting).

### 3. Décodeur (Expansive Path)
Son rôle est de reconstruire l'image à sa taille originale pour localiser les classes pixel par pixel.
*   **Conv2DTranspose** : Convolution inversée qui agrandit l'image (Upsampling x2).
*   **Concatenate (Skip Connections)** : **C'est la clé du U-Net**. On fusionne la sortie de l'upsampling avec la couche correspondante de l'encodeur. Cela réinjecte les détails de haute résolution perdus lors du Pooling.

### 4. Head (Sortie)
*   **Conv2D (1x1, 8 filtres)** : Projette les features finales vers les 8 classes cibles.
*   **Softmax** : Transforme les scores en probabilités (la somme fait 1 par pixel).

---

## Modèle 2 : U-Net avec Backbone MobileNetV2
Ici, on remplace l'Encodeur "Maison" par un réseau ultra-optimisé et pré-entraîné.

### 1. Backbone (MobileNetV2)
MobileNetV2 sert d'extracteur de caractéristiques.
*   **Depthwise Separable Convolutions** : Technologie clé. Sépare la convolution spatiale (3x3 par canal) de la convolution de mélange (1x1). Réduit le coût de calcul par ~9.
*   **Inverted Residual Blocks** : Connecte les couches "minces" (peu de canaux) entre elles via des "expansions" temporaires. Optimise le flux d'information avec peu de mémoire.
*   **Transfer Learning (ImageNet)** : Le réseau arrive avec des poids qui savent déjà reconnaître des textures complexes (voitures, routes, végétation).

### 2. Décodeur Custom
Identique au principe du U-Net standard, mais adapté aux dimensions de sortie de MobileNet (112x112, 56x56, 28x28, 7x7).
*   Reconstruit le masque de segmentation 224x224 à partir des features riches du MobileNet.

---

## Modèle 3 : DeepLabV3+ (Light)
L'état de l'art pour la segmentation sémantique, optimisé ici avec un backbone MobileNetV2.

### 1. Backbone (MobileNetV2)
Extrait les features de bas niveau (Détails) et de haut niveau (Sémantique).

### 2. Module ASPP (Atrous Spatial Pyramid Pooling)
Le cœur de l'innovation DeepLab. Il permet de voir le contexte à plusieurs échelles SANS réduire la résolution de l'image.
*   **Convolutions Dilatées (Atrous)** : Les filtres ont des "trous" (Taux de dilation = 6, 12, 18).
    *   *Rate 6* : Voit les détails moyens.
    *   *Rate 12* : Voit les gros objets.
    *   *Rate 18* : Voit le contexte global (toute la scène).
*   **Global Average Pooling** : Ajoute une information sur l'image entière (moyenne globale).
*   Toutes ces branches sont concaténées pour former une vision "Super-Contextuelle".

### 3. Décodeur Avancé
DeepLabV3+ a un décodeur plus intelligent que U-Net.
*   Il récupère les **Low-Level Features** (détails bruts du début du réseau).
*   Il les fusionne avec les **High-Level Features** issues de l'ASPP.
*   Cela permet des frontières d'objets (ex: contours des piétons) beaucoup plus nettes et précises.

---
*Document généré pour le support technique du projet P8.*
