IMAGE_SIZE = (100, 100)
DATABASE_PATH = "../data/database"

import os

# --- Chemins et Noms ---
# Chemin vers le dossier contenant les images anonymisées (structure plate)
# Utilise un chemin relatif depuis la racine du projet
LFW_ANONY_PATH = 'data/lfw_anony_processed'

# Dossier pour sauvegarder les modèles entraînés, l'encodeur de labels, etc.
MODEL_SAVE_DIR = 'models/trained/'

# Dossier pour les logs TensorBoard (optionnel, laisser vide ou None pour désactiver)
LOG_DIR = 'models/logs/'

# Nom de base pour les fichiers sauvegardés (modèle, logs, courbes)
MODEL_NAME = 'simple_cnn_lfw_anony_v1' # Ex: 'mobilenet_transfer_v1'

# --- Paramètres des Données et Prétraitement ---
IMG_WIDTH = 64      # Largeur cible des images
IMG_HEIGHT = 64     # Hauteur cible des images
# Mode couleur: 'grayscale' (1 canal) ou 'rgb' (3 canaux)
# IMPORTANT: 'rgb' (3 canaux) est souvent requis pour les modèles de transfert pré-entraînés.
#            'grayscale' est plus simple et peut suffire pour un CNN simple.
COLOR_MODE = 'grayscale'
# Nombre de canaux, doit correspondre à COLOR_MODE
CHANNELS = 1 if COLOR_MODE == 'grayscale' else 3

# --- Paramètres de Division des Données ---
# Stratégie de division: 'stratified' ou 'fixed_per_subject'
SPLIT_STRATEGY = 'stratified'

# -- Pour 'stratified' --
# Proportion du dataset total à utiliser pour l'ensemble de test
TEST_SPLIT_RATIO = 0.2  # 20% pour le test
# Proportion du dataset total à utiliser pour l'ensemble de validation
# Sera déduite de l'ensemble d'entraînement si non nulle.
VALIDATION_SPLIT_RATIO = 0.15 # 15% pour la validation (environ 65% restants pour train)

# -- Pour 'fixed_per_subject' --
# Nombre exact d'images par sujet pour l'ensemble d'entraînement
# (Ignoré si SPLIT_STRATEGY n'est pas 'fixed_per_subject')
N_TRAIN_PER_SUBJECT = 16 # Exemple: si chaque sujet a 20 images -> 4 pour test

# Seed pour la reproductibilité des divisions et initialisations
RANDOM_STATE = 42

# --- Paramètres du Modèle ---
# Choix de l'architecture dans ml_models.py: 'simple_cnn', 'transfer_MobileNetV2', 'transfer_ResNet50', etc.
MODEL_ARCHITECTURE = 'simple_cnn'

# -- Pour Transfer Learning (si MODEL_ARCHITECTURE commence par 'transfer_') --
# Nom du modèle de base à charger depuis tf.keras.applications
TRANSFER_BASE_MODEL_NAME = 'MobileNetV2'
# Faut-il geler les poids du modèle de base lors du premier entraînement ?
# Mettre à False pour du fine-tuning (nécessite souvent un LEARNING_RATE plus bas).
TRANSFER_FREEZE_BASE = True

# --- Paramètres d'Entraînement ---
EPOCHS = 50              # Nombre maximal d'époques d'entraînement
BATCH_SIZE = 32          # Taille du lot (batch size)
LEARNING_RATE = 0.001    # Taux d'apprentissage pour l'optimiseur Adam

# Patience pour l'EarlyStopping (nombre d'époques sans amélioration sur val_accuracy avant d'arrêter)
# Mettre à 0 ou une valeur négative pour désactiver l'EarlyStopping.
EARLY_STOPPING_PATIENCE = 10

# --- Vérifications et Ajustements Automatiques ---
# Assurer la cohérence entre COLOR_MODE et CHANNELS
if COLOR_MODE == 'grayscale' and CHANNELS != 1:
    print(f"Configuration Warning: COLOR_MODE is '{COLOR_MODE}', overriding CHANNELS to 1.")
    CHANNELS = 1
elif COLOR_MODE == 'rgb' and CHANNELS != 3:
    print(f"Configuration Warning: COLOR_MODE is '{COLOR_MODE}', overriding CHANNELS to 3.")
    CHANNELS = 3

# Avertissement si on tente le transfert avec des images grayscale
if MODEL_ARCHITECTURE.startswith('transfer_') and COLOR_MODE == 'grayscale':
    print(f"Configuration Warning: Transfer learning ({MODEL_ARCHITECTURE}) "
          f"est généralement conçu pour des images RGB (3 canaux), mais COLOR_MODE='grayscale'. "
          f"Cela peut causer des erreurs ou de mauvaises performances. Envisagez COLOR_MODE='rgb'.")

print("Configuration chargée et vérifiée.")