import os
import numpy as np
from tensorflow import keras
from PIL import Image
import argparse
import time
from typing import Optional

# --- Import des modules du projet ---
try:
    from ..modules import data_loader
    try:
        from .. import config
    except ImportError:
        print("Erreur: Impossible d'importer src/config.py.")
        exit()
except ImportError as e:
    print(f"Erreur d'importation des modules du projet : {e}")
    exit()

# --- Fonction de Prétraitement pour une image unique ---
def preprocess_single_image(
    image_path: str,
    img_width: int,
    img_height: int,
    color_mode: str
) -> Optional[np.ndarray]:
    """
    Charge, redimensionne, normalise et formate une image unique pour la prédiction.
    """
    try:
        img = Image.open(image_path)

        pil_mode = 'L' if color_mode == 'grayscale' else 'RGB'
        img_converted = img.convert(pil_mode)

        img_resized = img_converted.resize((img_width, img_height))

        img_array = np.array(img_resized)

        img_normalized = img_array.astype('float32') / 255.0

        if color_mode == 'grayscale':
            img_final = np.expand_dims(img_normalized, axis=-1)
        else:
            img_final = img_normalized

        img_batch = np.expand_dims(img_final, axis=0)

        print(f"Image prétraitée, shape final: {img_batch.shape}")
        return img_batch

    except FileNotFoundError:
        print(f"Erreur: Fichier image introuvable : {image_path}")
        return None
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image {image_path}: {e}")
        return None

# --- Fonction Principale de Prédiction ---
def predict_image(image_path: str):
    """
    Charge le modèle et l'encodeur, prédit l'identité pour une image donnée.
    """
    print("--- Démarrage du Script de Prédiction ---")
    start_time = time.time()

    # --- 1. Charger Configuration et Chemins ---
    print("Chargement de la configuration...")
    model_filename = f"{config.MODEL_NAME}.h5" # ou .keras
    model_filepath = os.path.join(config.MODEL_SAVE_DIR, model_filename)
    encoder_filename = f"{config.MODEL_NAME}_label_encoder.joblib"
    encoder_filepath = os.path.join(config.MODEL_SAVE_DIR, encoder_filename)

    print(f"  - Modèle utilisé: {model_filepath}")
    print(f"  - Encodeur utilisé: {encoder_filepath}")
    print(f"  - Image à prédire: {image_path}")

    # --- 2. Charger Modèle et Encodeur ---
    print("\n--- Chargement du modèle et de l'encodeur ---")
    if not os.path.exists(model_filepath):
        print(f"Erreur: Fichier modèle non trouvé: {model_filepath}")
        return
    try:
        model = keras.models.load_model(model_filepath)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Keras: {e}")
        return

    # Charger l'encodeur de labels
    label_encoder = data_loader.load_label_encoder(encoder_filepath)
    if label_encoder is None:
        print("Erreur critique : Impossible de charger l'encodeur de labels.")
        return

    # --- 3. Prétraiter l'Image d'Entrée ---
    print("\n--- Prétraitement de l'image d'entrée ---")
    preprocessed_image = preprocess_single_image(
        image_path=image_path,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        color_mode=config.COLOR_MODE
    )

    if preprocessed_image is None:
        print("Échec du prétraitement de l'image.")
        return

    # --- 4. Faire la Prédiction ---
    print("\n--- Prédiction ---")
    try:
        prediction_probabilities = model.predict(preprocessed_image)

        predicted_index = np.argmax(prediction_probabilities[0])
        prediction_confidence = prediction_probabilities[0][predicted_index]

        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        print("\n--- Résultat de la Prédiction ---")
        print(f"  - Image : {os.path.basename(image_path)}")
        print(f"  - Identité Prédite (Subject ID) : {predicted_label}")
        print(f"  - Confiance : {prediction_confidence:.4f} ({prediction_confidence*100:.2f}%)")

    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Script de Prédiction Terminé en {duration:.2f} secondes ---")


# --- Point d'Entrée pour Ligne de Commande ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédit l'identité d'une personne sur une image anonymisée.")
    parser.add_argument("image_path", help="Chemin vers le fichier image à prédire.")
    # Ajouter d'autres arguments si nécessaire (ex: --model-name pour choisir un modèle spécifique)

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
         print(f"Erreur: Le fichier image spécifié n'existe pas : {args.image_path}")
    else:
        predict_image(args.image_path)