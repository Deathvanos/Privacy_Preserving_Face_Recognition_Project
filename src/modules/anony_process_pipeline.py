import os
import io
import base64
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.modules.image_preprocessing import preprocess_image
from src.modules.eigenface import EigenfaceGenerator
from src.modules.noise_generator import NoiseGenerator
from src.modules.utils_image import pillow_image_to_bytes, numpy_image_to_pillow

from src.config import IMAGE_SIZE

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SHOW_DIR = "show_test_subject"
os.makedirs(SHOW_DIR, exist_ok=True)

# Répertoire pour enregistrer les images reconstruites
RECONSTRUCTED_DIR = "../../data/reconstructed"
os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)


def save_show_images(subject_id, images):
    """
    Sauvegarde les images encodées en base64 dans le dossier SHOW_DIR.
    """
    for i, b64img in enumerate(images):
        img_bytes = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(SHOW_DIR, f"subject_{subject_id}_image_{i}.jpg"))


def save_reconstructed_images(subject_id, images):
    """
    Enregistre le lot d'images reconstruites pour un sujet donné dans un sous-dossier.
    Chaque image est enregistrée au format JPEG.
    """
    subject_dir = os.path.join(RECONSTRUCTED_DIR, f"subject_{subject_id}")
    os.makedirs(subject_dir, exist_ok=True)
    for i, b64img in enumerate(images):
        img_bytes = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(subject_dir, f"reconstructed_{i}.jpg"))
    logger.info(f"Images reconstruites enregistrées pour le sujet {subject_id} dans {subject_dir}")


# ------------------------
# Étape 1 : Preprocessing
# ------------------------
def run_preprocessing(folder_path: str = None, df_images: pd.DataFrame = None) -> dict:
    """
    Charge et prétraite les images depuis un dossier ou à partir d'un DataFrame.
    Si 'folder_path' est fourni, les images sont lues depuis le dossier.
    Si 'df_images' est fourni, celui-ci doit contenir :
        - 'userFaces' : images PIL,
        - 'subject_number' : identifiant du sujet,
        - 'imageId' : identifiant unique de l'image.
    Retourne un dictionnaire où chaque clé est un identifiant de sujet associé
    à une liste d'images prétraitées (dictionnaires contenant 'resized_image',
    'grayscale_image', 'normalized_image' et 'flattened_image').
    """
    image_groups = defaultdict(list)

    if df_images is not None:
        for index, row in df_images.iterrows():
            try:
                img = row['userFaces']
                subject_id = str(row['subject_number'])
                preprocessed = preprocess_image(img, resize_size=IMAGE_SIZE)
                if preprocessed and preprocessed.get('flattened_image') is not None:
                    image_groups[subject_id].append(preprocessed)
            except Exception as e:
                logger.warning(f"Erreur lors du traitement de l'image à l'index {index}: {e}")
                continue
    elif folder_path is not None:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for filename in tqdm(image_files, desc="Prétraitement des images"):
            try:
                parts = filename.split("_")
                if len(parts) < 3:
                    continue
                subject_id = parts[1]
                img_path = os.path.join(folder_path, filename)
                with Image.open(img_path) as img:
                    preprocessed = preprocess_image(img, resize_size=IMAGE_SIZE)
                if preprocessed and preprocessed.get('flattened_image') is not None:
                    image_groups[subject_id].append(preprocessed)
            except Exception as e:
                logger.warning(f"Erreur lors du traitement de {filename}: {e}")
                continue
    else:
        logger.error("Aucune source d'images fournie. Fournir folder_path ou df_images.")
        raise ValueError("Aucune source d'images fournie.")

    return image_groups


# ------------------------
# PARTIE PEEP : Algorithme d'anonymisation par l'application du bruit sur des eigenfaces
# ------------------------

# ------------------------
# Étape 2 : Calcul des eigenfaces (PEEP)
# ------------------------
def run_eigenface(flattened_stack: np.ndarray, n_components: int):
    """
    Calcule les eigenfaces à partir d'une pile d'images aplaties.
    Retourne l'objet PCA, la liste des eigenfaces, la face moyenne,
    et la projection des images dans le sous-espace eigenface.
    """
    eigen_gen = EigenfaceGenerator(flattened_stack, n_components=n_components)
    eigen_gen.generate()
    eigenfaces = eigen_gen.get_eigenfaces()
    mean_face = eigen_gen.get_mean_face()
    pca = eigen_gen.get_pca_object()
    projection = pca.transform(flattened_stack)
    return pca, eigenfaces, mean_face, projection


# ------------------------
# Étape 3 : Ajout de bruit différentiel (PEEP)
# ------------------------
def run_add_noise(projection: np.ndarray, epsilon: float, sensitivity: float = 1.0):
    """
    Ajoute du bruit Laplacien à la projection des images.
    Retourne la projection bruitée.
    """
    noise_gen = NoiseGenerator(projection, epsilon)
    noise_gen.normalize_images()
    noise_gen.add_laplace_noise(sensitivity)
    noised_projection = noise_gen.get_noised_eigenfaces()
    return noised_projection


# ------------------------
# Étape 4 : Reconstruction (PEEP)
# ------------------------
def run_reconstruction(pca, noised_projection: np.ndarray) -> list:
    """
    Reconstruit les images à partir de la projection bruitée.
    Retourne une liste d'images reconstruites sous forme de chaînes base64.
    """
    reconstructions = pca.inverse_transform(noised_projection)
    reconstructed_images = []
    for recon in reconstructions:
        pil_img = numpy_image_to_pillow(recon, resized_size=IMAGE_SIZE)
        b64_img = pillow_image_to_bytes(pil_img)
        reconstructed_images.append(b64_img)
    return reconstructed_images


# ------------------------
# Fonction globale pour exécuter la pipeline complète
# ------------------------
def run_pipeline(folder_path: str = None, df_images: pd.DataFrame = None,
                 epsilon: float = 9.0, n_components_ratio: float = 0.9) -> dict:
    """
    Exécute la pipeline complète pour chaque sujet en utilisant :
      1. Preprocessing (depuis un dossier ou un DataFrame)
      2. PARTIE PEEP :
           - Calcul des eigenfaces
           - Ajout de bruit différentiel
           - Reconstruction
    Retourne un dictionnaire contenant, pour chaque sujet, les résultats suivants :
        {
            "resized": [images redimensionnées en base64],
            "grayscale": [images en niveaux de gris en base64],
            "normalized": [images normalisées sous forme de listes],
            "flattened": [images aplaties],
            "mean_face": image moyenne en base64,
            "projection": [projection initiale],
            "noised_projection": [projection avec bruit],
            "reconstructed": [images reconstruites en base64]
        }
    """
    pipeline_result = {}

    # Étape 1 : Preprocessing
    image_groups = run_preprocessing(folder_path=folder_path, df_images=df_images)

    for subject_id in tqdm(image_groups, desc="Traitement par sujet"):
        images = image_groups[subject_id]
        if len(images) < 2:
            logger.info(f"Sujet {subject_id} ignoré (moins de 2 images)")
            continue

        flattened_stack = np.array([img['flattened_image'] for img in images])
        n_components = min(int(n_components_ratio * flattened_stack.shape[0]), flattened_stack.shape[1])

        # ------------------------------
        # PARTIE PEEP : Début
        # ------------------------------
        # Étape 2 : Calcul des eigenfaces
        pca, eigenfaces, mean_face, projection = run_eigenface(flattened_stack, n_components)
        # Étape 3 : Ajout de bruit différentiel
        noised_projection = run_add_noise(projection, epsilon, sensitivity=1.0)
        # Étape 4 : Reconstruction
        reconstructed_images = run_reconstruction(pca, noised_projection)
        # ------------------------------
        # PARTIE PEEP : Fin
        # ------------------------------

        pipeline_result[subject_id] = {
            "resized": pillow_image_to_bytes(images['resized_image']),
            "grayscale": pillow_image_to_bytes(images['grayscale_image']),
            "normalized": images['normalized_image'].tolist(),
            "flattened": flattened_stack.tolist(),
            "mean_face": pillow_image_to_bytes(numpy_image_to_pillow(mean_face, resized_size=IMAGE_SIZE)),
            "projection": projection.tolist(),
            "noised_projection": noised_projection.tolist(),
            "reconstructed": reconstructed_images
        }

        # Enregistrement des images reconstruites dans le répertoire dédié
        save_reconstructed_images(subject_id, reconstructed_images)

        logger.debug(f"Sujet {subject_id} traité avec succès.")

    return pipeline_result
