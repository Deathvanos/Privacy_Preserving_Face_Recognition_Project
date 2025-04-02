import binascii
import os
import io
import base64
import logging
from collections import defaultdict
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.modules.image_preprocessing import preprocess_image
import src.modules.k_same_pixel as ksp
from src.modules.eigenface import EigenfaceGenerator
from src.modules.noise_generator import NoiseGenerator
from src.modules.utils_image import pillow_image_to_bytes, numpy_image_to_pillow

from src.config import IMAGE_SIZE

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SHOW_DIR = "show_test_subject"
# Retirer le commentaire pour ajouter le dossier "SHOW_DIR"
#os.makedirs(SHOW_DIR, exist_ok=True)

# Répertoire pour enregistrer les images reconstruites
RECONSTRUCTED_DIR = "../../data/reconstructed"
os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)


# --- Fonctions de Sauvegarde ---
def save_show_images(subject_id, images):
    """Sauvegarde les images exemple (base64) dans SHOW_DIR."""
    os.makedirs(SHOW_DIR, exist_ok=True)
    for i, b64img in enumerate(images):
        if b64img is None: continue
        try:
            img_bytes = base64.b64decode(b64img)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(os.path.join(SHOW_DIR, f"subject_{subject_id}_image_{i}.jpg"))
        except Exception as e:
            logger.error(f"Erreur sauvegarde image exemple {i} sujet {subject_id}: {e}")

def save_reconstructed_images(subject_id, images):
    """Enregistre les images reconstruites PEEP (base64) pour un sujet."""
    subject_dir = os.path.join(RECONSTRUCTED_DIR, f"peep_subject_{subject_id}") # Nom distinct
    os.makedirs(subject_dir, exist_ok=True)
    saved_count = 0
    for i, b64img in enumerate(images):
        if b64img is None: continue
        try:
            img_bytes = base64.b64decode(b64img)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(os.path.join(subject_dir, f"reconstructed_peep_{i}.jpg"))
            saved_count +=1
        except Exception as e:
            logger.error(f"Erreur sauvegarde image reconstruite {i} sujet {subject_id}: {e}")
    if saved_count > 0:
        logger.info(f"{saved_count} images PEEP enregistrées pour sujet {subject_id} dans {subject_dir}")

def save_ksame_images(subject_id, images):
    """Enregistre les images anonymisées k-same (base64) pour un sujet."""
    subject_dir = os.path.join(RECONSTRUCTED_DIR, f"ksame_subject_{subject_id}") # Nom distinct
    os.makedirs(subject_dir, exist_ok=True)
    saved_count = 0
    for i, b64img in enumerate(images):
        if b64img is None: continue
        try:
            img_bytes = base64.b64decode(b64img)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(os.path.join(subject_dir, f"anonymized_ksame_{i}.jpg"))
            saved_count += 1
        except Exception as e:
            logger.error(f"Erreur sauvegarde image k-same {i} sujet {subject_id}: {e}")
    if saved_count > 0:
        logger.info(f"{saved_count} images K-Same enregistrées pour sujet {subject_id} dans {subject_dir}")

# ---------------------------------------
# Étape 1 : Preprocessing (Simplifiée)
# ---------------------------------------
# (Version fournie par l'utilisateur, ne fait plus k-same ici)
def run_preprocessing(
    folder_path: Optional[str] = None,
    df_images: Optional[pd.DataFrame] = None,
    b64_image_list: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Charge et prétraite les images depuis UNE source. Stocke l'imageId original.
    Ne fait PAS de k-same pixel ici.
    """
    num_sources = sum(p is not None for p in [folder_path, df_images, b64_image_list])
    if num_sources != 1: raise ValueError("Fournir exactement une source d'images.")
    logger.info("Exécution du pré-traitement standard...")
    image_groups = defaultdict(list)
    # --- Cas 1: DataFrame ---
    if df_images is not None:
        logger.info(f"Traitement de {len(df_images)} images depuis DataFrame.")
        for index, row in tqdm(df_images.iterrows(), total=df_images.shape[0], desc="Preprocessing (DataFrame)"):
            try:
                img = row['userFaces']; subject_id = str(row['subject_number']); image_id = row['imageId']
                if not isinstance(img, Image.Image): logger.warning(f"Index {index} non-PIL Image. Skip."); continue
                preprocessed = preprocess_image(img, resize_size=IMAGE_SIZE)
                # Vérifier si les clés nécessaires existent (adaptez si preprocess_image change)
                if preprocessed and all(k in preprocessed for k in ['grayscale_image', 'flattened_image']):
                    preprocessed['imageId'] = image_id
                    image_groups[subject_id].append(preprocessed)
                else: logger.warning(f"Prétraitement échoué ou clés manquantes index {index}. Skip.")
            except Exception as e: logger.error(f"Erreur image index {index}: {e}", exc_info=True)
    # --- Cas 2: Dossier ---
    elif folder_path is not None:
        logger.info(f"Traitement images depuis dossier: {folder_path}")
        try:
            all_files = os.listdir(folder_path); image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pgm'))]
            logger.info(f"Trouvé {len(image_files)} fichiers image potentiels.")
        except Exception as e: logger.error(f"Erreur accès dossier {folder_path}: {e}"); raise
        for filename in tqdm(image_files, desc="Preprocessing (Folder)"):
             try:
                parts = filename.split("_");
                if len(parts) < 2: logger.warning(f"Skip: nom fichier non standard '{filename}'."); continue
                subject_id = parts[1]; image_id = os.path.splitext(filename)[0]
                img_path = os.path.join(folder_path, filename)
                with Image.open(img_path) as img: preprocessed = preprocess_image(img.convert('RGB'), resize_size=IMAGE_SIZE)
                if preprocessed and all(k in preprocessed for k in ['grayscale_image', 'flattened_image']):
                    preprocessed['imageId'] = image_id
                    image_groups[subject_id].append(preprocessed)
                else: logger.warning(f"Prétraitement échoué pour '{filename}'. Skip.")
             except UnidentifiedImageError: logger.error(f"Erreur ouverture/identification '{filename}'. Skip."); continue
             except Exception as e: logger.error(f"Erreur traitement fichier '{filename}': {e}", exc_info=True)
    # --- Cas 3: Liste Base64 ---
    elif b64_image_list is not None:
        logger.info(f"Traitement de {len(b64_image_list)} images depuis liste Base64.")
        subject_id = "1" # ID fixe
        for i, b64_string in enumerate(tqdm(b64_image_list, desc="Preprocessing (Base64 List)")):
            try:
                image_id = f"b64_img_{i}"
                img_bytes = base64.b64decode(b64_string); img = Image.open(io.BytesIO(img_bytes))
                preprocessed = preprocess_image(img.convert('RGB'), resize_size=IMAGE_SIZE)
                if preprocessed and all(k in preprocessed for k in ['grayscale_image', 'flattened_image']):
                    preprocessed['imageId'] = image_id
                    image_groups[subject_id].append(preprocessed)
                else: logger.warning(f"Prétraitement échoué image base64 index {i}. Skip.")
            except (binascii.Error, IOError, UnidentifiedImageError) as decode_err: logger.error(f"Erreur décodage/ouverture image base64 index {i}: {decode_err}. Skip."); continue
            except Exception as e: logger.error(f"Erreur traitement image base64 index {i}: {e}", exc_info=True)
    logger.info(f"Pré-traitement terminé. {len(image_groups)} sujets traités.")
    return dict(image_groups)


# ---------------------------------------
# Étape 2 : K-Same-Pixel (Nouvelle Fonction Modulaire)
# ---------------------------------------
def run_k_same_anonymization(
    image_groups: Dict[str, List[Dict[str, Any]]],
    k_value: int
) -> Dict[str, List[Optional[np.ndarray]]]:
    """
    Applique l'anonymisation k-same-pixel à chaque groupe d'images d'un sujet.

    Args:
        image_groups (Dict): Dictionnaire résultat de run_preprocessing.
                             Clés=subject_id, Valeurs=Listes de dicts d'images prétraitées.
                             Chaque dict doit contenir 'grayscale_image' (PIL) et 'imageId'.
        k_value (int): Paramètre K pour k-same-pixel.

    Returns:
        Dict[str, List[Optional[np.ndarray]]]: Dictionnaire par subject_id. Chaque valeur est
                                               une liste d'arrays NumPy anonymisés (ou None si échec),
                                               dans le même ordre que les images d'entrée du sujet.
    """
    logger.info(f"Application de K-Same-Pixel avec k={k_value}...")
    k_same_results_arrays = defaultdict(list) # Pour stocker les arrays NumPy ordonnés

    if not image_groups:
        logger.warning("image_groups est vide, K-Same Pixel sauté.")
        return {}

    for subject_id, subject_preprocessed_list in tqdm(image_groups.items(), desc="K-Same Pixel"):
        subject_output_arrays = [None] * len(subject_preprocessed_list) # Pré-initialiser avec None

        if len(subject_preprocessed_list) < k_value:
            logger.warning(f"Sujet {subject_id}: Moins d'images ({len(subject_preprocessed_list)}) que k={k_value}. K-Same Pixel sauté.")
            k_same_results_arrays[subject_id] = subject_output_arrays # Garder la liste de None
            continue

        # Préparer l'entrée pour k_same_pixel_individual: [(np_array, imageId), ...]
        k_same_input_list = []
        id_order_map = {} # Pour retrouver l'ordre original
        valid_input_indices = [] # Indices des images valides pour k-same

        for idx, img_dict in enumerate(subject_preprocessed_list):
            img_id = img_dict.get('imageId')
            grayscale_img = img_dict.get('grayscale_image')

            if img_id is None or grayscale_img is None:
                 logger.warning(f"Image index {idx} sujet {subject_id} manque 'imageId' ou 'grayscale_image'. Skip pour K-Same.")
                 continue # Ne peut pas traiter cette image

            try:
                 # Utiliser l'image grayscale PIL pour la convertir en array ici
                 grayscale_np = np.array(grayscale_img, dtype=np.uint8)
                 k_same_input_list.append((grayscale_np, img_id))
                 id_order_map[img_id] = idx # Stocker index original
                 valid_input_indices.append(idx) # Marquer comme valide pour traitement
            except Exception as conv_err:
                 logger.error(f"Erreur conversion image {img_id} (sujet {subject_id}) en NumPy: {conv_err}")

        if not k_same_input_list or len(k_same_input_list) < k_value:
             logger.warning(f"Pas assez d'images valides préparées ({len(k_same_input_list)}) pour K-Same sujet {subject_id} (k={k_value}).")
             k_same_results_arrays[subject_id] = subject_output_arrays # Garder la liste de None
             continue

        try:
            # Appeler la fonction k-same importée
            # Retour attendu: [(anonymized_array, original_imageId), ...]
            k_same_output_tuples = ksp.k_same_pixel_individual(k_same_input_list, k=k_value)
        except Exception as k_err:
            logger.error(f"Erreur durant k_same_pixel_individual sujet {subject_id}: {k_err}", exc_info=True)
            k_same_results_arrays[subject_id] = subject_output_arrays # Garder la liste de None
            continue

        # Réorganiser les résultats dans l'ordre original
        id_to_anonymized_map = {img_id: anon_array for anon_array, img_id in k_same_output_tuples}

        for img_id, original_index in id_order_map.items():
             anon_array = id_to_anonymized_map.get(img_id)
             if anon_array is not None:
                 subject_output_arrays[original_index] = anon_array
             else:
                  logger.warning(f"Résultat K-Same non trouvé pour imageId {img_id} (sujet {subject_id}).")

        k_same_results_arrays[subject_id] = subject_output_arrays

    logger.info("Traitement K-Same-Pixel terminé.")
    return dict(k_same_results_arrays)


# ---------------------------------------------
# Étape 3 : Calcul des eigenfaces (PEEP)
# ---------------------------------------------
# (Fonction run_eigenface inchangée)
def run_eigenface(flattened_stack: np.ndarray, n_components: int):
    """Calcule les eigenfaces (PCA)."""
    logger.debug(f"Calcul Eigenfaces avec n_components={n_components}")
    eigen_gen = EigenfaceGenerator(flattened_stack, n_components=n_components)
    eigen_gen.generate()
    mean_face = eigen_gen.get_mean_face()
    pca = eigen_gen.get_pca_object()
    projection = pca.transform(flattened_stack)
    # Note: eigenfaces ne sont pas utilisées directement dans la suite de CETTE pipeline
    # eigenfaces = eigen_gen.get_eigenfaces()
    return pca, mean_face, projection


# ---------------------------------------------
# Étape 4 : Ajout de bruit différentiel (PEEP)
# ---------------------------------------------
# (Fonction run_add_noise inchangée)
def run_add_noise(projection: np.ndarray, epsilon: float, sensitivity: float = 1.0):
    """Ajoute du bruit Laplacien à la projection."""
    logger.debug(f"Ajout bruit Laplacien (epsilon={epsilon}, sensitivity={sensitivity})")
    # Vérifier si la projection est valide avant de continuer
    if projection is None or projection.size == 0:
         logger.error("Projection invalide reçue pour ajout de bruit.")
         return None # Ou lever une exception
    noise_gen = NoiseGenerator(projection, epsilon)
    noise_gen.normalize_images() # Normalise les projections en interne
    noise_gen.add_laplace_noise(sensitivity)
    noised_projection = noise_gen.get_noised_eigenfaces()
    return noised_projection


# ---------------------------------------
# Étape 5 : Reconstruction (PEEP)
# ---------------------------------------
# (Fonction run_reconstruction inchangée)
def run_reconstruction(pca, noised_projection: Optional[np.ndarray]) -> List[Optional[str]]:
    """Reconstruit les images à partir de la projection bruitée."""
    if pca is None or noised_projection is None:
         logger.error("PCA ou projection bruitée manquante pour la reconstruction.")
         # Retourner une liste de None de la bonne taille serait idéal, mais difficile sans connaître la taille attendue ici.
         # Retourner une liste vide pour indiquer l'échec.
         return []

    logger.debug("Reconstruction des images depuis projection bruitée...")
    reconstructions = pca.inverse_transform(noised_projection)
    reconstructed_images_b64 = []
    for recon_flat in reconstructions:
        try:
            # Assurer que la taille correspond à IMAGE_SIZE pour la reconversion
            expected_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0]) # H, W
            pil_img = numpy_image_to_pillow(recon_flat, resized_size=expected_shape)
            b64_img = pillow_image_to_bytes(pil_img)
            reconstructed_images_b64.append(b64_img)
        except Exception as e:
            logger.error(f"Erreur reconversion image reconstruite: {e}")
            reconstructed_images_b64.append(None) # Ajouter None si erreur
    return reconstructed_images_b64


# -----------------------------------------------------
# Fonction globale pour exécuter la pipeline complète (Modifiée)
# -----------------------------------------------------
def run_pipeline(
    folder_path: Optional[str] = None,
    df_images: Optional[pd.DataFrame] = None,
    b64_image_list: Optional[List[str]] = None,
    epsilon: float = 9.0,
    n_components_ratio: float = 0.9,
    k_same_k_value: Optional[int] = None # Paramètre K-Same ajouté ici
) -> dict:
    """
    Exécute la pipeline complète : Preprocessing -> K-Same (optionnel) -> PEEP.

    Args:
        folder_path, df_images, b64_image_list: Source (une seule).
        epsilon (float): Epsilon pour bruit différentiel PEEP.
        n_components_ratio (float): Ratio pour PCA PEEP.
        k_same_k_value (int, optional): K pour K-Same-Pixel. Si None, étape sautée.

    Returns:
        dict: Dictionnaire par subject_id avec résultats (images base64), incluant
              'k_same_anonymized' (si K>0) et 'reconstructed' (PEEP).
    """
    pipeline_result = {}
    logger.info(f"Démarrage pipeline complète: eps={epsilon}, ratio={n_components_ratio}, k={k_same_k_value}")

    # --- Étape 1 : Preprocessing ---
    image_groups = run_preprocessing(
        folder_path=folder_path, df_images=df_images, b64_image_list=b64_image_list
    )
    if not image_groups:
        logger.error("Preprocessing n'a retourné aucune image. Arrêt.")
        return {}

    # --- Étape 2 : K-Same-Pixel (Optionnelle) ---
    k_same_results_b64 = defaultdict(list) # Pour stocker {subject_id: [b64_img1, b64_img2,...]}
    if k_same_k_value is not None and k_same_k_value > 0:
        # Appelle la fonction modulaire k-same
        k_same_arrays_dict = run_k_same_anonymization(image_groups, k_same_k_value)

        # Convertir les résultats en Base64
        logger.info("Conversion des résultats K-Same en Base64...")
        for subject_id, ordered_k_same_arrays in k_same_arrays_dict.items():
             b64_list = []
             for arr in ordered_k_same_arrays:
                 if arr is not None:
                     try:
                         pil_img = numpy_image_to_pillow(arr, resized_size=IMAGE_SIZE)
                         b64_img = pillow_image_to_bytes(pil_img)
                         b64_list.append(b64_img)
                     except Exception as conv_err:
                          logger.error(f"Erreur conversion K-Same array B64 sujet {subject_id}: {conv_err}")
                          b64_list.append(None)
                 else:
                     b64_list.append(None)
             k_same_results_b64[subject_id] = b64_list

        # Sauvegarde des images K-Same (Optionnelle)
        for subject_id, b64_images in k_same_results_b64.items():
            if any(img is not None for img in b64_images): # Sauvegarder seulement si qqc a été généré
                 save_ksame_images(subject_id, b64_images)
    else:
        logger.info("Étape K-Same-Pixel sautée (k non fourni ou invalide).")


    # --- Étapes 3-5 : PEEP (Eigenface + Bruit + Reconstruction) ---
    logger.info("Démarrage des étapes PEEP...")
    peep_results = {} # Stocke temporairement les sorties PEEP pour chaque sujet

    for subject_id in tqdm(image_groups, desc="Traitement PEEP par sujet"):
        subject_preprocessed_list = image_groups[subject_id]
        peep_subject_output = { # Initialiser les sorties pour ce sujet
            "mean_face_b64": None,
            "projection": [],
            "noised_projection": [],
            "reconstructed_b64": [None] * len(subject_preprocessed_list) # Pré-remplir avec None
        }

        if len(subject_preprocessed_list) < 2:
            logger.warning(f"Sujet {subject_id}: Moins de 2 images ({len(subject_preprocessed_list)}). PEEP sauté.");
            peep_results[subject_id] = peep_subject_output; continue

        # Préparer flattened_stack et garder trace des indices/IDs valides
        flattened_stack_list = []
        valid_indices_map = {} # Map original_index -> index_in_stack
        original_image_ids = [img_dict.get('imageId') for img_dict in subject_preprocessed_list]

        for idx, img_dict in enumerate(subject_preprocessed_list):
             flattened = img_dict.get('flattened_image')
             if flattened is not None:
                 flattened_stack_list.append(np.array(flattened))
                 valid_indices_map[idx] = len(flattened_stack_list) - 1 # Stoker le nouvel index
             else: logger.warning(f"Sujet {subject_id}, Img Idx {idx}: 'flattened_image' manquante.")

        if len(flattened_stack_list) < 2:
            logger.warning(f"Sujet {subject_id}: Moins de 2 images valides pour PEEP. PEEP sauté.");
            peep_results[subject_id] = peep_subject_output; continue

        flattened_stack_np = np.array(flattened_stack_list)
        n_samples, n_features = flattened_stack_np.shape
        n_components = min(max(1, int(n_components_ratio * n_samples)), n_features, n_samples) # Assurer >= 1

        logger.debug(f"Sujet {subject_id}: PCA avec n_components={n_components}")
        try:
            pca, mean_face, projection = run_eigenface(flattened_stack_np, n_components)
            noised_projection = run_add_noise(projection, epsilon) if epsilon > 0 else projection # Appliquer bruit si epsilon > 0
            reconstructed_b64_list_valid = run_reconstruction(pca, noised_projection) # Liste pour les images valides seulement

            # Stocker résultats PEEP (pour les images traitées)
            peep_subject_output["mean_face_b64"] = pillow_image_to_bytes(numpy_image_to_pillow(mean_face, resized_size=IMAGE_SIZE))
            peep_subject_output["projection"] = projection.tolist() if projection is not None else []
            peep_subject_output["noised_projection"] = noised_projection.tolist() if noised_projection is not None else []

            # Ré-insérer les images reconstruites aux bons indices dans la liste finale
            final_reconstructed_list = [None] * len(subject_preprocessed_list)
            if len(reconstructed_b64_list_valid) == len(valid_indices_map):
                 for original_idx, stack_idx in valid_indices_map.items():
                      final_reconstructed_list[original_idx] = reconstructed_b64_list_valid[stack_idx]
                 peep_subject_output["reconstructed_b64"] = final_reconstructed_list
            else:
                 logger.error(f"Sujet {subject_id}: Discordance de taille entre images reconstruites PEEP et images valides. Reconstruction échouée.")
                 # Garde la liste de None

        except Exception as peep_err:
            logger.error(f"Erreur durant PEEP pour sujet {subject_id}: {peep_err}", exc_info=True)
            # Garde les valeurs par défaut (None/[]) dans peep_subject_output

        peep_results[subject_id] = peep_subject_output
        # Sauvegarde PEEP (reconstruit)
        if any(img is not None for img in peep_subject_output["reconstructed_b64"]):
             save_reconstructed_images(subject_id, peep_subject_output["reconstructed_b64"])


    # --- Assemblage Final du Résultat ---
    logger.info("Assemblage du résultat final de la pipeline...")
    for subject_id, subject_preprocessed_list in image_groups.items():
        # Récupérer les résultats PEEP pour ce sujet
        peep_output = peep_results.get(subject_id, {})
        # Récupérer les résultats K-Same pour ce sujet (liste de b64 ou liste vide)
        ksame_output_b64 = k_same_results_b64.get(subject_id, [None] * len(subject_preprocessed_list))

        # Créer le dictionnaire final pour le sujet
        pipeline_result[subject_id] = {
            # Données du preprocessing (converties en b64/list si nécessaire)
            "imageIds": [img.get('imageId') for img in subject_preprocessed_list],
            "resized": [pillow_image_to_bytes(img['resized_image']) if img.get('resized_image') else None for img in subject_preprocessed_list],
            "grayscale": [pillow_image_to_bytes(img['grayscale_image']) if img.get('grayscale_image') else None for img in subject_preprocessed_list],
            "normalized": [img['normalized_image'].tolist() if img.get('normalized_image') is not None else None for img in subject_preprocessed_list],
            "flattened": [img['flattened_image'].tolist() if img.get('flattened_image') is not None else None for img in subject_preprocessed_list],

             # Données K-Same (déjà en b64)
             "k_same_anonymized": ksame_output_b64,

            # Données PEEP
            "mean_face": peep_output.get("mean_face_b64"),
            "projection": peep_output.get("projection", []),
            "noised_projection": peep_output.get("noised_projection", []),
            "reconstructed": peep_output.get("reconstructed_b64", [])
        }
        logger.debug(f"Résultat assemblé pour sujet {subject_id}.")

    logger.info("Pipeline terminée.")
    return pipeline_result
