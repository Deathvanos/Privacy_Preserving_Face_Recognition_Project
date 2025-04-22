# -*- coding: utf-8 -*-
import binascii
import os
import io
import base64
import logging
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from werkzeug.datastructures import FileStorage

# Assurez-vous que les chemins d'importation sont corrects pour votre structure de projet
try:
    from src.modules.image_preprocessing import preprocess_image
    import src.modules.k_same_pixel as ksp
    from src.modules.eigenface import EigenfaceGenerator
    from src.modules.noise_generator import NoiseGenerator
    from src.modules.utils_image import numpy_image_to_pillow, pillow_image_to_bytes
    from src.config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE
except ImportError:
    # Fallback si exécuté directement ou structure différente
    from src.modules.image_preprocessing import preprocess_image
    import src.modules.k_same_pixel as ksp
    from src.modules.eigenface import EigenfaceGenerator
    from src.modules.noise_generator import NoiseGenerator
    from src.modules.utils_image import pillow_image_to_bytes, numpy_image_to_pillow
    DEFAULT_IMAGE_SIZE = (100, 100)
    print("Warning: Using fallback imports and default IMAGE_SIZE.")



# Répertoire pour enregistrer les images reconstruites finales (dossier unique)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: If hard coded - May cause some problems depending on where the console is executed.
RECONSTRUCTED_DIR = ''
def set_reconstructed_dir(reconstructed_dir="../../data/reconstructed_pipeline"):
    global RECONSTRUCTED_DIR
    RECONSTRUCTED_DIR = reconstructed_dir
    os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)


# --- Fonctions de Sauvegarde ---

def save_final_reconstructed_images(subject_id: str, images: List[Optional[str]], original_image_ids: List[Optional[str]]):
    """
    Enregistre les images finales reconstruites (base64) pour un sujet
    directement dans RECONSTRUCTED_DIR. Le nom de fichier suit le format
    reconstructed_<id_subject>_<num_img>.png.
    """
    if not images or all(img is None for img in images):
        logger.warning(f"Aucune image finale à sauvegarder pour le sujet {subject_id}.")
        return

    saved_count = 0
    # Nettoyer l'ID sujet une seule fois
    safe_subject_id = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in str(subject_id))

    for i, b64img in enumerate(images):
        if b64img is None: continue
        try:
            img_bytes = base64.b64decode(b64img)
            img = Image.open(io.BytesIO(img_bytes))

            # Nouveau nom de fichier: reconstructed_<id_subject>_<num_img>.png
            # Utilise l'index 'i' comme num_img pour ce sujet
            filename = f"reconstructed_{safe_subject_id}_{i}.png"
            full_path = os.path.join(RECONSTRUCTED_DIR, filename)

            img.save(full_path)
            saved_count +=1
        except binascii.Error as b64_err:
             # Récupérer l'ID original juste pour le log d'erreur si possible
             original_id_for_log = original_image_ids[i] if i < len(original_image_ids) else f"index {i}"
             logger.error(f"Erreur décodage Base64 image {i} (orig ID: {original_id_for_log}) sujet {subject_id}: {b64_err}. Image non sauvegardée.")
        except Exception as e:
            original_id_for_log = original_image_ids[i] if i < len(original_image_ids) else f"index {i}"
            logger.error(f"Erreur sauvegarde image finale {i} (orig ID: {original_id_for_log}) sujet {subject_id} vers {full_path}: {e}. Image non sauvegardée.")

    if saved_count > 0:
        logger.info(f"{saved_count} images finales reconstruites pour sujet {subject_id} enregistrées dans {RECONSTRUCTED_DIR}")


# --- Étape 1 : Preprocessing ---

# ---------------------------------------
# Étape 1 : Preprocessing
# ---------------------------------------
def run_preprocessing(
    folder_path: Optional[str] = None,
    df_images: Optional[pd.DataFrame] = None,
    b64_image_list: Optional[List[str]] = None,
    filestorage_list: Optional[List[FileStorage]] = None,
    image_size_override: Optional[Tuple[int, int]] = None
) -> Tuple[Dict[str, List[Dict[str, Any]]], Tuple[int, int]]:
    """
    Charge et prétraite les images depuis UNE source. Stocke l'imageId original.
    Permet de spécifier la taille des images.
    """
    num_sources = sum(p is not None for p in [folder_path, df_images, b64_image_list, filestorage_list])
    if num_sources != 1: raise ValueError("Fournir exactement une source d'images.")

    target_image_size = image_size_override if image_size_override is not None else DEFAULT_IMAGE_SIZE
    logger.info(f"Exécution du pré-traitement standard avec IMAGE_SIZE={target_image_size}...")

    image_groups = defaultdict(list)
    processed_count = 0
    # --- Cas 1: DataFrame ---
    if df_images is not None:
        logger.info(f"Traitement de {len(df_images)} images depuis DataFrame.")
        for index, row in tqdm(df_images.iterrows(), total=df_images.shape[0], desc="Preprocessing (DataFrame)"):
            try:
                img = row['userFaces']; subject_id = str(row['subject_number']); image_id = row.get('imageId', f"df_img_{index}")
                if not isinstance(img, Image.Image):
                     logger.warning(f"Index {index} (ID: {image_id}) n'est pas une image PIL valide. Skip.")
                     continue
                preprocessed = preprocess_image(img, resize_size=target_image_size, create_flattened=False)
                if preprocessed and 'grayscale_image' in preprocessed:
                    preprocessed['imageId'] = image_id
                    image_groups[subject_id].append(preprocessed)
                    processed_count += 1
                else: logger.warning(f"Prétraitement échoué ou 'grayscale_image' manquante pour index {index} (ID: {image_id}). Skip.")
            except Exception as e: logger.error(f"Erreur image index {index} (ID: {image_id}): {e}", exc_info=True)
    # --- Cas 2: Dossier ---
    elif folder_path is not None:
        logger.info(f"Traitement images depuis dossier: {folder_path}")
        try:
            image_files = [f for f in os.listdir(folder_path) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pgm'))]
        except Exception as e:
            logger.error(f"Erreur accès dossier {folder_path}: {e}")
            raise
        logger.info(f"Trouvé {len(image_files)} fichiers image potentiels.")
        for filename in tqdm(image_files, desc="Preprocessing (Folder)"):
             try:
                parts = filename.split("_")
                subject_id = "unknown"
                if len(parts) >= 2: subject_id = parts[1]
                else: logger.warning(f"Nom fichier '{filename}' ne suit pas la convention 'prefix_sujetId_...'. Sujet assigné 'unknown'.")
                image_id = os.path.splitext(filename)[0]
                img_path = os.path.join(folder_path, filename)
                with Image.open(img_path) as img:
                    preprocessed = preprocess_image(img.convert('RGB'), resize_size=target_image_size, create_flattened=False)
                if preprocessed and 'grayscale_image' in preprocessed:
                    preprocessed['imageId'] = image_id
                    image_groups[subject_id].append(preprocessed)
                    processed_count += 1
                else: logger.warning(f"Prétraitement échoué ou 'grayscale_image' manquante pour '{filename}'. Skip.")
             except UnidentifiedImageError: logger.error(f"Erreur ouverture/identification '{filename}'. Skip."); continue
             except Exception as e: logger.error(f"Erreur traitement fichier '{filename}': {e}", exc_info=True)
    # --- Cas 3: Liste Base64 ---
    elif b64_image_list is not None:
        logger.info(f"Traitement de {len(b64_image_list)} images depuis liste Base64.")
        subject_id = "b64_subject_1"
        image_groups[subject_id] = []
        for i, b64_string in enumerate(tqdm(b64_image_list, desc="Preprocessing (Base64 List)")):
            try:
                image_id = f"b64_img_{i}"
                img_bytes = base64.b64decode(b64_string); img = Image.open(io.BytesIO(img_bytes))
                preprocessed = preprocess_image(img.convert('RGB'), resize_size=target_image_size, create_flattened=False)
                if preprocessed and 'grayscale_image' in preprocessed:
                    preprocessed['imageId'] = image_id
                    image_groups[subject_id].append(preprocessed)
                    processed_count += 1
                else: logger.warning(f"Prétraitement échoué ou 'grayscale_image' manquante image base64 index {i}. Skip.")
            except (binascii.Error, IOError, UnidentifiedImageError) as decode_err: logger.error(f"Erreur décodage/ouverture image base64 index {i}: {decode_err}. Skip."); continue
            except Exception as e: logger.error(f"Erreur traitement image base64 index {i}: {e}", exc_info=True)
    # --- 4. Liste FileStorage ---
    elif filestorage_list is not None:
        logger.info(f"Traitement de {len(filestorage_list)} fichiers FileStorage.")
        subject_id = "upload_subject_1"
        image_groups[subject_id] = []
        for i, file_storage in enumerate(tqdm(filestorage_list, desc="Preprocessing (FileStorage List)")):
            try:
                image_id = f"upload_img_{i}"
                img = Image.open(file_storage.stream)
                preprocessed = preprocess_image(img.convert('RGB'), resize_size=target_image_size, create_flattened=False)
                if preprocessed and 'grayscale_image' in preprocessed:
                    preprocessed['imageId'] = image_id
                    image_groups[subject_id].append(preprocessed)
                    processed_count += 1
                else:
                    logger.warning(f"Prétraitement échoué ou 'grayscale_image' manquante pour fichier FileStorage index {i}. Skip.")
            except UnidentifiedImageError:
                logger.error(f"Erreur ouverture/identification fichier FileStorage index {i}. Skip.")
                continue
            except Exception as e:
                logger.error(f"Erreur traitement fichier FileStorage index {i}: {e}", exc_info=True)
    logger.info(f"Pré-traitement terminé. {processed_count} images traitées pour {len(image_groups)} sujets.")
    if processed_count == 0:
        logger.error("Aucune image n'a pu être prétraitée. Vérifiez la source et les logs.")
    return dict(image_groups), target_image_size


# --- Étape 2 : K-Same-Pixel ---

def run_k_same_anonymization(
    image_groups: Dict[str, List[Dict[str, Any]]],
    k_value: int
) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    """
    Applique l'anonymisation k-same-pixel et retourne les résultats prêts pour PEEP.
    """
    logger.info(f"Application de K-Same-Pixel avec k={k_value}...")
    k_same_results_for_peep = defaultdict(list)

    if not image_groups:
        logger.warning("image_groups est vide, K-Same Pixel sauté.")
        return {}

    for subject_id, subject_preprocessed_list in tqdm(image_groups.items(), desc="K-Same Pixel"):
        subject_output_list = [None] * len(subject_preprocessed_list)

        if len(subject_preprocessed_list) < k_value:
            logger.warning(f"Sujet {subject_id}: Moins d'images ({len(subject_preprocessed_list)}) que k={k_value}. K-Same Pixel sauté pour ce sujet.")
            k_same_results_for_peep[subject_id] = subject_output_list
            continue

        k_same_input_list = []
        id_order_map = {}
        original_indices_processed = []

        for idx, img_dict in enumerate(subject_preprocessed_list):
            img_id = img_dict.get('imageId')
            grayscale_img = img_dict.get('grayscale_image')

            if img_id is None or grayscale_img is None:
                 logger.warning(f"Image index {idx} sujet {subject_id} manque 'imageId' ou 'grayscale_image'. Skip pour K-Same.")
                 continue

            try:
                 grayscale_np = np.array(grayscale_img, dtype=np.uint8)
                 k_same_input_list.append((grayscale_np, img_id))
                 id_order_map[img_id] = idx
                 original_indices_processed.append(idx)
            except Exception as conv_err:
                 logger.error(f"Erreur conversion image {img_id} (sujet {subject_id}) en NumPy: {conv_err}")

        if not k_same_input_list or len(k_same_input_list) < k_value:
             logger.warning(f"Pas assez d'images valides préparées ({len(k_same_input_list)}) pour K-Same sujet {subject_id} (k={k_value}).")
             k_same_results_for_peep[subject_id] = subject_output_list
             continue

        try:
            k_same_output_tuples = ksp.k_same_pixel_individual(k_same_input_list, k=k_value)
        except Exception as k_err:
            logger.error(f"Erreur durant k_same_pixel_individual sujet {subject_id}: {k_err}", exc_info=True)
            k_same_results_for_peep[subject_id] = subject_output_list
            continue

        id_to_anonymized_map = {img_id: anon_array for anon_array, img_id in k_same_output_tuples}

        for img_id, original_index in id_order_map.items():
             anon_array_uint8 = id_to_anonymized_map.get(img_id)
             if anon_array_uint8 is not None:
                 try:
                    normalized_flattened_anon = anon_array_uint8.astype(np.float32).flatten() / 255.0
                    subject_output_list[original_index] = {
                        'flattened_anonymized_image': normalized_flattened_anon,
                        'imageId': img_id
                    }
                 except Exception as post_proc_err:
                      logger.error(f"Erreur post-traitement (norm/flat) K-Same imageId {img_id} sujet {subject_id}: {post_proc_err}")
             else:
                  logger.warning(f"Résultat K-Same non trouvé pour imageId {img_id} (sujet {subject_id}) après exécution.")

        k_same_results_for_peep[subject_id] = subject_output_list

    logger.info("Traitement K-Same-Pixel terminé.")
    return dict(k_same_results_for_peep)


# --- Étape 3 : Calcul des eigenfaces (PEEP) ---

def run_eigenface(flattened_stack: np.ndarray, n_components: int):
    """Calcule les eigenfaces (PCA) sur le stack fourni."""
    logger.debug(f"Calcul Eigenfaces avec n_components={n_components} sur stack de shape {flattened_stack.shape}")
    effective_n_components = min(n_components, flattened_stack.shape[0], flattened_stack.shape[1])
    if effective_n_components < n_components:
         logger.warning(f"Réduction de n_components de {n_components} à {effective_n_components} à cause des dimensions du stack ({flattened_stack.shape[0]} samples, {flattened_stack.shape[1]} features).")
    if effective_n_components < 1:
         logger.error(f"Impossible de calculer PCA avec n_components={effective_n_components}. Stack shape: {flattened_stack.shape}")
         return None, None, None

    eigen_gen = EigenfaceGenerator(flattened_stack, n_components=effective_n_components)
    try:
        eigen_gen.generate()
        mean_face = eigen_gen.get_mean_face()
        pca = eigen_gen.get_pca_object()
        projection = pca.transform(flattened_stack)
        logger.debug(f"PCA calculée. Shape projection: {projection.shape}. Mean face shape: {mean_face.shape}")
        return pca, mean_face, projection
    except ValueError as pca_err:
        logger.error(f"Erreur PCA: {pca_err}. Stack shape: {flattened_stack.shape}, n_components: {effective_n_components}")
        return None, None, None


# --- Étape 4 : Ajout de bruit différentiel (PEEP) ---

def run_add_noise(projection: np.ndarray, epsilon: float, sensitivity: float = 1.0):
    """Ajoute du bruit Laplacien à la projection."""
    if projection is None or projection.size == 0:
         logger.error("Projection invalide (None ou vide) reçue pour ajout de bruit. Skip.")
         return None
    if epsilon <= 0:
        logger.info("Epsilon <= 0, aucun bruit Laplacien ne sera ajouté.")
        return projection

    logger.debug(f"Ajout bruit Laplacien (epsilon={epsilon}, sensitivity={sensitivity}) à projection shape {projection.shape}")
    try:
        noise_gen = NoiseGenerator(projection, epsilon)
        noise_gen.normalize_images()
        noise_gen.add_laplace_noise(sensitivity)
        noised_projection = noise_gen.get_noised_eigenfaces()
        logger.debug(f"Bruit ajouté. Shape projection bruitée: {noised_projection.shape}")
        return noised_projection
    except Exception as noise_err:
        logger.error(f"Erreur lors de l'ajout de bruit Laplacien: {noise_err}", exc_info=True)
        return None


# --- Étape 5 : Reconstruction (PEEP) ---

def run_reconstruction(
    pca: Optional[Any],
    noised_projection: Optional[np.ndarray],
    target_image_size: Tuple[int, int]
) -> List[Optional[str]]:
    """
    Reconstruit les images à partir de la projection bruitée.
    """
    if pca is None:
         logger.error("Objet PCA manquant pour la reconstruction.")
         return []
    if noised_projection is None:
         logger.error("Projection bruitée manquante pour la reconstruction.")
         return []
    if noised_projection.size == 0:
        logger.warning("Projection vide fournie pour reconstruction. Retourne liste vide.")
        return []

    logger.debug(f"Reconstruction des images depuis projection bruitée shape {noised_projection.shape}...")
    reconstructed_images_b64 = []
    try:
        reconstructions_flat = pca.inverse_transform(noised_projection)
        logger.debug(f"Inverse transform effectué. Shape reconstructions aplaties: {reconstructions_flat.shape}")

        expected_shape_hw = (target_image_size[1], target_image_size[0])

        for recon_flat in reconstructions_flat:
            try:
                if recon_flat.size != expected_shape_hw[0] * expected_shape_hw[1]:
                    logger.error(f"Taille de l'image reconstruite aplatie ({recon_flat.size}) "
                                 f"ne correspond pas à la taille attendue {expected_shape_hw[0] * expected_shape_hw[1]}. Skip cette image.")
                    reconstructed_images_b64.append(None)
                    continue

                pil_img = numpy_image_to_pillow(recon_flat, resized_size=expected_shape_hw)
                b64_img = pillow_image_to_bytes(pil_img)
                reconstructed_images_b64.append(b64_img)
            except ValueError as reshape_err:
                 logger.error(f"Erreur lors du reshape/conversion PIL pour une image reconstruite: {reshape_err}. Recon flat size: {recon_flat.size}, target shape HW: {expected_shape_hw}", exc_info=True)
                 reconstructed_images_b64.append(None)
            except Exception as e:
                logger.error(f"Erreur conversion/encodage Base64 d'une image reconstruite: {e}", exc_info=True)
                reconstructed_images_b64.append(None)

    except Exception as inv_tf_err:
        logger.error(f"Erreur durant pca.inverse_transform: {inv_tf_err}", exc_info=True)
        num_images_expected = noised_projection.shape[0]
        return [None] * num_images_expected

    logger.debug(f"{len(reconstructed_images_b64)} images préparées pour le retour (certaines peuvent être None).")
    return reconstructed_images_b64


# --- Fonction globale pour exécuter la pipeline SÉQUENTIELLE ---

def run_pipeline(
    folder_path: Optional[str] = None,
    df_images: Optional[pd.DataFrame] = None,
    b64_image_list: Optional[List[str]] = None,
    image_size_override: Optional[Tuple[int, int]] = None,
    k_same_k_value: int = 3,
    n_components_ratio: float = 0.8,
    epsilon: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Exécute la pipeline séquentielle: Preprocessing -> K-Same -> PEEP -> Reconstruction/Sauvegarde.
    """
    pipeline_results = {}
    logger.info(f"Démarrage pipeline SÉQUENTIELLE: k={k_same_k_value}, ratio={n_components_ratio}, eps={epsilon}")

    if k_same_k_value < 2:
         logger.error(f"k_same_k_value ({k_same_k_value}) doit être >= 2. Arrêt.")
         return {}

    # --- Étape 1 : Preprocessing ---
    try:
        image_groups, used_image_size = run_preprocessing(
            folder_path=folder_path, df_images=df_images, b64_image_list=b64_image_list,
            image_size_override=image_size_override
        )
    except ValueError as e:
        logger.error(f"Erreur de configuration preprocessing: {e}")
        return {}
    except Exception as e:
         logger.error(f"Erreur inattendue durant preprocessing: {e}", exc_info=True)
         return {}

    if not image_groups:
        logger.error("Preprocessing n'a retourné aucune image valide. Arrêt.")
        return {}
    logger.info(f"Preprocessing terminé. Taille d'image utilisée: {used_image_size}")

    # --- Étape 2 : K-Same-Pixel ---
    k_same_results = run_k_same_anonymization(image_groups, k_same_k_value)

    if not k_same_results:
         logger.warning("K-Same n'a retourné aucun résultat. PEEP ne peut pas continuer.")

    # --- Étapes 3-5 : PEEP (Eigenface + Bruit + Reconstruction) + Sauvegarde ---
    logger.info("Démarrage des étapes PEEP (Eigenface, Bruit, Reconstruction) sur les résultats K-Same...")

    for subject_id in tqdm(image_groups.keys(), desc="Traitement PEEP par sujet"):
        pipeline_results[subject_id] = {
            "imageIds": [img_dict.get('imageId', f'unknown_{i}') for i, img_dict in enumerate(image_groups[subject_id])],
            "final_reconstructed_b64": [None] * len(image_groups[subject_id]),
            "errors": []
        }
        subject_errors = []

        subject_ksame_output_list = k_same_results.get(subject_id)

        if not subject_ksame_output_list or all(item is None for item in subject_ksame_output_list):
            msg = f"Aucun résultat K-Same valide trouvé pour sujet {subject_id}. PEEP sauté."
            logger.warning(msg)
            pipeline_results[subject_id]['errors'].append(msg)
            continue

        # Préparer le stack aplati à partir des résultats K-Same valides
        flattened_stack_list = []
        valid_indices_map = {}
        original_indices_processed_ksame = []

        for idx, ksame_item in enumerate(subject_ksame_output_list):
             if ksame_item and 'flattened_anonymized_image' in ksame_item:
                 flattened_anon = ksame_item['flattened_anonymized_image']
                 if flattened_anon is not None and flattened_anon.ndim == 1:
                    flattened_stack_list.append(flattened_anon)
                    valid_indices_map[idx] = len(flattened_stack_list) - 1
                    original_indices_processed_ksame.append(idx)
                 else:
                    msg = f"Sujet {subject_id}, Idx Orig {idx}: Donnée K-Same invalide (None, pas 1D, ou manquante)."
                    logger.warning(msg)
                    subject_errors.append(msg + f" Image ID: {ksame_item.get('imageId', 'N/A')}")

        if len(flattened_stack_list) < 2:
            msg = f"Sujet {subject_id}: Moins de 2 images valides après K-Same ({len(flattened_stack_list)}). PEEP sauté."
            logger.warning(msg)
            pipeline_results[subject_id]['errors'].append(msg)
            pipeline_results[subject_id]['errors'].extend(subject_errors)
            continue

        flattened_stack_np = np.array(flattened_stack_list, dtype=np.float32)
        n_samples, n_features = flattened_stack_np.shape

        n_components = min(max(1, int(n_components_ratio * n_samples)), n_features)
        logger.debug(f"Sujet {subject_id}: PCA avec n_components={n_components} sur stack K-Same shape {flattened_stack_np.shape}")

        # Exécuter les étapes PEEP
        pca, _, projection = run_eigenface(flattened_stack_np, n_components)

        if pca is None or projection is None:
             msg = f"Sujet {subject_id}: Échec du calcul PCA/projection. PEEP arrêté pour ce sujet."
             logger.error(msg)
             pipeline_results[subject_id]['errors'].append(msg)
             pipeline_results[subject_id]['errors'].extend(subject_errors)
             continue

        noised_projection = run_add_noise(projection, epsilon)
        if noised_projection is None:
             msg = f"Sujet {subject_id}: Échec de l'ajout de bruit. PEEP arrêté pour ce sujet."
             logger.error(msg)
             pipeline_results[subject_id]['errors'].append(msg)
             pipeline_results[subject_id]['errors'].extend(subject_errors)
             continue

        reconstructed_b64_list_valid = run_reconstruction(pca, noised_projection, used_image_size)

        # Ré-insérer les images reconstruites aux bons indices
        if len(reconstructed_b64_list_valid) == len(valid_indices_map):
             for original_idx, stack_idx in valid_indices_map.items():
                  if stack_idx < len(reconstructed_b64_list_valid):
                     pipeline_results[subject_id]["final_reconstructed_b64"][original_idx] = reconstructed_b64_list_valid[stack_idx]
                  else:
                     msg = f"Sujet {subject_id}: Index de stack ({stack_idx}) hors limites pour les résultats reconstruits ({len(reconstructed_b64_list_valid)}). Erreur logique interne."
                     logger.error(msg)
                     subject_errors.append(msg)
        else:
             msg = (f"Sujet {subject_id}: Discordance de taille entre images reconstruites PEEP "
                    f"({len(reconstructed_b64_list_valid)}) et images valides K-Same traitées ({len(valid_indices_map)}). "
                    f"Reconstruction partielle ou nulle pour ce sujet.")
             logger.error(msg)
             subject_errors.append(msg)
             for original_idx, stack_idx in valid_indices_map.items():
                  if stack_idx < len(reconstructed_b64_list_valid):
                      pipeline_results[subject_id]["final_reconstructed_b64"][original_idx] = reconstructed_b64_list_valid[stack_idx]

        # --- Sauvegarde Finale ---
        save_final_reconstructed_images(
            subject_id,
            pipeline_results[subject_id]["final_reconstructed_b64"],
            pipeline_results[subject_id]["imageIds"]
        )

        pipeline_results[subject_id]['errors'].extend(subject_errors)

    logger.info("Pipeline SÉQUENTIELLE terminée.")
    # Nettoyer les résultats
    final_results = {sid: data for sid, data in pipeline_results.items() if any(img is not None for img in data["final_reconstructed_b64"])}
    if len(final_results) < len(pipeline_results):
        logger.warning(f"{len(pipeline_results) - len(final_results)} sujets n'ont produit aucune image finale valide et ont été exclus du retour.")

    return final_results