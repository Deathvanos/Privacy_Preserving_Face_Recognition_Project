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

try:
    from src.modules.image_preprocessing import preprocess_image, flatten_image
    import src.modules.k_same_pixel as ksp
    from src.modules.eigenface import EigenfaceGenerator
    from src.modules.noise_generator import NoiseGenerator
    from src.modules.utils_image import image_pillow_to_bytes, image_numpy_to_pillow
    from src.config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE
except ImportError:
    print("WARN: Using fallback imports for modules.")
    from image_preprocessing import preprocess_image, flatten_image
    import k_same_pixel as ksp
    from eigenface import EigenfaceGenerator
    from noise_generator import NoiseGenerator
    from utils_image import image_pillow_to_bytes, image_numpy_to_pillow
    DEFAULT_IMAGE_SIZE = (100, 100) # Taille par défaut

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FINAL_OUTPUT_DIR = "../../data/anonymized_output"
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
logger.info(f"Les images finales (K-Same -> Preproc -> PEEP -> Recons) seront enregistrées dans : {FINAL_OUTPUT_DIR}")

# --- Fonction de Sauvegarde Finale ---
def save_final_image(subject_id: str, image_index: int, b64img: str, output_dir: str):
    if b64img is None: return
    try:
        img_bytes = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(img_bytes))
        file_name = f"final_subject_{subject_id}_image_{image_index}.png"
        file_path = os.path.join(output_dir, file_name)
        img.save(file_path, format='PNG')
    except Exception as e:
        logger.error(f"Erreur sauvegarde image finale index {image_index} sujet {subject_id} dans {output_dir}: {e}")

# --- Fonctions utilitaires internes ---
def initial_load_and_prep(
    folder_path: Optional[str] = None,
    df_images: Optional[pd.DataFrame] = None,
    b64_image_list: Optional[List[str]] = None,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
) -> Dict[str, List[Dict[str, Any]]]:
    num_sources = sum(p is not None for p in [folder_path, df_images, b64_image_list])
    if num_sources != 1: raise ValueError("Fournir exactement une source d'images.")
    logger.info(f"Étape 0: Chargement initial et préparation minimale (taille={image_size}, grayscale)...")
    initial_groups = defaultdict(list)

    if df_images is not None:
        logger.info(f"Traitement de {len(df_images)} images depuis DataFrame.")
        for index, row in tqdm(df_images.iterrows(), total=df_images.shape[0], desc="Chargement initial (DataFrame)"):
            try:
                img = row.get('userFaces')
                subject_id = str(row.get('subject_number', 'unknown_subject'))
                image_id = row.get('imageId', f'df_img_{index}')
                if img is None: continue
                if not isinstance(img, Image.Image):
                    if isinstance(img, np.ndarray) and img.ndim == 2: img = Image.fromarray(img)
                    else: continue
                resized_img = img.resize(image_size)
                grayscale_pil = resized_img.convert("L")
                initial_groups[subject_id].append({'imageId': image_id, 'grayscale_pil_image': grayscale_pil})
            except Exception as e: logger.error(f"Erreur chargement initial index {index}: {e}")
    elif folder_path is not None:
        logger.info(f"Traitement images depuis dossier: {folder_path}")
        all_files = os.listdir(folder_path); image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.pgm'))]
        for filename in tqdm(image_files, desc="Chargement initial (Dossier)"):
             try:
                 subject_id = filename.split('_')[1]
                 image_id = os.path.splitext(filename)[0]
                 img_path = os.path.join(folder_path, filename)
                 with Image.open(img_path) as img:
                      resized_img = img.resize(image_size)
                      grayscale_pil = resized_img.convert("L")
                      initial_groups[subject_id].append({'imageId': image_id, 'grayscale_pil_image': grayscale_pil})
             except UnidentifiedImageError: logger.warning(f"Fichier '{filename}' non reconnu comme image.")
             except Exception as e: logger.error(f"Erreur chargement initial fichier '{filename}': {e}")
    elif b64_image_list is not None:
        logger.info(f"Traitement de {len(b64_image_list)} images depuis liste Base64.")
        subject_id = "single_subject_b64"
        for i, b64_string in enumerate(tqdm(b64_image_list, desc="Chargement initial (Base64)")):
            try:
                image_id = f"b64_img_{i}"
                img_bytes = base64.b64decode(b64_string); img = Image.open(io.BytesIO(img_bytes))
                resized_img = img.resize(image_size)
                grayscale_pil = resized_img.convert("L")
                initial_groups[subject_id].append({'imageId': image_id, 'grayscale_pil_image': grayscale_pil})
            except UnidentifiedImageError: logger.warning(f"Image base64 index {i} non reconnue.")
            except Exception as e: logger.error(f"Erreur chargement initial image base64 index {i}: {e}")

    logger.info(f"Chargement initial terminé. {len(initial_groups)} sujets préparés pour K-Same.")
    return dict(initial_groups)

def run_ksame_on_initial(initial_groups: Dict[str, List[Dict[str, Any]]], k_value: int) -> Dict[str, List[Optional[np.ndarray]]]:
    logger.info(f"Étape 1: Application de K-Same-Pixel avec k={k_value}...")
    ksame_results_arrays = defaultdict(list)

    if not initial_groups:
        logger.warning("initial_groups est vide, K-Same Pixel sauté.")
        return {}

    for subject_id, subject_initial_list in tqdm(initial_groups.items(), desc="K-Same Pixel"):
        subject_output_arrays = [None] * len(subject_initial_list)

        if len(subject_initial_list) < k_value:
            logger.warning(f"Sujet {subject_id}: Moins d'images initiales ({len(subject_initial_list)}) que k={k_value}. K-Same Pixel sauté.")
            ksame_results_arrays[subject_id] = subject_output_arrays
            continue

        k_same_input_list = []
        id_order_map = {}

        for idx, img_dict in enumerate(subject_initial_list):
            img_id = img_dict.get('imageId')
            grayscale_pil_image = img_dict.get('grayscale_pil_image')

            if img_id is None or not isinstance(grayscale_pil_image, Image.Image):
                 logger.warning(f"Image index {idx} sujet {subject_id} manque 'imageId' ou 'grayscale_pil_image' valide. Skip pour K-Same.")
                 continue

            try:
                 grayscale_np = np.array(grayscale_pil_image, dtype=np.uint8)
                 k_same_input_list.append((grayscale_np, img_id))
                 id_order_map[img_id] = idx
            except Exception as conv_err:
                 logger.error(f"Erreur conversion PIL->NumPy pour K-Same img {img_id} (sujet {subject_id}): {conv_err}")

        if not k_same_input_list or len(k_same_input_list) < k_value:
             logger.warning(f"Pas assez d'images valides préparées ({len(k_same_input_list)}) pour K-Same sujet {subject_id} (k={k_value}).")
             ksame_results_arrays[subject_id] = subject_output_arrays
             continue

        try:
            k_same_output_tuples = ksp.k_same_pixel_individual(k_same_input_list, k=k_value)
        except Exception as k_err:
            logger.error(f"Erreur durant k_same_pixel_individual sujet {subject_id}: {k_err}", exc_info=True)
            ksame_results_arrays[subject_id] = subject_output_arrays
            continue

        id_to_anonymized_map = {img_id: anon_array for anon_array, img_id in k_same_output_tuples}
        for img_id, original_index in id_order_map.items():
             anon_array = id_to_anonymized_map.get(img_id)
             if anon_array is not None and isinstance(anon_array, np.ndarray) and anon_array.dtype == np.uint8:
                 subject_output_arrays[original_index] = anon_array
             elif anon_array is not None:
                 logger.warning(f"Sujet {subject_id} img {img_id}: K-Same a retourné un type/dtype inattendu ({type(anon_array)} / {anon_array.dtype if isinstance(anon_array, np.ndarray) else 'N/A'}). Tentative de conversion.")
                 try:
                     subject_output_arrays[original_index] = anon_array.astype(np.uint8)
                 except Exception:
                     logger.error(f"Échec conversion K-Same vers uint8 pour img {img_id}")


        ksame_results_arrays[subject_id] = subject_output_arrays

    logger.info("Traitement K-Same-Pixel terminé.")
    return dict(ksame_results_arrays)

def preprocess_ksame_results(
    ksame_arrays_dict: Dict[str, List[Optional[np.ndarray]]],
    initial_groups: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
    logger.info("Étape 2: Preprocessing des résultats K-Same (Normalisation, Aplatissement)...")
    processed_ksame_groups = defaultdict(list)

    for subject_id, subject_ksame_arrays in tqdm(ksame_arrays_dict.items(), desc="Preprocessing K-Same"):
        processed_list_for_subject = []
        original_image_ids = [d.get('imageId', f'unknown_{i}') for i, d in enumerate(initial_groups.get(subject_id, []))]

        num_images = len(original_image_ids)
        if len(subject_ksame_arrays) != num_images:
             logger.warning(f"Sujet {subject_id}: Discordance de taille entre K-Same ({len(subject_ksame_arrays)}) et initial ({num_images}). Troncation/Padding avec None.")
             temp_arrays = [None] * num_images
             for i in range(min(len(subject_ksame_arrays), num_images)):
                  temp_arrays[i] = subject_ksame_arrays[i]
             subject_ksame_arrays = temp_arrays

        for idx, ksame_array in enumerate(subject_ksame_arrays):
            image_id = original_image_ids[idx]
            flattened_output = None
            if ksame_array is not None and isinstance(ksame_array, np.ndarray):
                try:
                    # Normaliser (0-1) en float32 pour PCA
                    normalized_ksame = ksame_array.astype(np.float32) / 255.0
                    flattened_output = normalized_ksame.flatten()
                except Exception as e:
                    logger.error(f"Erreur preprocessing K-Same img {image_id} (sujet {subject_id}): {e}")
                    flattened_output = None
            processed_list_for_subject.append({'imageId': image_id, 'flattened_ksame': flattened_output})

        processed_ksame_groups[subject_id] = processed_list_for_subject

    logger.info("Preprocessing des résultats K-Same terminé.")
    return dict(processed_ksame_groups)


# --- Fonctions PEEP ---
def run_eigenface(flattened_stack: np.ndarray, n_components: int):
    logger.debug(f"Calcul Eigenfaces avec n_components={n_components}")
    n_samples = flattened_stack.shape[0]
    if n_components > n_samples:
        logger.warning(f"n_components ({n_components}) > n_samples ({n_samples}). Ajustement à {n_samples}.")
        n_components = n_samples
    if n_components <= 0:
        logger.error(f"Impossible de lancer PCA avec n_components={n_components}.")
        return None, None, None
    eigen_gen = EigenfaceGenerator(flattened_stack, n_components=n_components)
    try:
        eigen_gen.generate()
    except ValueError as e:
         logger.error(f"Erreur lors de la génération des eigenfaces: {e}")
         return None, None, None
    mean_face = eigen_gen.get_mean_face()
    pca = eigen_gen.get_pca_object()
    projection = pca.transform(flattened_stack)
    return pca, mean_face, projection

def run_add_noise(projection: np.ndarray, epsilon: float, sensitivity: float = 1.0):
    logger.debug(f"Ajout bruit Laplacien (epsilon={epsilon}, sensitivity={sensitivity})")
    if projection is None or projection.size == 0:
         logger.error("Projection invalide reçue pour ajout de bruit.")
         return None
    if epsilon <= 0:
        logger.warning(f"Epsilon ({epsilon}) <= 0, aucun bruit Laplacien ne sera ajouté.")
        return projection
    try:
        noise_gen = NoiseGenerator(projection, epsilon)
        noise_gen.normalize_images()
        noise_gen.add_laplace_noise(sensitivity)
        noised_projection = noise_gen.get_noised_eigenfaces()
        return noised_projection
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout du bruit Laplacien: {e}", exc_info=True)
        return None

def run_reconstruction(pca, projection_data: Optional[np.ndarray], image_size: Tuple[int, int]) -> List[Optional[str]]:
    """Reconstruit des images Base64 à partir d'une projection (bruitée ou non)."""
    if pca is None:
         logger.error("Objet PCA manquant pour la reconstruction.")
         return []
    if projection_data is None:
         logger.error("Données de projection manquantes pour la reconstruction.")
         return []
    logger.debug(f"Reconstruction images depuis projection (shape={projection_data.shape})...")
    try:
        if projection_data.shape[1] != pca.n_components_:
             logger.error(f"Discordance de dimensions pour inverse_transform: projection a {projection_data.shape[1]} features, PCA attend {pca.n_components_}.")
             return []
        reconstructions = pca.inverse_transform(projection_data)
    except ValueError as e:
        logger.error(f"Erreur lors de pca.inverse_transform: {e}")
        return []
    reconstructed_images_b64 = []
    expected_shape = (image_size[1], image_size[0])
    for recon_flat in reconstructions:
        try:
            recon_flat_denorm = np.clip(recon_flat * 255.0, 0, 255)
            pil_img = Image.fromarray(recon_flat_denorm.reshape(expected_shape).astype(np.uint8), mode='L')
            b64_img = image_pillow_to_bytes(pil_img)
            reconstructed_images_b64.append(b64_img)
        except Exception as e:
            logger.error(f"Erreur reconversion image reconstruite: {e}", exc_info=True)
            reconstructed_images_b64.append(None)
    return reconstructed_images_b64


# --- Fonction globale pipeline ---
def run_pipeline(
    folder_path: Optional[str] = None,
    df_images: Optional[pd.DataFrame] = None,
    b64_image_list: Optional[List[str]] = None,
    epsilon: float = 9.0,
    n_components_ratio: float = 0.9,
    k_same_k_value: Optional[int] = None,
    image_size_override: Optional[Tuple[int, int]] = None
) -> dict:
    """
    Exécute la pipeline: InitialLoad -> K-Same -> Preproc(K-Same) -> PEEP -> Reconstruct -> Save.
    Retourne un dictionnaire incluant les résultats intermédiaires de K-Same et PEEP (PCA seul).

    Args:
        image_size_override (Optional[Tuple[int, int]]): Tuple (largeur, hauteur) pour remplacer la taille d'image par défaut.

    Returns:
        dict: Dictionnaire par subject_id contenant:
              - 'imageIds': Liste des IDs originaux.
              - 'k_same_intermediate_b64': Liste des images (Base64) après K-Same.
              - 'mean_face': Image moyenne (Base64) calculée par PCA.
              - 'projection': Coefficients de projection après PCA (avant bruit).
              - 'noised_projection': Coefficients de projection après ajout de bruit.
              - 'reconstructed_pca_only_b64': Liste images (Base64) reconstruites depuis la projection PCA (avant bruit).
              - 'reconstructed_final_b64': Liste images finales (Base64) après ajout de bruit et reconstruction.
    """
    pipeline_result = {}
    effective_image_size = image_size_override if image_size_override is not None else DEFAULT_IMAGE_SIZE
    logger.info(f"Démarrage pipeline SÉQUENTIELLE: eps={epsilon}, ratio={n_components_ratio}, k={k_same_k_value}, image_size={effective_image_size}")

    if k_same_k_value is None or k_same_k_value <= 0:
        logger.error("k_same_k_value doit être fourni et > 0. Arrêt.")
        return {}

    # --- Étape 0 : Chargement Initial ---
    initial_groups = initial_load_and_prep(
        folder_path=folder_path, df_images=df_images, b64_image_list=b64_image_list, image_size=effective_image_size
    )
    if not initial_groups:
        logger.error("Chargement initial n'a retourné aucune image. Arrêt.")
        return {}
    num_subjects = len(initial_groups)
    total_initial_images = sum(len(v) for v in initial_groups.values())
    logger.info(f"Chargement initial OK: {total_initial_images} images pour {num_subjects} sujets.")

    # --- Étape 1 : K-Same Pixel ---
    k_same_results_arrays = run_ksame_on_initial(initial_groups, k_same_k_value)
    if not k_same_results_arrays:
         logger.warning("K-Same n'a retourné aucun résultat array.")
         return {}

    # --- Conversion K-Same en Base64 pour le retour ---
    logger.info("Conversion des résultats K-Same (intermédiaires) en Base64...")
    ksame_intermediate_b64_results = defaultdict(list)
    for subject_id, subject_arrays in tqdm(k_same_results_arrays.items(), desc="Conversion K-Same->B64"):
        b64_list_for_subject = []
        num_expected = len(initial_groups.get(subject_id, []))
        temp_arrays = subject_arrays if len(subject_arrays) == num_expected else ([None] * num_expected)

        for arr in temp_arrays:
            b64_img = None
            if arr is not None and isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
                try:
                    pil_img = image_numpy_to_pillow(arr)
                    b64_img = image_pillow_to_bytes(pil_img)
                except Exception as e:
                    logger.error(f"Erreur conversion K-Same array (sujet {subject_id}) en B64: {e}")
            b64_list_for_subject.append(b64_img)
        ksame_intermediate_b64_results[subject_id] = b64_list_for_subject
    logger.info("Conversion K-Same -> B64 terminée.")


    # --- Étape 2 : Preprocessing sur les résultats K-Same ---
    processed_ksame_groups = preprocess_ksame_results(k_same_results_arrays, initial_groups)
    if not processed_ksame_groups:
         logger.error("Preprocessing des résultats K-Same a échoué. Arrêt.")
         return {}

    # --- Étapes 3-5 : PEEP (Eigenface, Bruit, Reconstruction) ---
    peep_results = {} # Stockera tous les outputs PEEP par sujet

    logger.info("Démarrage des étapes PEEP (sur K-Same prétraité)...")
    for subject_id in tqdm(processed_ksame_groups, desc="Traitement PEEP par sujet"):
        subject_processed_ksame_list = processed_ksame_groups[subject_id]
        num_initial_images_for_subject = len(subject_processed_ksame_list)

        peep_subject_output = {
            "mean_face_b64": None,
            "projection": [],
            "noised_projection": [],
            "reconstructed_pca_only_b64": [None] * num_initial_images_for_subject,
            "reconstructed_final_b64": [None] * num_initial_images_for_subject
        }

        flattened_stack_list = []
        valid_indices_map = {}

        for idx, data_dict in enumerate(subject_processed_ksame_list):
             flattened_ksame = data_dict.get('flattened_ksame')
             if flattened_ksame is not None and isinstance(flattened_ksame, np.ndarray):
                 expected_flat_size = effective_image_size[0] * effective_image_size[1]
                 if flattened_ksame.shape[0] == expected_flat_size:
                     flattened_stack_list.append(flattened_ksame)
                     valid_indices_map[idx] = len(flattened_stack_list) - 1
                 else:
                     logger.error(f"Sujet {subject_id} img idx {idx}: Taille aplatie inattendue ({flattened_ksame.shape[0]}) vs attendue ({expected_flat_size}). Skip pour PEEP.")

        num_valid_for_peep = len(flattened_stack_list)
        if num_valid_for_peep < 2:
            logger.warning(f"Sujet {subject_id}: Moins de 2 images valides/taille correcte ({num_valid_for_peep}). PEEP sauté.")
            peep_results[subject_id] = peep_subject_output # Stocke les Nones/listes vides
            continue

        flattened_stack_np = np.array(flattened_stack_list, dtype=np.float32) # Assurer float32
        n_samples, n_features = flattened_stack_np.shape
        # Ajustement n_components pour éviter n_components > n_features ou n_samples
        n_components = min(max(1, int(n_components_ratio * n_samples)), n_features, n_samples)

        if n_components <= 0:
             logger.error(f"Sujet {subject_id}: n_components={n_components} invalide. PEEP sauté.")
             peep_results[subject_id] = peep_subject_output
             continue

        logger.debug(f"Sujet {subject_id}: PCA avec n_components={n_components} sur {n_samples} échantillons valides.")
        try:
            # --- Exécution PEEP Core ---
            pca, mean_face, projection = run_eigenface(flattened_stack_np, n_components)

            if pca is None:
                 logger.error(f"Sujet {subject_id}: Échec run_eigenface. PEEP interrompu.")
                 peep_results[subject_id] = peep_subject_output
                 continue # Passe au sujet suivant

            # --- Stockage Mean Face et Projection Originale ---
            if mean_face is not None:
                 mean_face_denorm = np.clip(mean_face * 255.0, 0, 255)
                 mean_face_img = image_numpy_to_pillow(mean_face_denorm.reshape(effective_image_size[1], effective_image_size[0]).astype(np.uint8))
                 peep_subject_output["mean_face_b64"] = image_pillow_to_bytes(mean_face_img)
            peep_subject_output["projection"] = projection.tolist() if projection is not None else []

            # --- [NOUVEAU] Reconstruction PCA Seule (avant bruit) ---
            reconstructed_pca_only_b64_list_valid = run_reconstruction(pca, projection, effective_image_size)
            temp_recon_pca_only = [None] * num_initial_images_for_subject
            if len(reconstructed_pca_only_b64_list_valid) == len(valid_indices_map):
                 for original_idx, stack_idx in valid_indices_map.items():
                      if stack_idx < len(reconstructed_pca_only_b64_list_valid):
                           temp_recon_pca_only[original_idx] = reconstructed_pca_only_b64_list_valid[stack_idx]
                 peep_subject_output["reconstructed_pca_only_b64"] = temp_recon_pca_only
            else:
                 logger.error(f"Sujet {subject_id}: Discordance taille PCA reconstruit ({len(reconstructed_pca_only_b64_list_valid)}) vs valides ({len(valid_indices_map)}).")
                 # Remplissage partiel possible mais complexe à gérer pour l'utilisateur final

            # --- Ajout Bruit ---
            noised_projection = run_add_noise(projection, epsilon)
            peep_subject_output["noised_projection"] = noised_projection.tolist() if noised_projection is not None else []

            # --- Reconstruction Finale (après bruit) ---
            reconstructed_final_b64_list_valid = run_reconstruction(pca, noised_projection, effective_image_size)
            temp_recon_final = [None] * num_initial_images_for_subject
            if len(reconstructed_final_b64_list_valid) == len(valid_indices_map):
                 for original_idx, stack_idx in valid_indices_map.items():
                      if stack_idx < len(reconstructed_final_b64_list_valid):
                           temp_recon_final[original_idx] = reconstructed_final_b64_list_valid[stack_idx]
                 peep_subject_output["reconstructed_final_b64"] = temp_recon_final
            else:
                 logger.error(f"Sujet {subject_id}: Discordance taille finale reconstruit ({len(reconstructed_final_b64_list_valid)}) vs valides ({len(valid_indices_map)}).")

            # --- SAUVEGARDE DISQUE (uniquement résultat final) ---
            saved_count_subject = 0
            for img_idx, final_b64 in enumerate(peep_subject_output["reconstructed_final_b64"]):
                if final_b64 is not None:
                    save_final_image(subject_id, img_idx, final_b64, FINAL_OUTPUT_DIR)
                    saved_count_subject += 1
            if saved_count_subject > 0:
                 logger.info(f"{saved_count_subject} images finales enregistrées pour sujet {subject_id} dans {FINAL_OUTPUT_DIR}")

        except Exception as peep_err:
            logger.error(f"Erreur majeure durant PEEP pour sujet {subject_id}: {peep_err}", exc_info=True)
            # peep_subject_output garde ses valeurs initialisées (None/listes vides)

        peep_results[subject_id] = peep_subject_output # Stocker les résultats PEEP pour ce sujet

    # --- Assemblage Final du Résultat (incluant intermédiaires) ---
    logger.info("Assemblage du résultat final (dictionnaire)...")
    for subject_id, subject_initial_list in initial_groups.items():
        num_images_subj = len(subject_initial_list)
        # Récupérer les résultats K-Same B64
        ksame_b64 = ksame_intermediate_b64_results.get(subject_id, [None] * num_images_subj)
        # Récupérer les résultats PEEP (ou valeurs par défaut si PEEP a échoué)
        peep_output = peep_results.get(subject_id, {
            "mean_face_b64": None, "projection": [], "noised_projection": [],
            "reconstructed_pca_only_b64": [None] * num_images_subj,
            "reconstructed_final_b64": [None] * num_images_subj
        })
        original_image_ids = [d.get('imageId', f'unknown_{i}') for i, d in enumerate(subject_initial_list)]

        pipeline_result[subject_id] = {
            "imageIds": original_image_ids,
            # K-Same
            "k_same_intermediate_b64": ksame_b64,
            # PEEP
            "mean_face": peep_output.get("mean_face_b64"),
            "projection": peep_output.get("projection", []),
            "noised_projection": peep_output.get("noised_projection", []),
            "reconstructed_pca_only_b64": peep_output.get("reconstructed_pca_only_b64", [None] * num_images_subj), # PCA seul
            "reconstructed_final_b64": peep_output.get("reconstructed_final_b64", [None] * num_images_subj) # Final
        }
        # Vérification rapide des tailles de listes retournées
        for key in ["k_same_intermediate_b64", "reconstructed_pca_only_b64", "reconstructed_final_b64"]:
            if len(pipeline_result[subject_id][key]) != num_images_subj:
                 logger.warning(f"Sujet {subject_id}: Taille incohérente pour '{key}' dans le résultat final ({len(pipeline_result[subject_id][key])} vs {num_images_subj} attendus).")


    logger.info(f"Pipeline SÉQUENTIELLE terminée. Images finales enregistrées dans {FINAL_OUTPUT_DIR}. Dictionnaire de résultats complet généré.")
    return pipeline_result
