# Imports remain largely the same
import binascii
import os
import sys
import io
import base64
import logging
import time
import warnings
import datetime
import itertools
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
# Removed sklearn.datasets import from core logic - data loading is now separate
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Setting `channel_axis=-1`", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inputs have mismatched dtype", category=UserWarning)


# --- Adjust PYTHONPATH for local modules ---
# (Keep this section if your modules are in a 'src' directory)
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
# --- End PYTHONPATH Adjustment ---

# --- Import local modules ---
# (Keep this section, crucial for the pipeline)
try:
    from src.modules import anony_process_pipeline
    from src.modules import utils_image # Assuming this might be needed by pipeline
    from src.config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE # Default, but can be overridden
    _modules_imported = True
except ImportError as e:
    print(f"[ERREUR] Échec de l'importation des modules locaux : {e}")
    print("Vérifiez la structure de votre projet et que les fichiers __init__.py sont présents.")
    print(f"Chemin de recherche Python actuel : {sys.path}")
    _modules_imported = False
except FileNotFoundError:
    print("[ERREUR] Le fichier 'src/config.py' est introuvable. Vérifiez son existence.")
    _modules_imported = False
# Add specific handling if config is missing but we want to proceed with a default size
except Exception as e:
    print(f"[ERREUR] Erreur inattendue lors de l'importation : {e}")
    _modules_imported = False

if not _modules_imported:
      print("[CRITIQUE] Importations locales échouées. Arrêt du script.")
      sys.exit(1)
# --- End Imports ---

# --- Global Logger Setup ---
# Configure logger once globally
logger = logging.getLogger(__name__) # Use module-specific logger name
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# Avoid adding handlers multiple times if script/module is reloaded
if not logger.handlers:
    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)

    # File Handler (Optional - can be configured in the main script)
    # log_filename = f"grid_search_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    # file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    # file_handler.setFormatter(log_formatter)
    # logger.addHandler(file_handler)
    # logger.info(f"Logging initialisé. Sortie console et fichier : {os.path.abspath(log_filename)}")

# --- Utility Functions (Mostly Unchanged, but ensure they use logger) ---

def decode_b64_to_numpy(b64_string: str) -> Optional[np.ndarray]:
    """Décode une chaîne image base64 en tableau NumPy (niveaux de gris)."""
    if not isinstance(b64_string, str): return None
    try:
        img_bytes = base64.b64decode(b64_string)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('L')
        return np.array(img_pil, dtype=np.uint8)
    except (binascii.Error, UnidentifiedImageError, IOError, ValueError) as e:
        # logger.debug(f"Erreur décodage Base64 : {e}. String(début): {b64_string[:50]}...")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue décodage b64 : {e}", exc_info=False)
        return None

def preprocess_originals_for_metrics(
    df_input: pd.DataFrame,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    image_col: str = 'userFaces',
    id_col: str = 'imageId'
) -> Dict[str, np.ndarray]:
    """
    Prétraite les images originales du DataFrame pour la comparaison de métriques.
    Args:
        df_input: DataFrame contenant les images originales (PIL Images).
        target_size: Taille (height, width) cible pour le redimensionnement.
        image_col: Nom de la colonne contenant les objets Image PIL.
        id_col: Nom de la colonne contenant l'identifiant unique de l'image.
    Returns:
        Dictionnaire mappant imageId -> image_originale_numpy (prétraitée).
    """
    logger.info(f"Prétraitement des images originales pour calcul des métriques (taille cible: {target_size})...")
    original_images_map = {}

    if image_col not in df_input.columns or id_col not in df_input.columns:
        logger.error(f"Les colonnes requises ('{image_col}', '{id_col}') ne sont pas dans le DataFrame.")
        return {}

    # Utilise les fonctions d'aide de la pipeline si possible pour cohérence
    if not hasattr(anony_process_pipeline, 'preprocess_image'):
         logger.error("La fonction 'preprocess_image' est introuvable dans 'anony_process_pipeline'.")
         return {}

    for index, row in tqdm(df_input.iterrows(), total=df_input.shape[0], desc="Préparation originaux"):
        try:
            img_pil = row[image_col]
            img_id = str(row[id_col]) # Assure ID en string

            if not isinstance(img_pil, Image.Image):
                logger.warning(f"Image invalide (type: {type(img_pil)}) pour imageId {img_id}. Skip.")
                continue

            # Applique le prétraitement (grayscale, resize) sans aplatissement
            preprocessed_data = anony_process_pipeline.preprocess_image(
                img_pil,
                resize_size=target_size,
                create_flattened=False # Important: On veut l'image 2D
            )

            if preprocessed_data and 'grayscale_image' in preprocessed_data:
                 img_np = np.array(preprocessed_data['grayscale_image'], dtype=np.uint8)
                 original_images_map[img_id] = img_np
            else:
                 logger.warning(f"Échec du prétraitement initial pour imageId {img_id}. Skip.")

        except Exception as e:
            logger.error(f"Erreur lors du prétraitement de l'image originale {row.get(id_col, index)}: {e}", exc_info=False)

    processed_count = len(original_images_map)
    total_count = len(df_input)
    logger.info(f"{processed_count}/{total_count} images originales prétraitées et stockées pour comparaison.")
    if processed_count < total_count:
         logger.warning(f"{total_count - processed_count} images originales n'ont pas pu être prétraitées.")
    return original_images_map


def calculate_metrics_for_combination(
    pipeline_output: Dict[str, Dict[str, Any]],
    original_images_map: Dict[str, np.ndarray]
) -> Tuple[float, float]:
    """
    Calcule MSE et SSIM moyens pour une sortie de pipeline donnée.
    (Fonction largement inchangée, mais revue pour la robustesse)
    Returns:
        Tuple (avg_mse, avg_ssim). Retourne (np.nan, np.nan) si le calcul est impossible.
    """
    all_mse = []
    all_ssim = []
    processed_pairs_count = 0
    total_possible_pairs = 0

    if not pipeline_output:
        # logger.warning("Pipeline n'a retourné aucune sortie.") # Loggé dans la boucle principale
        return np.nan, np.nan
    if not original_images_map:
        logger.error("Map des images originales est vide. Impossible de calculer les métriques.")
        return np.nan, np.nan

    for subject_id, data in pipeline_output.items():
        reconstructed_b64_list = data.get("final_reconstructed_b64", [])
        image_ids_list = data.get("imageIds", [])
        total_possible_pairs += len(image_ids_list) # Compte basé sur les IDs d'entrée

        if len(reconstructed_b64_list) != len(image_ids_list):
            logger.warning(f"Sujet {subject_id}: Discordance entre images reconstruites ({len(reconstructed_b64_list)}) et IDs ({len(image_ids_list)}). Comparaison partielle possible.")
            # On continue, on essaiera de matcher ceux qui existent

        for i, img_id in enumerate(image_ids_list):
            # Vérifie si une image reconstruite existe à cet index
            if i >= len(reconstructed_b64_list) or reconstructed_b64_list[i] is None:
                # logger.debug(f"Pas d'image reconstruite pour imageId {img_id}. Skip paire.")
                continue

            recon_b64 = reconstructed_b64_list[i]
            img_orig_np = original_images_map.get(img_id)

            if img_orig_np is None:
                # logger.warning(f"Image originale non trouvée dans le map pour imageId {img_id}. Skip paire.")
                continue

            img_recon_np = decode_b64_to_numpy(recon_b64)
            if img_recon_np is None:
                # logger.debug(f"Échec décodage image reconstruite pour imageId {img_id}. Skip paire.")
                continue

            if img_orig_np.shape != img_recon_np.shape:
                logger.warning(f"Discordance de forme pour imageId {img_id}: Originale {img_orig_np.shape}, Reconstruite {img_recon_np.shape}. Skip paire.")
                continue

            # --- Calcul MSE ---
            try:
                mse = mean_squared_error(img_orig_np, img_recon_np)
                all_mse.append(mse)
            except Exception as e:
                logger.warning(f"Erreur calcul MSE pour imageId {img_id}: {e}. Skip MSE.")
                all_mse.append(np.nan)

            # --- Calcul SSIM ---
            try:
                min_dim = min(img_orig_np.shape)
                # win_size impair et <= min_dim
                win_size = min(7, min_dim) if min_dim >= 7 else (min_dim if min_dim % 2 != 0 else max(1, min_dim - 1))

                if win_size < 3:
                     logger.debug(f"Fenêtre SSIM trop petite ({win_size}) pour imageId {img_id}. Skip SSIM.")
                     ssim_val = np.nan
                else:
                     ssim_val = ssim(img_orig_np, img_recon_np,
                                     data_range=img_orig_np.max() - img_orig_np.min(), # Range dynamique ou 255 si uint8
                                     win_size=win_size,
                                     channel_axis=None) # Grayscale
                all_ssim.append(ssim_val)
                processed_pairs_count += 1

            except ValueError as e:
                logger.warning(f"Erreur valeur SSIM pour imageId {img_id} (win_size={win_size}): {e}. Skip SSIM.")
                all_ssim.append(np.nan)
            except Exception as e:
                logger.error(f"Erreur inattendue SSIM pour imageId {img_id}: {e}", exc_info=False) # Limit exc_info logging
                all_ssim.append(np.nan)

    avg_mse = np.nanmean(all_mse) if all_mse else np.nan
    avg_ssim = np.nanmean(all_ssim) if all_ssim else np.nan

    if total_possible_pairs > 0:
        logger.debug(f"Métriques calculées sur {processed_pairs_count}/{total_possible_pairs} paires valides. Avg MSE={avg_mse:.4f}, Avg SSIM={avg_ssim:.4f}")
    elif not pipeline_output:
         logger.debug("Aucune sortie de pipeline, donc aucune métrique calculée.")
    else:
         logger.debug("Aucune paire valide trouvée pour le calcul des métriques.")


    return avg_mse, avg_ssim

# --- MODULAR Grid Search Function ---

def run_parameter_grid_search(
    df_images: pd.DataFrame,
    original_images_map: Dict[str, np.ndarray],
    k_values: List[int],
    ratio_values: List[float],
    epsilon_values: List[float],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    image_col: str = 'userFaces', # Ajout pour spécifier la colonne image
    id_col: str = 'imageId',       # Ajout pour spécifier la colonne ID
    subject_col: str = 'subject_number' # Ajout pour spécifier la colonne sujet
) -> Optional[pd.DataFrame]:
    """
    Exécute un Grid Search 3D sur k, n_component_ratio, et epsilon pour la pipeline d'anonymisation.

    Args:
        df_images: DataFrame contenant les données d'images. Doit avoir au moins
                   les colonnes spécifiées par image_col, id_col, subject_col.
                   La colonne image_col doit contenir des objets PIL.Image.
        original_images_map: Dictionnaire pré-calculé mappant imageId -> image originale prétraitée (np.ndarray).
        k_values: Liste des valeurs de k (pour k-same-k) à tester.
        ratio_values: Liste des valeurs de n_component_ratio (pour PCA) à tester.
        epsilon_values: Liste des valeurs d'epsilon (pour bruit Laplacien) à tester.
        image_size: Taille (height, width) à utiliser pour le redimensionnement dans la pipeline.
        image_col: Nom de la colonne contenant les objets Image PIL dans df_images.
        id_col: Nom de la colonne contenant l'identifiant unique de l'image dans df_images.
        subject_col: Nom de la colonne contenant l'identifiant du sujet dans df_images.


    Returns:
        DataFrame Pandas contenant les résultats (paramètres, avg_mse, avg_ssim) pour chaque combinaison,
        ou None si une erreur majeure survient.
    """
    logger.info(f"--- DÉMARRAGE DU GRID SEARCH PARAMÉTRISÉ (k, ratio, epsilon) ---")
    logger.info(f"Paramètres K      : {k_values}")
    logger.info(f"Paramètres Ratio  : {ratio_values}")
    logger.info(f"Paramètres Epsilon: {epsilon_values}")
    logger.info(f"Taille image cible: {image_size}")

    # Validation des entrées
    required_cols = [image_col, id_col, subject_col]
    if df_images is None or df_images.empty:
        logger.error("Le DataFrame d'images fourni est vide. Arrêt.")
        return None
    if not all(col in df_images.columns for col in required_cols):
        logger.error(f"Le DataFrame doit contenir les colonnes : {required_cols}. Colonnes trouvées : {df_images.columns.tolist()}. Arrêt.")
        return None
    if not original_images_map:
        logger.error("Le map des images originales est vide. Métriques non calculables. Arrêt.")
        return None
    if not hasattr(anony_process_pipeline, 'run_pipeline'):
        logger.error("La fonction 'run_pipeline' est introuvable dans 'anony_process_pipeline'. Arrêt.")
        return None


    param_combinations = list(itertools.product(k_values, ratio_values, epsilon_values))
    total_combinations = len(param_combinations)
    logger.info(f"Nombre total de combinaisons de paramètres à tester : {total_combinations}")

    if total_combinations == 0:
        logger.warning("Aucune combinaison de paramètres à tester.")
        return pd.DataFrame(columns=['k', 'n_component_ratio', 'epsilon', 'avg_mse', 'avg_ssim']) # Retourne un DF vide

    results_list = []
    pbar = tqdm(param_combinations, total=total_combinations, desc="Grid Search 3D")

    for k_val, ratio_val, eps_val in pbar:
        pbar.set_postfix_str(f"k={k_val}, r={ratio_val:.2f}, ε={eps_val:.2f}")
        start_time_comb = time.time()
        pipeline_output = None
        avg_mse, avg_ssim = np.nan, np.nan # Default to NaN

        try:
            # Crée une copie légère pour la pipeline si elle modifie en place (bonne pratique)
            # Note: La pipeline devrait idéalement ne pas modifier df_images
            df_images_copy = df_images[[id_col, image_col, subject_col]].copy()

            # Exécute la pipeline complète
            # Assurez-vous que run_pipeline accepte bien ces noms de colonnes ou adaptez
            pipeline_output = anony_process_pipeline.run_pipeline(
                df_images=df_images_copy,
                k_same_k_value=k_val,
                n_components_ratio=ratio_val,
                epsilon=eps_val,
                image_size_override=image_size, # Utilise la taille fournie
                # Potentiellement passer les noms de colonnes si la pipeline les utilise
                # image_id_col=id_col,
                # face_col=image_col,
                # subject_id_col=subject_col
            )

            # Calcule les métriques si la pipeline a retourné quelque chose
            if pipeline_output is not None:
                 avg_mse, avg_ssim = calculate_metrics_for_combination(
                     pipeline_output,
                     original_images_map
                 )
            else:
                 logger.warning(f"La pipeline n'a rien retourné pour k={k_val}, r={ratio_val:.2f}, ε={eps_val:.2f}. Métriques non calculées.")


        except Exception as e:
            logger.error(f"Erreur critique PENDANT l'exécution de la pipeline pour k={k_val}, r={ratio_val:.2f}, ε={eps_val:.2f}: {e}", exc_info=True)
            # avg_mse, avg_ssim restent NaN

        finally:
            # Stocke les résultats (même en cas d'erreur, avec NaN)
            results_list.append({
                'k': k_val,
                'n_component_ratio': ratio_val,
                'epsilon': eps_val,
                'avg_mse': avg_mse,
                'avg_ssim': avg_ssim
            })
            end_time_comb = time.time()
            # logger.debug(f"Combinaison (k={k_val}, r={ratio_val:.1f}, ε={eps_val:.1f}) traitée en {end_time_comb - start_time_comb:.2f} sec. MSE={avg_mse:.4f}, SSIM={avg_ssim:.4f}")


    if not results_list:
        logger.error("Aucun résultat n'a été collecté pendant le Grid Search.")
        return None

    df_results = pd.DataFrame(results_list)
    logger.info("--- GRID SEARCH PARAMÉTRISÉ TERMINÉ ---")
    return df_results

# --- (Optional) Data Loading Example (LFW specific - keep outside core logic) ---
def load_and_prepare_lfw_data(
    min_faces: int,
    n_samples: int,
    target_image_size: Tuple[int, int], # Nécessaire pour le prétraitement initial
    image_col: str = 'userFaces',
    id_col: str = 'imageId',
    subject_col: str = 'subject_number'
) -> Optional[Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
    """
    Charge et prépare les données LFW spécifiquement pour le format requis
    par `run_parameter_grid_search`. Inclut le prétraitement des originaux.

    Returns:
        Tuple (df_prepared, original_images_map) ou None en cas d'erreur.
    """
    logger.info(f"Chargement et préparation des données LFW (min_faces={min_faces}, n_samples={n_samples})...")
    try:
        # Importation locale pour garder la fonction autonome si LFW n'est pas toujours utilisé
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        logger.critical("Scikit-learn n'est pas installé. Impossible de charger LFW. `pip install scikit-learn`")
        return None

    try:
        # Tente de charger LFW
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4, color=False) # Charge en grayscale
        native_h, native_w = lfw_people.images.shape[1], lfw_people.images.shape[2]
        logger.info(f"Images LFW chargées ({len(lfw_people.data)} images brutes) avec shape native (après resize 0.4): ({native_h}, {native_w})")

        # Normalisation et type
        data = lfw_people.data.clip(0, 255).astype(np.uint8) # Assure uint8 [0, 255]

        df = pd.DataFrame(data)
        df['subject_id_temp'] = lfw_people.target # ID numérique du sujet LFW

        # Équilibrage et échantillonnage
        grouped = df.groupby('subject_id_temp')
        balanced_dfs = []
        logger.info("Échantillonnage pour équilibrer le dataset...")
        for subject_id, group in tqdm(grouped, desc="Échantillonnage par sujet"):
            if len(group) >= n_samples:
                balanced_dfs.append(group.sample(n=n_samples, random_state=42)) # Sample without replacement by default if n <= len(group)

        if not balanced_dfs:
            logger.error(f"Aucun sujet trouvé avec au moins {n_samples} images après chargement.")
            return None

        df_balanced = pd.concat(balanced_dfs).reset_index(drop=True)
        logger.info(f"{df_balanced['subject_id_temp'].nunique()} sujets échantillonnés, {len(df_balanced)} images au total.")

        # Conversion en objets Image PIL
        def row_to_pil(row_pixels):
            pixels = row_pixels.values.astype(np.uint8) # Assure le bon type
            return Image.fromarray(pixels.reshape((native_h, native_w)), mode='L') # 'L' pour grayscale

        logger.info("Conversion des lignes de pixels en objets Image PIL...")
        pixel_columns = list(range(native_h * native_w))
        df_balanced[image_col] = df_balanced[pixel_columns].apply(row_to_pil, axis=1)

        # Création d'un ID unique et sélection/renommage des colonnes
        df_balanced[id_col] = df_balanced.index.astype(str)
        df_final = df_balanced[[image_col, id_col, 'subject_id_temp']].copy()
        df_final.rename(columns={'subject_id_temp': subject_col}, inplace=True)

        logger.info(f"DataFrame LFW préparé : {df_final.shape[0]} images ({df_final[subject_col].nunique()} sujets).")

        # --- Étape cruciale : Prétraiter les originaux pour la comparaison ---
        # Utilise la taille cible fournie pour cette étape !
        original_images_map = preprocess_originals_for_metrics(
            df_final,
            target_size=target_image_size,
            image_col=image_col,
            id_col=id_col
        )

        if not original_images_map:
            logger.error("Échec du pré-traitement des images originales après chargement LFW.")
            return None

        return df_final, original_images_map

    except FileNotFoundError:
        logger.critical("Données LFW non trouvées. Essayez de les télécharger manuellement ou vérifiez votre installation scikit-learn.")
        return None
    except Exception as e:
        logger.critical(f"Erreur critique lors du chargement/préparation de LFW : {e}", exc_info=True)
        return None


# --- Main Execution Example ---
def run_analysis(
    k_vals: List[int],
    ratio_vals: List[float],
    eps_vals: List[float],
    output_dir: str = "analysis_results_grid_search_search_3d",
    image_size_for_pipeline: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    use_lfw: bool = True, # Flag pour utiliser LFW ou charger des données autrement
    lfw_min_faces: int = 20,
    lfw_n_samples: int = 10,
    # Ajoutez ici des paramètres si vous chargez des données custom
    # custom_data_path: Optional[str] = None
):
    """
    Orchestre l'exécution de l'analyse de grid search.
    """
    start_global_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Configurer le gestionnaire de fichiers log ici si désiré
    log_filename = f"grid_search_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(output_dir, log_filename)
    try:
        # Vérifier si le gestionnaire de fichiers existe déjà pour éviter les doublons
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging fichier activé: {os.path.abspath(log_filepath)}")
        else:
             logger.info(f"Logging fichier déjà configuré (probablement vers {log_filepath})")
    except Exception as e:
        logger.error(f"Impossible de créer le fichier log à {log_filepath}: {e}")

    logger.info("==========================================================")
    logger.info("===== DÉMARRAGE SCRIPT ANALYSE GRID SEARCH MODULAIRE =====")
    logger.info(f"Répertoire de sortie : {os.path.abspath(output_dir)}")
    logger.info("==========================================================")

    # 1. Charger et préparer les données
    df_input_images = None
    originals_map = None

    if use_lfw:
        data_load_result = load_and_prepare_lfw_data(
            min_faces=lfw_min_faces,
            n_samples=lfw_n_samples,
            target_image_size=image_size_for_pipeline # Important: la taille utilisée PARTOUT
        )
        if data_load_result:
            df_input_images, originals_map = data_load_result
    else:
        # === SECTION POUR CHARGER VOS PROPRES DONNÉES ===
        logger.info("Chargement de données personnalisées (logique à implémenter)...")
        # Exemple :
        # df_input_images = load_my_custom_images(custom_data_path) # Doit retourner un DataFrame avec PIL Images, imageId, subject_number
        # if df_input_images is not None:
        #     originals_map = preprocess_originals_for_metrics(
        #         df_input_images,
        #         target_size=image_size_for_pipeline,
        #         image_col='ma_colonne_image', # Adaptez les noms de colonnes
        #         id_col='mon_id_image'
        #     )
        # =============================================
        logger.error("Chargement de données personnalisées non implémenté dans cet exemple.")
        # Pour l'instant, on arrête si LFW n'est pas utilisé
        return


    if df_input_images is None or originals_map is None:
        logger.critical("Échec du chargement ou de la préparation des données. Arrêt du script.")
        return

    # 2. Exécuter le Grid Search Modulaire
    df_grid_results = run_parameter_grid_search(
        df_images=df_input_images,
        original_images_map=originals_map,
        k_values=k_vals,
        ratio_values=ratio_vals,
        epsilon_values=eps_vals,
        image_size=image_size_for_pipeline,
        # Assurez-vous que ces noms de colonnes correspondent à ceux de votre DataFrame préparé
        image_col='userFaces',
        id_col='imageId',
        subject_col='subject_number'
    )

    # 3. Sauvegarder et analyser (optionnel) les résultats
    if df_grid_results is not None:
        logger.info("\n--- APERÇU DES RÉSULTATS DU GRID SEARCH ---")
        with pd.option_context('display.max_rows', 10, 'display.float_format', '{:.4f}'.format):
            print(df_grid_results)
        logger.info(f"Nombre total de résultats collectés : {len(df_grid_results)}")

        csv_filename = "grid_search_3d_results.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        try:
            df_grid_results.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Résultats complets sauvegardés dans : {os.path.abspath(csv_path)}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du fichier CSV ({csv_path}): {e}")

        # --- Analyse simple des résultats ---
        if not df_grid_results.empty:
             # Meilleur SSIM (compromis utilité - plus proche de l'original)
             if 'avg_ssim' in df_grid_results.columns and df_grid_results['avg_ssim'].notna().any():
                 try:
                     best_ssim_row = df_grid_results.loc[df_grid_results['avg_ssim'].idxmax()]
                     logger.info("\n--- Combinaison avec le MEILLEUR SSIM (max) ---")
                     print(best_ssim_row)
                 except Exception as e: logger.error(f"Erreur recherche SSIM max: {e}")

             # Pire SSIM (compromis vie privée - plus différent de l'original)
             if 'avg_ssim' in df_grid_results.columns and df_grid_results['avg_ssim'].notna().any():
                 try:
                      worst_ssim_row = df_grid_results.loc[df_grid_results['avg_ssim'].idxmin()]
                      logger.info("\n--- Combinaison avec le PIRE SSIM (min) ---")
                      print(worst_ssim_row)
                 except Exception as e: logger.error(f"Erreur recherche SSIM min: {e}")

             # Meilleur MSE (compromis utilité - erreur la plus basse)
             if 'avg_mse' in df_grid_results.columns and df_grid_results['avg_mse'].notna().any():
                 try:
                      best_mse_row = df_grid_results.loc[df_grid_results['avg_mse'].idxmin()]
                      logger.info("\n--- Combinaison avec le MEILLEUR MSE (min) ---")
                      print(best_mse_row)
                 except Exception as e: logger.error(f"Erreur recherche MSE min: {e}")
    else:
        logger.error("Le Grid Search n'a pas retourné de DataFrame de résultats.")

    end_global_time = time.time()
    logger.info("==========================================================")
    logger.info("===== ANALYSE GRID SEARCH MODULAIRE TERMINÉE =====")
    logger.info(f"Temps d'exécution total : {end_global_time - start_global_time:.2f} secondes.")
    logger.info(f"Résultats et logs dans : {os.path.abspath(output_dir)}")
    logger.info("Prochaines étapes : Analyser le CSV, visualiser (heatmaps?), inspection visuelle.")
    logger.info("==========================================================")


if __name__ == "__main__":
    # --- Définissez vos paramètres et lancez l'analyse ici ---

    # 1. Définir les listes de paramètres à tester
    ks_to_test = [2, 5, 10] # Exemple
    ratios_to_test = np.linspace(0.1, 0.7, 4).tolist() # Exemple: [0.1, 0.3, 0.5, 0.7]
    epsilons_to_test = [0.1, 0.5, 1.0] # Exemple

    # 2. Définir la taille d'image (doit être cohérente)
    #    Utilise la valeur de config par défaut ou spécifiez une autre taille
    pipeline_image_size = DEFAULT_IMAGE_SIZE # ou par ex., (64, 64)

    # 3. Configurer le répertoire de sortie
    results_directory = "analysis_results_custom_run"

    # 4. Choisir la source de données (LFW ou custom)
    use_lfw_data = True
    lfw_params = {
        "lfw_min_faces": 15, # Moins strict pour tester
        "lfw_n_samples": 5   # Moins d'échantillons pour tester
    } if use_lfw_data else {}

    # 5. Lancer l'analyse principale
    run_analysis(
        k_vals=ks_to_test,
        ratio_vals=ratios_to_test,
        eps_vals=epsilons_to_test,
        output_dir=results_directory,
        image_size_for_pipeline=pipeline_image_size,
        use_lfw=use_lfw_data,
        **lfw_params # Passe les paramètres LFW si use_lfw_data est True
    )