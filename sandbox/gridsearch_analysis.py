# -*- coding: utf-8 -*-
import binascii
import os
import sys
import io
import base64
import logging
import time
import warnings
import datetime
import itertools # Ajout pour le produit cartésien des paramètres

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# Ignorer les avertissements spécifiques de scikit-image si nécessaire
warnings.filterwarnings("ignore", message="Setting `channel_axis=-1`", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inputs have mismatched dtype", category=UserWarning)


# --- Adjust PYTHONPATH for local modules ---
# S'assure que le script peut trouver les modules dans le dossier 'src' parent
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
# --- End PYTHONPATH Adjustment ---

# --- Import local modules ---
try:
    # Tente d'importer les modules nécessaires depuis la structure attendue
    from src.modules import anony_process_pipeline
    from src.modules import utils_image # On aura besoin de fonctions utilitaires
    from src.config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE # Taille d'image par défaut
    _modules_imported = True
except ImportError as e:
    print(f"[ERREUR] Échec de l'importation des modules locaux : {e}")
    print("Vérifiez la structure de votre projet et que les fichiers __init__.py sont présents.")
    print(f"Chemin de recherche Python actuel : {sys.path}")
    _modules_imported = False
except FileNotFoundError:
    print("[ERREUR] Le fichier 'src/config.py' est introuvable. Vérifiez son existence.")
    _modules_imported = False
except Exception as e:
    print(f"[ERREUR] Erreur inattendue lors de l'importation : {e}")
    _modules_imported = False


if not _modules_imported:
      print("[CRITIQUE] Importations locales échouées. Arrêt du script.")
      sys.exit(1) # Arrête le script si les imports échouent
# --- End Imports ---

# --- Global Constants and Parameters ---
# Dataset Parameters (Specific to LFW)
MIN_FACES_PER_PERSON = 20 # Minimum de visages par personne pour LFW
N_SAMPLES_PER_PERSON = 10 # Nombre d'images à échantillonner par personne

# === Grid Search Parameters ===
K_VALUES = np.linspace(2, 10, 5, dtype=int).tolist()
RATIO_VALUES = np.linspace(0.1, 0.90, 10).tolist()
EPSILON_VALUES = np.linspace(0.1, 1.0, 20).tolist()
# ==============================

# Output directory for results
OUTPUT_DIR = "analysis_results_grid_search_3d" # Nom du dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Constants ---

# --- Logging Configuration (Console and File with UTF-8 Encoding) ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Niveau de log (INFO, DEBUG, WARNING, ERROR, CRITICAL)

# Supprime les anciens handlers pour éviter les logs dupliqués si le script est relancé
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Handler pour la console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Handler pour le fichier log
log_filename = f"grid_search_3d_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(OUTPUT_DIR, log_filename)
try:
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging initialisé. Sortie console et fichier : {os.path.abspath(log_filepath)}")
except Exception as e:
    logger.error(f"Impossible de créer le fichier log à {log_filepath}: {e}")

logger.info(f"Répertoire de sortie pour les résultats : {os.path.abspath(OUTPUT_DIR)}")
# --- End Logging Configuration ---


# --- Utility Functions ---

def load_lfw_dataframe_for_analysis(min_faces: int, n_samples: int) -> pd.DataFrame | None:
    """Charge et prépare le DataFrame LFW pour l'analyse (version robuste)."""
    logger.info(f"Chargement du dataset LFW (min_faces={min_faces}, n_samples={n_samples})...")
    try:
        # Charge les données, en niveaux de gris, avec redimensionnement
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4, color=False)
        height, width = lfw_people.images.shape[1], lfw_people.images.shape[2]
        logger.info(f"Images LFW chargées avec shape originale (après resize): ({height}, {width})")

        # S'assure que les données sont en uint8 [0, 255]
        data = lfw_people.data
        if data.max() <= 1.0 + 1e-6: # Si les données sont entre [0, 1]
            data = (data * 255).clip(0, 255).astype(np.uint8)
        else: # Sinon, on s'assure juste qu'elles sont dans la bonne plage
            data = data.clip(0, 255).astype(np.uint8)

        # Crée un DataFrame initial
        df = pd.DataFrame(data)
        df['subject_id'] = lfw_people.target # Ajoute l'ID du sujet

        # Équilibre le dataset en échantillonnant n_samples par sujet
        grouped = df.groupby('subject_id')
        balanced_dfs = []
        logger.info("Échantillonnage pour équilibrer le dataset...")
        for subject_id, group in tqdm(grouped, desc="Échantillonnage par sujet"):
            if len(group) >= n_samples:
                # Échantillonne sans remplacement
                balanced_dfs.append(group.sample(n=n_samples, random_state=42, replace=False))

        if not balanced_dfs:
            logger.error(f"Aucun sujet trouvé avec au moins {n_samples} images.")
            return None

        df_balanced = pd.concat(balanced_dfs)
        logger.info(f"{len(balanced_dfs)} sujets échantillonnés.")

        # Convertit les lignes de pixels en objets Image PIL
        def row_to_pil_image(row_pixels):
            # Remodèle la ligne de pixels en image 2D
            return Image.fromarray(row_pixels.values.reshape((height, width)), mode='L') # 'L' pour niveaux de gris

        logger.info("Conversion des lignes de pixels en objets Image PIL...")
        pixel_columns = list(range(height * width)) # Colonnes contenant les pixels
        # Applique la conversion à chaque ligne
        df_balanced['userFaces'] = df_balanced[pixel_columns].apply(row_to_pil_image, axis=1)

        # Réinitialise l'index et ajoute un imageId unique
        df_balanced = df_balanced.reset_index(drop=True)
        df_balanced['imageId'] = df_balanced.index.astype(str) # Utilise l'index comme ID unique (en string)

        # Sélectionne et renomme les colonnes finales
        df_final = df_balanced[['userFaces', 'imageId', 'subject_id']].copy()
        df_final.rename(columns={'subject_id': 'subject_number'}, inplace=True) # Renomme pour correspondre à la pipeline

        logger.info(f"DataFrame final prêt : {df_final.shape[0]} images ({df_final['subject_number'].nunique()} sujets).")
        return df_final

    except Exception as e:
        logger.critical(f"Erreur critique lors du chargement/préparation de LFW : {e}", exc_info=True)
        return None

def preprocess_originals_for_metrics(df_input: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Prétraite les images originales du DataFrame et retourne un mapping
    imageId -> image_originale_numpy (redimensionnée et grayscale).
    """
    logger.info("Prétraitement des images originales pour calcul des métriques...")
    original_images_map = {}
    target_size = DEFAULT_IMAGE_SIZE # Utilise la taille définie dans config

    for index, row in tqdm(df_input.iterrows(), total=df_input.shape[0], desc="Préparation originaux"):
        try:
            img_pil = row['userFaces']
            img_id = str(row['imageId']) # Assure que l'ID est une chaîne

            if not isinstance(img_pil, Image.Image):
                logger.warning(f"Image invalide pour imageId {img_id}. Skip.")
                continue

            # Applique le même prétraitement que la pipeline (grayscale, resize)
            # mais sans l'aplatissement. On utilise directement la fonction de la pipeline
            # pour assurer la cohérence.
            preprocessed_data = anony_process_pipeline.preprocess_image(img_pil, resize_size=target_size, create_flattened=False)

            if preprocessed_data and 'grayscale_image' in preprocessed_data:
                 # Convertit l'image PIL grayscale en NumPy array
                 img_np = np.array(preprocessed_data['grayscale_image'], dtype=np.uint8)
                 original_images_map[img_id] = img_np
            else:
                 logger.warning(f"Échec du prétraitement initial pour imageId {img_id}. Skip.")

        except Exception as e:
            logger.error(f"Erreur lors du prétraitement de l'image originale {row.get('imageId', index)}: {e}", exc_info=False)

    logger.info(f"{len(original_images_map)} images originales prétraitées et stockées pour comparaison.")
    if len(original_images_map) < len(df_input):
         logger.warning(f"{len(df_input) - len(original_images_map)} images originales n'ont pas pu être prétraitées.")
    return original_images_map


def decode_b64_to_numpy(b64_string: str) -> np.ndarray | None:
    """Décode une chaîne image base64 en tableau NumPy (niveaux de gris)."""
    if not isinstance(b64_string, str): return None # Vérifie si c'est bien une chaîne
    try:
        img_bytes = base64.b64decode(b64_string)
        # Ouvre l'image et s'assure qu'elle est en niveaux de gris ('L')
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('L')
        return np.array(img_pil, dtype=np.uint8) # Convertit en NumPy array
    except (binascii.Error, UnidentifiedImageError, IOError, ValueError) as e:
        # Log l'erreur mais ne pollue pas trop les logs si ça arrive souvent
        # logger.debug(f"Erreur décodage Base64 : {e}. String(début): {b64_string[:50]}...")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue décodage b64 : {e}", exc_info=False)
        return None

def calculate_metrics_for_combination(
    pipeline_output: dict[str, dict[str, any]],
    original_images_map: dict[str, np.ndarray]
) -> tuple[float, float]:
    """
    Calcule MSE et SSIM moyens pour une combinaison de paramètres donnée.
    Prend la sortie de la pipeline et le map des images originales prétraitées.
    """
    all_mse = []
    all_ssim = []
    processed_pairs_count = 0
    total_reconstructed_count = 0

    if not pipeline_output:
        # logger.warning("Pipeline n'a retourné aucune sortie pour cette combinaison.")
        return np.nan, np.nan
    if not original_images_map:
        logger.error("Map des images originales est vide. Impossible de calculer les métriques.")
        return np.nan, np.nan

    # Itère sur chaque sujet dans la sortie de la pipeline
    for subject_id, data in pipeline_output.items():
        reconstructed_b64_list = data.get("final_reconstructed_b64", [])
        image_ids_list = data.get("imageIds", [])
        total_reconstructed_count += len(reconstructed_b64_list)

        if len(reconstructed_b64_list) != len(image_ids_list):
            logger.warning(f"Sujet {subject_id}: Discordance entre nombre d'images reconstruites ({len(reconstructed_b64_list)}) et IDs ({len(image_ids_list)}). Skip sujet.")
            continue

        # Itère sur chaque image reconstruite pour ce sujet
        for i, recon_b64 in enumerate(reconstructed_b64_list):
            img_id = image_ids_list[i]

            if recon_b64 is None:
                # logger.debug(f"Image reconstruite est None pour imageId {img_id}. Skip paire.")
                continue # Image non reconstruite, on ne peut pas calculer

            # Récupère l'image originale correspondante depuis le map
            img_orig_np = original_images_map.get(img_id)
            if img_orig_np is None:
                # logger.warning(f"Image originale non trouvée dans le map pour imageId {img_id}. Skip paire.")
                continue # Originale non trouvée

            # Décode l'image reconstruite
            img_recon_np = decode_b64_to_numpy(recon_b64)
            if img_recon_np is None:
                # logger.debug(f"Échec décodage image reconstruite pour imageId {img_id}. Skip paire.")
                continue # Échec décodage

            # --- Vérification de forme ---
            if img_orig_np.shape != img_recon_np.shape:
                logger.warning(f"Discordance de forme pour imageId {img_id}: Originale {img_orig_np.shape}, Reconstruite {img_recon_np.shape}. Skip paire.")
                continue

            # --- Calcul MSE ---
            try:
                mse = mean_squared_error(img_orig_np, img_recon_np)
                all_mse.append(mse)
            except Exception as e:
                logger.warning(f"Erreur calcul MSE pour imageId {img_id}: {e}. Skip MSE.")
                all_mse.append(np.nan) # Ajoute NaN si erreur

            # --- Calcul SSIM ---
            try:
                # Détermine win_size dynamiquement pour éviter les erreurs
                min_dim = min(img_orig_np.shape)
                # win_size doit être impair et <= min_dim
                win_size = min(7, min_dim) if min_dim >= 7 else (min_dim if min_dim % 2 != 0 else max(1, min_dim - 1))

                if win_size < 3: # SSIM non fiable pour très petites fenêtres
                     logger.debug(f"Fenêtre SSIM trop petite ({win_size}) pour imageId {img_id}. Skip SSIM.")
                     ssim_val = np.nan
                else:
                     # data_range: valeur max possible (255 pour uint8)
                     # channel_axis=None car image grayscale
                     ssim_val = ssim(img_orig_np, img_recon_np,
                                     data_range=255.0,
                                     win_size=win_size,
                                     channel_axis=None) # Important pour grayscale
                all_ssim.append(ssim_val)
                processed_pairs_count += 1 # Compte seulement si SSIM a pu être calculé (ou tenté)

            except ValueError as e: # Erreurs spécifiques SSIM (ex: win_size > image size)
                logger.warning(f"Erreur valeur SSIM pour imageId {img_id} (win_size={win_size}): {e}. Skip SSIM.")
                all_ssim.append(np.nan)
            except Exception as e:
                logger.error(f"Erreur inattendue SSIM pour imageId {img_id}: {e}", exc_info=True)
                all_ssim.append(np.nan)

    # Calcule les moyennes globales en ignorant les NaN
    avg_mse = np.nanmean(all_mse) if all_mse else np.nan
    avg_ssim = np.nanmean(all_ssim) if all_ssim else np.nan

    # Log un résumé si utile (peut être commenté si trop verbeux)
    # logger.debug(f"Métriques calculées : {processed_pairs_count}/{total_reconstructed_count} paires valides. Avg MSE={avg_mse:.4f}, Avg SSIM={avg_ssim:.4f}")

    return avg_mse, avg_ssim
# --- End Utility Functions ---


# --- Main Grid Search Function ---
def run_3d_grid_search(df_images: pd.DataFrame, original_images_map: dict[str, np.ndarray]):
    """
    Exécute le Grid Search 3D sur k, n_component_ratio, et epsilon.
    """
    logger.info("--- DÉMARRAGE DU GRID SEARCH 3D (k, ratio, epsilon) ---")
    if df_images is None or df_images.empty:
        logger.error("DataFrame d'images est vide. Arrêt du Grid Search.")
        return None
    if not original_images_map:
        logger.error("Map des images originales est vide. Arrêt du Grid Search.")
        return None

    # Crée toutes les combinaisons de paramètres
    param_combinations = list(itertools.product(K_VALUES, RATIO_VALUES, EPSILON_VALUES))
    total_combinations = len(param_combinations)
    logger.info(f"Nombre total de combinaisons de paramètres à tester : {total_combinations}")

    results_list = [] # Pour stocker les résultats de chaque combinaison

    # Barre de progression pour suivre l'avancement
    pbar = tqdm(param_combinations, total=total_combinations, desc="Grid Search 3D")

    # Boucle sur chaque combinaison de paramètres
    for k_val, ratio_val, eps_val in pbar:
        # Met à jour la description de la barre de progression
        pbar.set_postfix_str(f"k={k_val}, r={ratio_val:.1f}, ε={eps_val:.1f}")
        # logger.info(f"Test combinaison : k={k_val}, ratio={ratio_val:.2f}, epsilon={eps_val:.2f}")
        start_time_comb = time.time()
        pipeline_output = None # Réinitialise la sortie pour cette combinaison

        try:
            # Crée une copie du DataFrame pour éviter modifications inattendues par la pipeline
            # Bien que la pipeline actuelle ne semble pas modifier le df d'entrée, c'est une bonne pratique.
            df_images_copy = df_images.copy()

            # Exécute la pipeline d'anonymisation complète
            pipeline_output = anony_process_pipeline.run_pipeline(
                df_images=df_images_copy, # Utilise la copie
                k_same_k_value=k_val,
                n_components_ratio=ratio_val,
                epsilon=eps_val,
                image_size_override=DEFAULT_IMAGE_SIZE # Assure la cohérence de taille
            )

            # Calcule les métriques pour cette combinaison
            avg_mse, avg_ssim = calculate_metrics_for_combination(pipeline_output, original_images_map)

            # Stocke les résultats
            results_list.append({
                'k': k_val,
                'n_component_ratio': ratio_val,
                'epsilon': eps_val,
                'avg_mse': avg_mse,
                'avg_ssim': avg_ssim
            })

            end_time_comb = time.time()
            # logger.debug(f"Combinaison (k={k_val}, r={ratio_val:.1f}, ε={eps_val:.1f}) terminée en {end_time_comb - start_time_comb:.2f} sec. MSE={avg_mse:.4f}, SSIM={avg_ssim:.4f}")

        except Exception as e:
            logger.error(f"Erreur critique lors de l'exécution de la pipeline pour k={k_val}, ratio={ratio_val:.2f}, epsilon={eps_val:.2f}: {e}", exc_info=True)
            # Ajoute un résultat avec NaN en cas d'erreur pour ne pas perdre la trace de la combinaison échouée
            results_list.append({
                'k': k_val,
                'n_component_ratio': ratio_val,
                'epsilon': eps_val,
                'avg_mse': np.nan,
                'avg_ssim': np.nan
            })
            continue # Passe à la combinaison suivante

    # --- Fin de la boucle Grid Search ---

    if not results_list:
        logger.error("Aucun résultat n'a été collecté pendant le Grid Search.")
        return None

    # Convertit la liste de résultats en DataFrame Pandas
    df_results = pd.DataFrame(results_list)

    logger.info("\n--- APERÇU DES RÉSULTATS DU GRID SEARCH 3D ---")
    # Affiche les 5 premières et dernières lignes pour un aperçu rapide
    with pd.option_context('display.max_rows', 10):
        print(df_results)
    logger.info(f"Nombre total de résultats collectés : {len(df_results)}")

    # Sauvegarde le DataFrame complet dans un fichier CSV unique
    csv_filename = "grid_search_3d_results.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    try:
        df_results.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Résultats complets du Grid Search 3D sauvegardés dans : {os.path.abspath(csv_path)}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier CSV ({csv_path}): {e}", exc_info=True)

    logger.info("--- GRID SEARCH 3D TERMINÉ ---")
    return df_results # Retourne le DataFrame pour d'éventuelles analyses/visualisations ultérieures

# --- Main Execution Function ---
def main():
    """Orchestre l'exécution du script d'analyse."""
    start_global_time = time.time()
    logger.info("==========================================================")
    logger.info("===== DÉMARRAGE SCRIPT ANALYSE GRID SEARCH 3D =====")
    logger.info("==========================================================")

    # 1. Charger les données
    df_lfw = load_lfw_dataframe_for_analysis(min_faces=MIN_FACES_PER_PERSON, n_samples=N_SAMPLES_PER_PERSON)
    if df_lfw is None:
        logger.critical("Échec du chargement des données LFW. Arrêt du script.")
        return # Arrête si les données ne sont pas chargées

    # 2. Prétraiter les originaux UNE SEULE FOIS pour référence
    original_images_map = preprocess_originals_for_metrics(df_lfw)
    if not original_images_map:
         logger.critical("Échec du pré-traitement des images originales pour référence. Arrêt du script.")
         return

    # 3. Exécuter le Grid Search 3D
    df_grid_results = run_3d_grid_search(df_lfw, original_images_map)

    # 4. (Optionnel) Ajouter ici des étapes d'analyse ou de visualisation des df_grid_results
    if df_grid_results is not None:
        logger.info("Analyse terminée. Le fichier CSV contient les données pour l'analyse des compromis.")
        # Exemple simple : trouver la combinaison avec le SSIM le plus bas
        if not df_grid_results.empty and 'avg_ssim' in df_grid_results.columns:
             try:
                  min_ssim_row = df_grid_results.loc[df_grid_results['avg_ssim'].idxmin()]
                  logger.info("\n--- Exemple d'analyse : Combinaison avec le SSIM le plus bas ---")
                  print(min_ssim_row)
             except ValueError:
                  logger.warning("Impossible de trouver le SSIM minimum (peut-être que toutes les valeurs sont NaN).")
             except Exception as e:
                  logger.error(f"Erreur lors de la recherche du SSIM minimum: {e}")

        # Exemple simple : trouver la combinaison respectant des seuils (si définis)
        # SSIM_PRIVACY_THRESHOLD = 0.3
        # MSE_UTILITY_THRESHOLD = 2000
        # candidates = df_grid_results[
        #     (df_grid_results['avg_ssim'] < SSIM_PRIVACY_THRESHOLD) &
        #     (df_grid_results['avg_mse'] < MSE_UTILITY_THRESHOLD)
        # ]
        # logger.info(f"\n--- Exemple d'analyse : {len(candidates)} combinaisons respectant SSIM<{SSIM_PRIVACY_THRESHOLD} et MSE<{MSE_UTILITY_THRESHOLD} ---")
        # with pd.option_context('display.max_rows', 10):
        #      print(candidates.sort_values(by='avg_ssim')) # Affiche les 10 meilleurs candidats triés par SSIM

    else:
        logger.error("Le Grid Search n'a pas retourné de résultats.")

    end_global_time = time.time()
    logger.info("==========================================================")
    logger.info("===== ANALYSE GRID SEARCH 3D TERMINÉE =====")
    logger.info(f"Temps d'exécution total : {end_global_time - start_global_time:.2f} secondes.")
    logger.info(f"Résultats et logs sauvegardés dans le répertoire : {os.path.abspath(OUTPUT_DIR)}")
    logger.info("Prochaines étapes : Analyser le fichier CSV, visualiser les résultats (heatmaps, etc.),")
    logger.info("et surtout, effectuer une inspection visuelle et une évaluation ML sur les candidats prometteurs.")
    logger.info("==========================================================")

if __name__ == "__main__":
    # Point d'entrée principal du script
    main()
