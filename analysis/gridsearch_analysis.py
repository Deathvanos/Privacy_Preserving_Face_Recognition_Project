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
from tqdm import tqdm # Barre de progression
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message="Setting `channel_axis=-1`", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inputs have mismatched dtype", category=UserWarning)

# --- Adjust PYTHONPATH for local modules ---
# Assure que les modules dans 'src' peuvent être importés
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')) # Ajusté pour pointer vers src
if module_path not in sys.path:
    # Insère au début pour prioriser les modules locaux
    sys.path.insert(0, module_path)
    print(f"Added to sys.path: {module_path}") # Debug print

# --- Import local modules ---
try:
    # Assurez-vous que ces imports fonctionnent correctement avec votre structure de projet
    from modules import anony_process_pipeline
    from modules import utils_image
    from modules import image_preprocessing
    from modules import data_loader # Pour charger les données LFW ou custom
    from config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE
    from config import LFW_DATASET_PATH, CUSTOM_DATASET_PATH # Assurez-vous que ces chemins sont corrects dans config.py
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que le PYTHONPATH est correct ou que le script est lancé depuis le bon répertoire.")
    sys.exit(1)

# --- CONSTANTES DE SEUILS ---
DEFAULT_SSIM_THRESHOLD = 0.45
DEFAULT_MSE_THRESHOLD = 1500

# --- Configuration du Logging ---
# (Le logging sera configuré plus précisément dans le bloc main)
logger = logging.getLogger(__name__)


def setup_logging(log_dir: str) -> None:
    """Configure le logging pour fichier et console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"analysis_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Afficher aussi sur la console
        ]
    )

def load_data_sample(
    use_lfw: bool,
    image_size: Tuple[int, int],
    lfw_config: Optional[Dict] = None,
    custom_data_path: Optional[str] = None,
    n_samples: int = 5
) -> List[np.ndarray]:
    """Charge un échantillon d'images (LFW ou custom)."""
    images_sample = []
    if use_lfw:
        logger.info("Chargement des données LFW...")
        lfw_loader = data_loader.LFWLoader(
            lfw_path=LFW_DATASET_PATH, # Depuis config.py
            min_faces_per_person=lfw_config.get("lfw_min_faces", 10), # Utiliser config ou défaut
            target_size=image_size,
            slice_=(slice(70, 195), slice(78, 172)), # Utiliser config ou défaut?
            color_mode='grayscale', # Supposer grayscale pour SSIM/MSE simples
            resize_interpolation=Image.Resampling.LANCZOS # Utiliser config ou défaut
        )
        dataset = lfw_loader.load_dataset()
        if not dataset:
            logger.error("Aucune donnée LFW chargée. Vérifiez la configuration.")
            return []

        # Prendre un échantillon aléatoire parmi toutes les images LFW chargées
        all_lfw_images = [img for person_imgs in dataset.values() for img in person_imgs]
        if len(all_lfw_images) >= n_samples:
             # Sélection aléatoire sans remplacement
            indices = np.random.choice(len(all_lfw_images), n_samples, replace=False)
            images_sample = [all_lfw_images[i] for i in indices]
        else:
            logger.warning(f"Moins d'images LFW ({len(all_lfw_images)}) que demandé ({n_samples}). Utilisation de toutes les images.")
            images_sample = all_lfw_images
        logger.info(f"{len(images_sample)} images LFW chargées pour l'échantillon.")

    else:
        logger.info(f"Chargement des données depuis le dossier custom: {custom_data_path}")
        if not custom_data_path or not os.path.isdir(custom_data_path):
            logger.error(f"Chemin de données custom invalide: {custom_data_path}")
            return []

        custom_loader = data_loader.CustomDataLoader(
            dataset_path=custom_data_path,
            target_size=image_size,
            color_mode='grayscale' # Supposer grayscale
        )
        # Charger seulement N samples aléatoires
        all_files = custom_loader.find_image_files()
        if len(all_files) >= n_samples:
            sample_files = np.random.choice(all_files, n_samples, replace=False)
        else:
            logger.warning(f"Moins d'images custom ({len(all_files)}) que demandé ({n_samples}). Utilisation de toutes les images.")
            sample_files = all_files

        images_sample = [custom_loader.load_image(f) for f in sample_files if custom_loader.load_image(f) is not None]
        logger.info(f"{len(images_sample)} images custom chargées pour l'échantillon.")

    return images_sample


def run_analysis(
    ks: List[int],
    pca_ratios: List[float],
    epsilons: List[float],
    image_size: Tuple[int, int],
    output_dir: str,
    use_lfw: bool = True,
    lfw_config: Optional[Dict] = None,
    custom_data_path: Optional[str] = None,
    n_samples: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Exécute la Grid Search, évalue les combinaisons, sauvegarde les résultats,
    et identifie la meilleure combinaison basée sur les seuils SSIM/MSE.

    Retourne:
        Le dictionnaire de la meilleure combinaison trouvée, ou None si aucune ne respecte les seuils.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info("--- Démarrage de l'analyse Grid Search ---")
    logger.info(f"Paramètres à tester:")
    logger.info(f"  k: {ks}")
    logger.info(f"  PCA Ratios: {[f'{r:.3f}' for r in pca_ratios]}")
    logger.info(f"  Epsilons: {epsilons}")
    logger.info(f"Seuils: SSIM >= {DEFAULT_SSIM_THRESHOLD}, MSE <= {DEFAULT_MSE_THRESHOLD}")
    logger.info(f"Taille image pipeline: {image_size}")
    logger.info(f"Source de données: {'LFW' if use_lfw else 'Custom'}")
    logger.info(f"Nombre d'images par échantillon: {n_samples}")
    logger.info(f"Répertoire de sortie: {os.path.abspath(output_dir)}")

    # --- Chargement des données échantillon ---
    original_images_sample = load_data_sample(
        use_lfw=use_lfw,
        image_size=image_size,
        lfw_config=lfw_config,
        custom_data_path=custom_data_path,
        n_samples=n_samples
    )
    if not original_images_sample:
        logger.error("Échec du chargement des données échantillon. Arrêt de l'analyse.")
        return None

    # --- Initialisation de l'objet pipeline (sera réutilisé avec différents params) ---
    # Note: L'entraînement PCA doit être fait une seule fois si possible,
    # mais ici on le ré-entraîne pour chaque ratio pour simplifier la boucle.
    # Une optimisation serait de pré-entraîner PCA et de sélectionner les composantes.

    # --- Préparation des combinaisons ---
    param_combinations = list(itertools.product(ks, pca_ratios, epsilons))
    logger.info(f"Nombre total de combinaisons à tester: {len(param_combinations)}")

    # --- Collecte des résultats ---
    all_results = []
    start_time_grid = time.time()

    # --- Boucle Principale ---
    for k_val, pca_ratio, epsilon_val in tqdm(param_combinations, desc="Grid Search Progress", unit="combinaison"):
        combination_start_time = time.time()
        ssim_scores = []
        mse_scores = []
        error_occured = False
        error_msg = ""

        try:
            # Instancier le pipeline pour cette combinaison
            # Note: Le modèle EigenFace est re-fit à chaque fois ici.
            pipeline = anony_process_pipeline.AnonymizerPipeline(
                k_value=k_val,
                pca_variance_ratio=pca_ratio,
                epsilon=epsilon_val,
                image_shape=image_size, # Important pour EigenFace et KSamePixel
                fit_pca_on_sample=original_images_sample # Fit PCA sur l'échantillon
            )

            # Appliquer sur l'échantillon d'images
            for original_img_array in original_images_sample:
                # Assurer que l'image est dans le bon format (ex: uint8, grayscale)
                if original_img_array is None:
                    logger.warning("Image échantillon None rencontrée, ignorée.")
                    continue

                # S'assurer que l'image a la bonne taille (devrait déjà être le cas si load_data fait bien son travail)
                if original_img_array.shape[:2] != image_size:
                     img_pil = Image.fromarray(original_img_array)
                     img_resized = img_pil.resize(image_size, Image.Resampling.LANCZOS)
                     original_img_array = np.array(img_resized)

                 # Vérifier si l'image est déjà en niveaux de gris
                if len(original_img_array.shape) == 3 and original_img_array.shape[2] == 3:
                    # Convertir en niveaux de gris si nécessaire
                    prep_image = image_preprocessing.preprocess_image(original_img_array, target_size=image_size, grayscale=True)
                elif len(original_img_array.shape) == 2:
                    prep_image = original_img_array # Déjà en niveaux de gris
                else:
                     logger.warning(f"Format d'image inattendu: {original_img_array.shape}. Ignorée.")
                     continue

                # Appliquer toutes les étapes du pipeline
                anonymized_image_array = pipeline.apply_all(prep_image) # apply_all gère les étapes

                # Calculer les métriques (en s'assurant des types et range)
                # Convertir en float64 pour éviter les problèmes de type avec scikit-image/learn
                original_float = prep_image.astype(np.float64)
                anonymized_float = anonymized_image_array.astype(np.float64)

                # SSIM: data_range est important. channel_axis doit être None pour grayscale.
                current_ssim = ssim(original_float, anonymized_float,
                                    data_range=original_float.max() - original_float.min(), # Typiquement 255 pour uint8
                                    channel_axis=None) # Pas d'axe de canal pour grayscale

                # MSE
                current_mse = mean_squared_error(original_float, anonymized_float)

                ssim_scores.append(current_ssim)
                mse_scores.append(current_mse)

            # --- Agréger les métriques pour cette combinaison ---
            avg_ssim = np.mean(ssim_scores) if ssim_scores else np.nan
            avg_mse = np.mean(mse_scores) if mse_scores else np.nan
            processing_time = time.time() - combination_start_time

            # --- Stocker le résultat ---
            result_entry = {
                'k': k_val,
                'pca_ratio': pca_ratio,
                'epsilon': epsilon_val,
                'avg_ssim': avg_ssim,
                'avg_mse': avg_mse,
                'time_sec': processing_time,
                'n_samples_processed': len(ssim_scores),
                'error': None
            }
            all_results.append(result_entry)

        except Exception as e:
            logger.error(f"Erreur lors du traitement (k={k_val}, ratio={pca_ratio:.3f}, eps={epsilon_val}): {e}", exc_info=False) # exc_info=True pour traceback complet
            error_occured = True
            error_msg = str(e)
            all_results.append({
                'k': k_val, 'pca_ratio': pca_ratio, 'epsilon': epsilon_val,
                'avg_ssim': np.nan, 'avg_mse': np.nan,
                'time_sec': time.time() - combination_start_time,
                'n_samples_processed': 0,
                'error': error_msg
            })

    # --- Fin de la boucle Grid Search ---
    end_time_grid = time.time()
    total_duration = end_time_grid - start_time_grid
    logger.info(f"Grid search terminé en {total_duration:.2f} secondes.")

    # --- Traitement des résultats ---
    if not all_results:
        logger.warning("Aucun résultat n'a été collecté pendant la Grid Search.")
        return None

    results_df = pd.DataFrame(all_results)

    # --- Sauvegarde de TOUS les résultats (même ceux qui échouent ou hors seuils) ---
    csv_filename = f"grid_search_results_full_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    try:
        results_df.to_csv(csv_filepath, index=False, float_format='%.5f')
        logger.info(f"Résultats complets (incluant erreurs) sauvegardés dans : {csv_filepath}")
    except Exception as e:
        logger.error(f"Impossible de sauvegarder le fichier CSV complet : {e}")

    # Filtrer les résultats valides (non NaN) pour la sélection
    results_df_valid = results_df.dropna(subset=['avg_ssim', 'avg_mse']).copy()

    # --- Filtrage basé sur les seuils ---
    if not results_df_valid.empty:
        logger.info(f"Filtrage des {len(results_df_valid)} résultats valides avec SSIM >= {DEFAULT_SSIM_THRESHOLD} et MSE <= {DEFAULT_MSE_THRESHOLD}")
        valid_combinations_df = results_df_valid[
            (results_df_valid['avg_ssim'] >= DEFAULT_SSIM_THRESHOLD) &
            (results_df_valid['avg_mse'] <= DEFAULT_MSE_THRESHOLD)
        ]
    else:
        logger.warning("Aucun résultat valide (non-NaN) n'a été obtenu.")
        valid_combinations_df = pd.DataFrame() # DataFrame vide

    # --- Sélection de la meilleure combinaison ---
    best_combination_dict = None
    if not valid_combinations_df.empty:
        logger.info(f"{len(valid_combinations_df)} combinaisons respectent les seuils.")
        # Trier par SSIM descendant (meilleur), puis MSE ascendant (meilleur)
        valid_combinations_sorted = valid_combinations_df.sort_values(
            by=['avg_ssim', 'avg_mse'], ascending=[False, True]
        )
        # Prendre la première ligne (la meilleure)
        best_combination_row = valid_combinations_sorted.iloc[0]
        best_combination_dict = best_combination_row.to_dict()

        logger.info("==================== MEILLEURE COMBINAISON TROUVÉE ====================")
        logger.info(f"  Paramètres: k={best_combination_dict['k']}, ratio={best_combination_dict['pca_ratio']:.3f}, epsilon={best_combination_dict['epsilon']}")
        logger.info(f"  Métriques : SSIM={best_combination_dict['avg_ssim']:.4f} (Seuil >= {DEFAULT_SSIM_THRESHOLD})")
        logger.info(f"              MSE ={best_combination_dict['avg_mse']:.2f} (Seuil <= {DEFAULT_MSE_THRESHOLD})")
        logger.info(f"  Temps moyen par combinaison: {results_df_valid['time_sec'].mean():.2f}s")
        logger.info("=======================================================================")

    else:
        logger.warning("========================= ATTENTION ===========================")
        logger.warning(f"Aucune combinaison n'a respecté les seuils (SSIM >= {DEFAULT_SSIM_THRESHOLD} ET MSE <= {DEFAULT_MSE_THRESHOLD}).")
        logger.warning("Suggestions: Ajuster les seuils, élargir la plage des paramètres testés, ou vérifier les erreurs dans les logs.")
        # Afficher les meilleures valeurs globales obtenues pour info
        if not results_df_valid.empty:
             try:
                 best_ssim_overall = results_df_valid.loc[results_df_valid['avg_ssim'].idxmax()]
                 logger.info(f"  Meilleur SSIM obtenu (global): {best_ssim_overall['avg_ssim']:.4f} (avec k={best_ssim_overall['k']}, r={best_ssim_overall['pca_ratio']:.3f}, e={best_ssim_overall['epsilon']}, MSE={best_ssim_overall['avg_mse']:.2f})")
             except ValueError:
                 logger.info("  Impossible de déterminer le meilleur SSIM global (probablement que des NaN).")

             try:
                 best_mse_overall = results_df_valid.loc[results_df_valid['avg_mse'].idxmin()]
                 logger.info(f"  Meilleur MSE obtenu (global) : {best_mse_overall['avg_mse']:.2f} (avec k={best_mse_overall['k']}, r={best_mse_overall['pca_ratio']:.3f}, e={best_mse_overall['epsilon']}, SSIM={best_mse_overall['avg_ssim']:.4f})")
             except ValueError:
                 logger.info("  Impossible de déterminer le meilleur MSE global (probablement que des NaN).")
        logger.warning("=============================================================")

    logger.info(f"\nAnalyse terminée. Vérifiez le fichier CSV complet pour tous les détails: {csv_filepath}")
    logger.info("======================= FIN DE L'ANALYSE ======================\n")

    return best_combination_dict # Retourne le dictionnaire de la meilleure combinaison


# --- Point d'entrée principal ---
if __name__ == "__main__":

    # 1. Définir les listes de paramètres à tester
    ks_to_test = [2, 5, 10, 15] # Exemple étendu
    ratios_to_test = np.linspace(0.1, 0.8, 5).tolist() # Exemple: [0.1, 0.275, 0.45, 0.625, 0.8]
    epsilons_to_test = [0.1, 0.5, 1.0, 2.0, 5.0] # Exemple étendu

    # 2. Définir la taille d'image cible (utiliser celle de config.py par défaut)
    pipeline_image_size = DEFAULT_IMAGE_SIZE # Ex: (100, 100)

    # 3. Configurer le répertoire de sortie
    results_directory = "analysis_results_selection" # Nom du dossier de sortie

    # 4. Choisir la source de données (LFW ou custom) et le nombre d'échantillons
    use_lfw_data = True # Mettre à False pour utiliser custom
    number_of_samples = 10 # Nombre d'images à utiliser pour calculer SSIM/MSE moyen par combinaison
    lfw_parameters = {
        "lfw_min_faces": 10, # Minimum de visages par personne dans LFW
        # "slice_": (slice(70, 195), slice(78, 172)) # Décommenter et ajuster si besoin
    }
    custom_data_folder = CUSTOM_DATASET_PATH # Chemin vers dossier custom depuis config.py

    # 5. Configurer le logging
    setup_logging(os.path.join(results_directory, "logs"))

    # --- Lancer l'analyse ---
    best_params_found = run_analysis(
        ks=ks_to_test,
        pca_ratios=ratios_to_test,
        epsilons=epsilons_to_test,
        image_size=pipeline_image_size,
        output_dir=results_directory,
        use_lfw=use_lfw_data,
        lfw_config=lfw_parameters if use_lfw_data else None,
        custom_data_path=custom_data_folder if not use_lfw_data else None,
        n_samples=number_of_samples
    )

    # --- Conclusion ---
    if best_params_found:
        logger.info("Script principal terminé. La meilleure combinaison identifiée est:")
        # Afficher joliment le dictionnaire retourné
        for key, value in best_params_found.items():
             if isinstance(value, float):
                 logger.info(f"  {key}: {value:.4f}")
             else:
                 logger.info(f"  {key}: {value}")
    else:
        logger.info("Script principal terminé. Aucune combinaison optimale trouvée selon les critères définis.")