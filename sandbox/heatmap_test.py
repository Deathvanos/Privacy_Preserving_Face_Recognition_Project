# -*- coding: utf-8 -*-
import os
import sys
import logging
import warnings
import base64
import io
import binascii
import argparse # Pour passer des paramètres en ligne de commande (optionnel)

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm

# --- Configuration Initiale ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Ajustement PYTHONPATH pour les modules locaux ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
# --- Fin Ajustement PYTHONPATH ---

# --- Import des modules locaux ---
try:
    from src.modules import anony_process_pipeline
    from src.config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE
    _modules_imported = True
    logging.info(f"DEFAULT_IMAGE_SIZE importé depuis config.py: {DEFAULT_IMAGE_SIZE}")
    if not (isinstance(DEFAULT_IMAGE_SIZE, tuple) and len(DEFAULT_IMAGE_SIZE) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in DEFAULT_IMAGE_SIZE)):
        logging.error(f"DEFAULT_IMAGE_SIZE ({DEFAULT_IMAGE_SIZE}) invalide. Utilisation de (100, 100).")
        DEFAULT_IMAGE_SIZE = (100, 100)
except ImportError as e:
    logging.error(f"Échec import modules locaux/config.py: {e}. Vérifiez structure/PYTHONPATH.")
    logging.warning("Utilisation de DEFAULT_IMAGE_SIZE = (100, 100).")
    DEFAULT_IMAGE_SIZE = (100, 100)
    _modules_imported = False
except Exception as e:
    logging.error(f"Erreur importation: {e}")
    logging.warning("Utilisation de DEFAULT_IMAGE_SIZE = (100, 100).")
    DEFAULT_IMAGE_SIZE = (100, 100)
    _modules_imported = False

if not _modules_imported:
    logging.critical("Imports échoués. Arrêt.")
    sys.exit(1)
# --- Fin Imports ---

# --- Paramètres pour l'exécution du test ---
# (Peuvent être modifiés ou passés en arguments)
TARGET_SUBJECT_ID_HEATMAP = 1  # Index du sujet LFW à analyser
OPTIMAL_K = 8                 # Valeur K pour la pipeline
OPTIMAL_RATIO = 0.55           # Ratio PCA pour la pipeline
OPTIMAL_EPSILON = 0.1        # Epsilon DP pour la pipeline (valeur de l'article)

# Dataset Parameters (cohérent avec les autres scripts)
MIN_FACES_PER_PERSON_HEATMAP = 20 # Assure assez d'images pour PCA
N_SAMPLES_PER_PERSON_HEATMAP = 10 # Nombre d'images à charger pour le sujet

# --- Répertoire de sortie ---
HEATMAP_OUTPUT_DIR = "heatmap_analysis"
os.makedirs(HEATMAP_OUTPUT_DIR, exist_ok=True)
logging.info(f"Les heatmaps seront sauvegardées dans : {os.path.abspath(HEATMAP_OUTPUT_DIR)}")
# --- Fin Paramètres ---


# --- Fonctions Utilitaires (copiées/adaptées depuis anony_test_visu.py) ---
def load_lfw_subject_data(min_faces: int, n_samples: int, target_subject_id: int) -> tuple[pd.DataFrame | None, int | None, int | None, int | None]:
    """Charge LFW, équilibre, et retourne les données UNIQUEMENT pour le sujet cible."""
    logging.info(f"Chargement LFW (min_faces={min_faces}, n_samples={n_samples})...")
    try:
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4, color=False)
        height, width = lfw_people.images.shape[1], lfw_people.images.shape[2]
        logging.info(f"Shape LFW native (après resize=0.4): ({height}, {width})")
        data = lfw_people.data
        if data.max() <= 1.0 + 1e-6: data = (data * 255).clip(0, 255).astype(np.uint8)
        else: data = data.clip(0, 255).astype(np.uint8)
        df = pd.DataFrame(data); df['subject_id'] = lfw_people.target
        grouped = df.groupby('subject_id'); sampled_subject_ids = []; original_subject_ids_in_order = []
        all_subject_ids = sorted(list(df['subject_id'].unique())) # Tous les sujets LFW
        for subj_id in all_subject_ids: # Itère dans l'ordre LFW
             group = df[df['subject_id'] == subj_id]
             if len(group) >= n_samples:
                  sampled_subject_ids.append(subj_id)
                  original_subject_ids_in_order.append(subj_id)

        if not sampled_subject_ids: logging.error(f"Aucun sujet trouvé avec >= {n_samples} images."); return None, None, None, None
        if target_subject_id < 0 or target_subject_id >= len(original_subject_ids_in_order): logging.error(f"TARGET_SUBJECT_ID_HEATMAP {target_subject_id} hors limites (0 à {len(original_subject_ids_in_order)-1})."); return None, None, None, None

        lfw_original_target_id = original_subject_ids_in_order[target_subject_id]
        logging.info(f"ID Sujet Analyse {target_subject_id} correspond à ID Original LFW : {lfw_original_target_id}")

        df_subject = df[df['subject_id'] == lfw_original_target_id].copy()

        # Échantillonne N_SAMPLES si assez d'images, sinon prend tout
        if len(df_subject) >= n_samples:
            df_subject_sampled = df_subject.sample(n=n_samples, random_state=42, replace=False)
            logging.info(f"Échantillonné {n_samples} images pour sujet {lfw_original_target_id}.")
        else:
            logging.warning(f"Sujet {lfw_original_target_id} a seulement {len(df_subject)} images < {n_samples}. Utilisation de toutes.")
            df_subject_sampled = df_subject.copy()

        def row_to_pil_image(row_pixels): return Image.fromarray(row_pixels.values.reshape((height, width)), mode='L')
        logging.info("Conversion pixels -> PIL...")
        pixel_columns = list(range(height * width))
        df_subject_sampled['userFaces'] = df_subject_sampled[pixel_columns].apply(row_to_pil_image, axis=1)
        df_subject_sampled = df_subject_sampled.reset_index(drop=True); df_subject_sampled['imageId'] = df_subject_sampled.index.astype(str)
        df_final = df_subject_sampled[['userFaces', 'imageId', 'subject_id']].copy()
        df_final.rename(columns={'subject_id': 'subject_number'}, inplace=True)
        logging.info(f"Données chargées pour sujet {lfw_original_target_id}: {len(df_final)} images.")
        return df_final, height, width, lfw_original_target_id
    except Exception as e: logging.critical(f"Erreur chargement/préparation LFW : {e}", exc_info=True); return None, None, None, None

def decode_b64_to_pil(b64_string: str | None) -> Image.Image | None:
    """Décode une chaîne image base64 en objet PIL Image (niveaux de gris)."""
    if not isinstance(b64_string, str): return None
    try:
        img_bytes = base64.b64decode(b64_string)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('L')
        return img_pil
    except (binascii.Error, UnidentifiedImageError, IOError, ValueError):
        return None
    except Exception as e:
        logging.error(f"Erreur inattendue décodage b64 : {e}", exc_info=False)
        return None
# --- Fin Fonctions Utilitaires ---


# --- Fonction Principale ---
def main_heatmap():
    """Charge les données, exécute la pipeline, calcule et affiche la heatmap d'erreur."""
    logging.info("--- DÉMARRAGE SCRIPT HEATMAP ERREUR RECONSTRUCTION ---")
    logging.info(f"Sujet Cible: {TARGET_SUBJECT_ID_HEATMAP}, K={OPTIMAL_K}, Ratio={OPTIMAL_RATIO}, Epsilon={OPTIMAL_EPSILON}")
    logging.info(f"Taille Image Cible: {DEFAULT_IMAGE_SIZE}")

    # 1. Charger les données pour le sujet cible
    df_subject, h_lfw, w_lfw, lfw_id = load_lfw_subject_data(
        min_faces=MIN_FACES_PER_PERSON_HEATMAP,
        n_samples=N_SAMPLES_PER_PERSON_HEATMAP,
        target_subject_id=TARGET_SUBJECT_ID_HEATMAP
    )
    if df_subject is None:
        logging.critical("Échec chargement données sujet. Arrêt.")
        return

    # S'assure qu'on a au moins une image
    if df_subject.empty:
        logging.error("Aucune image chargée pour le sujet sélectionné.")
        return

    # Prend la première image comme image focus
    original_focus_pil_native = df_subject['userFaces'].iloc[0]
    original_focus_id = df_subject['imageId'].iloc[0]
    subject_id_str = str(df_subject['subject_number'].iloc[0])
    logging.info(f"Image focus sélectionnée: ID {original_focus_id} (Sujet LFW {lfw_id})")

    # 2. Prétraiter l'image focus pour obtenir la référence (taille DEFAULT_IMAGE_SIZE)
    original_focus_pil_proc = None
    original_focus_np = None
    try:
        logging.info(f"Prétraitement image focus (ID: {original_focus_id}) -> Taille {DEFAULT_IMAGE_SIZE}...")
        preprocessed_focus = anony_process_pipeline.preprocess_image(
            original_focus_pil_native,
            resize_size=DEFAULT_IMAGE_SIZE,
            create_flattened=False
        )
        if preprocessed_focus and 'grayscale_image' in preprocessed_focus:
            original_focus_pil_proc = preprocessed_focus['grayscale_image']
            original_focus_np = np.array(original_focus_pil_proc, dtype=np.uint8)
            logging.info(f"Image focus prétraitée. Shape NumPy: {original_focus_np.shape}")
        else:
            logging.error("Échec prétraitement image focus.")
            return # Arrêt si l'originale ne peut être préparée
    except Exception as e:
        logging.error(f"Erreur prétraitement image focus: {e}", exc_info=True)
        return

    # 3. Exécuter la pipeline complète sur toutes les images chargées du sujet
    logging.info(f"Exécution pipeline complète pour sujet {lfw_id} (k={OPTIMAL_K}, r={OPTIMAL_RATIO}, e={OPTIMAL_EPSILON})...")
    try:
        pipeline_output = anony_process_pipeline.run_pipeline(
            df_images=df_subject.copy(), # Utilise toutes les images pour PCA
            k_same_k_value=OPTIMAL_K,
            n_components_ratio=OPTIMAL_RATIO,
            epsilon=OPTIMAL_EPSILON,
            image_size_override=DEFAULT_IMAGE_SIZE
        )
    except Exception as e:
        logging.error(f"Erreur lors de l'exécution de la pipeline: {e}", exc_info=True)
        return # Arrêt si la pipeline échoue

    # 4. Récupérer l'image reconstruite pour l'image focus
    reconstructed_focus_pil = None
    reconstructed_focus_np = None
    if subject_id_str in pipeline_output:
        subject_results = pipeline_output[subject_id_str]
        try:
            focus_idx = subject_results['imageIds'].index(original_focus_id)
            b64_recon = subject_results['final_reconstructed_b64'][focus_idx]
            if b64_recon:
                reconstructed_focus_pil = decode_b64_to_pil(b64_recon)
                if reconstructed_focus_pil:
                    reconstructed_focus_np = np.array(reconstructed_focus_pil, dtype=np.uint8)
                    logging.info(f"Image focus reconstruite récupérée. Shape NumPy: {reconstructed_focus_np.shape}")
                else:
                    logging.error(f"Échec décodage Base64 pour image reconstruite focus (ID: {original_focus_id}).")
            else:
                logging.error(f"Image reconstruite pour focus (ID: {original_focus_id}) est None dans la sortie pipeline.")
        except ValueError:
            logging.error(f"Image focus ID {original_focus_id} non trouvée dans les résultats de la pipeline.")
        except IndexError:
            logging.error(f"Index hors limites lors de la recherche de l'image focus reconstruite.")
        except Exception as e:
             logging.error(f"Erreur récupération image reconstruite: {e}", exc_info=True)
    else:
        logging.error(f"Aucun résultat de pipeline trouvé pour le sujet {subject_id_str}.")

    # 5. Calculer et afficher la heatmap si tout est OK
    if original_focus_np is not None and reconstructed_focus_np is not None:
        if original_focus_np.shape != reconstructed_focus_np.shape:
            logging.error(f"Discordance de shape entre originale prétraitée ({original_focus_np.shape}) et reconstruite ({reconstructed_focus_np.shape}). Impossible de calculer l'erreur.")
        else:
            logging.info("Calcul de la carte d'erreur absolue...")
            # Calculer l'erreur absolue en float pour éviter problèmes uint8
            error_map = np.abs(original_focus_np.astype(float) - reconstructed_focus_np.astype(float))
            avg_error = np.mean(error_map)
            max_error = np.max(error_map)
            logging.info(f"Carte d'erreur calculée. Erreur Moyenne={avg_error:.2f}, Erreur Max={max_error:.2f}")

            # Affichage
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            fig.suptitle(f"Analyse Erreur Reconstruction - Sujet LFW {lfw_id} (ID: {original_focus_id})\nParams: k={OPTIMAL_K}, ratio={OPTIMAL_RATIO}, eps={OPTIMAL_EPSILON}", fontsize=12)

            # Image Originale Prétraitée
            ax = axes[0]
            im = ax.imshow(original_focus_np, cmap='gray')
            ax.set_title(f"Originale Prétraitée\n({original_focus_np.shape[1]}x{original_focus_np.shape[0]})")
            ax.axis('off')

            # Image Reconstruite
            ax = axes[1]
            im = ax.imshow(reconstructed_focus_np, cmap='gray')
            ax.set_title(f"Reconstruite\n({reconstructed_focus_np.shape[1]}x{reconstructed_focus_np.shape[0]})")
            ax.axis('off')

            # Carte d'Erreur
            ax = axes[2]
            im = ax.imshow(error_map, cmap='viridis') # 'hot', 'viridis', 'plasma', 'magma'
            ax.set_title(f"Erreur Absolue (Moy={avg_error:.1f}, Max={max_error:.0f})")
            ax.axis('off')
            # Ajout de la colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Différence Absolue Pixel')

            plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Ajuste pour le titre global

            # Sauvegarde
            heatmap_filename = f"heatmap_subj_{lfw_id}_img_{original_focus_id}_k{OPTIMAL_K}_r{OPTIMAL_RATIO:.2f}_e{OPTIMAL_EPSILON:.2f}.png"
            heatmap_path = os.path.join(HEATMAP_OUTPUT_DIR, heatmap_filename)
            try:
                plt.savefig(heatmap_path, bbox_inches='tight')
                logging.info(f"Heatmap sauvegardée: {heatmap_path}")
            except Exception as e:
                logging.error(f"Échec sauvegarde heatmap {heatmap_path}: {e}")
            plt.close(fig)

    else:
        logging.error("Impossible de générer la heatmap car l'image originale ou reconstruite est manquante/invalide.")

    logging.info("--- FIN SCRIPT HEATMAP ---")

if __name__ == "__main__":
    # Optionnel: Ajouter argparse pour passer les paramètres en ligne de commande
    # parser = argparse.ArgumentParser(description="Génère une heatmap d'erreur de reconstruction pour un sujet LFW.")
    # parser.add_argument("--subject_id", type=int, default=TARGET_SUBJECT_ID_HEATMAP, help="Index du sujet LFW à analyser.")
    # parser.add_argument("--k", type=int, default=OPTIMAL_K, help="Valeur K pour la pipeline.")
    # parser.add_argument("--ratio", type=float, default=OPTIMAL_RATIO, help="Ratio PCA pour la pipeline.")
    # parser.add_argument("--epsilon", type=float, default=OPTIMAL_EPSILON, help="Epsilon DP pour la pipeline.")
    # args = parser.parse_args()
    # TARGET_SUBJECT_ID_HEATMAP = args.subject_id
    # OPTIMAL_K = args.k
    # OPTIMAL_RATIO = args.ratio
    # OPTIMAL_EPSILON = args.epsilon

    main_heatmap()
