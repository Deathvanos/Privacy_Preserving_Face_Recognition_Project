import os
import sys
import logging
import warnings
import base64
import io
import binascii

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from tqdm import tqdm # Pour la barre de progression

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Adjust PYTHONPATH for local modules (modify as needed) ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
# --- End PYTHONPATH Adjustment ---

# --- Import local modules ---
try:
    from src.modules import anony_process_pipeline
    from src.modules import k_same_pixel as ksp # Import direct pour K-Same
    from src.modules import utils_image # Pour les conversions éventuelles
    # Importe DEFAULT_IMAGE_SIZE depuis config.py
    from src.config import IMAGE_SIZE as DEFAULT_IMAGE_SIZE
    _modules_imported = True
    # Log la taille importée
    logging.info(f"DEFAULT_IMAGE_SIZE importé depuis config.py: {DEFAULT_IMAGE_SIZE}")
    # Vérifie si la taille est valide
    if not (isinstance(DEFAULT_IMAGE_SIZE, tuple) and len(DEFAULT_IMAGE_SIZE) == 2 and
            all(isinstance(dim, int) and dim > 0 for dim in DEFAULT_IMAGE_SIZE)):
        logging.error(f"DEFAULT_IMAGE_SIZE ({DEFAULT_IMAGE_SIZE}) n'est pas un tuple de deux entiers positifs. Utilisation de (100, 100) par défaut.")
        DEFAULT_IMAGE_SIZE = (100, 100) # Fallback sûr

except ImportError as e:
    logging.error(f"Échec de l'importation des modules locaux ou de config.py : {e}")
    logging.error(f"Vérifiez la structure de votre projet et sys.path : {sys.path}")
    logging.warning("Utilisation de DEFAULT_IMAGE_SIZE = (100, 100) par défaut.")
    DEFAULT_IMAGE_SIZE = (100, 100) # Fallback sûr
    # Tente d'importer les modules sans config pour permettre une exécution partielle
    try:
        from src.modules import anony_process_pipeline
        from src.modules import k_same_pixel as ksp
        from src.modules import utils_image
        _modules_imported = True # Au moins les modules sont là
    except ImportError:
        _modules_imported = False # Échec critique
except Exception as e:
    logging.error(f"Erreur inattendue lors de l'importation : {e}")
    logging.warning("Utilisation de DEFAULT_IMAGE_SIZE = (100, 100) par défaut.")
    DEFAULT_IMAGE_SIZE = (100, 100) # Fallback sûr
    _modules_imported = False


if not _modules_imported:
      logging.critical("Sortie en raison d'erreurs d'importation critiques.")
      sys.exit(1)
# --- End Imports ---

# --- Parameters ---
TARGET_SUBJECT_ID = 1 # Index du sujet à visualiser (0, 1, 2, ...)
N_IMAGES_TO_SHOW_PER_SUBJECT = 1 # Combien d'images de ce sujet utiliser pour la visualisation

# Parameters for Visualization Lines
K_VALUES_TO_VISUALIZE = [2, 3, 5, 10] # Valeurs de K à tester
RATIOS_TO_VISUALIZE = [0.1, 0.4, 0.7, 0.9] # Ratios PCA à tester
NO_NOISE_EPSILON = float('inf') # Représente l'absence de bruit DP
EPSILONS_TO_VISUALIZE = [0.1, 0.5, 1.0, 5.0, NO_NOISE_EPSILON] # Epsilons à tester
NO_NOISE_EPSILON_LABEL = "Inf (Pas de DP)" # Étiquette pour epsilon infini

# Fixed parameters when varying others
FIXED_K_FOR_PEEP = 3 # Valeur de K utilisée lors de la variation du ratio et d'epsilon
FIXED_RATIO_FOR_EPSILON = 0.7 # Ratio utilisé lors de la variation d'epsilon

# Dataset Parameters
MIN_FACES_PER_PERSON = 20 # Minimum de visages requis par personne dans LFW
# Assure N_SAMPLES >= max(K) pour que K-Same ait assez d'images
N_SAMPLES_PER_PERSON = max(10, max(K_VALUES_TO_VISUALIZE) if K_VALUES_TO_VISUALIZE else 10)

# --- Output Directory for Plots ---
PLOT_OUTPUT_DIR = f"visualizations_subject_{TARGET_SUBJECT_ID}_pipeline"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
logging.info(f"Les graphiques seront sauvegardés dans : {os.path.abspath(PLOT_OUTPUT_DIR)}")
# --- End Output Directory ---

# --- Utility Functions ---
def load_lfw_subject_data(min_faces: int, n_samples: int, target_subject_id: int) -> tuple[pd.DataFrame | None, int | None, int | None, int | None]:
    """Charge LFW, équilibre, et retourne les données UNIQUEMENT pour le sujet cible."""
    logging.info(f"Chargement du dataset LFW (min_faces={min_faces}, n_samples={n_samples})...")
    try:
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4, color=False)
        height, width = lfw_people.images.shape[1], lfw_people.images.shape[2]
        logging.info(f"Images LFW chargées avec shape native (après resize=0.4): ({height}, {width})")
        data = lfw_people.data
        if data.max() <= 1.0 + 1e-6: data = (data * 255).clip(0, 255).astype(np.uint8)
        else: data = data.clip(0, 255).astype(np.uint8)
        df = pd.DataFrame(data); df['subject_id'] = lfw_people.target
        grouped = df.groupby('subject_id'); sampled_subject_ids = []; original_subject_ids_in_order = []
        for subj_id, group in grouped:
            if len(group) >= n_samples: sampled_subject_ids.append(subj_id); original_subject_ids_in_order.append(subj_id)
        if not sampled_subject_ids: logging.error(f"Aucun sujet trouvé avec >= {n_samples} images."); return None, None, None, None
        if TARGET_SUBJECT_ID < 0 or TARGET_SUBJECT_ID >= len(original_subject_ids_in_order): logging.error(f"TARGET_SUBJECT_ID {TARGET_SUBJECT_ID} hors limites (0 à {len(original_subject_ids_in_order)-1})."); return None, None, None, None
        lfw_original_target_id = original_subject_ids_in_order[TARGET_SUBJECT_ID]
        logging.info(f"ID Sujet Analyse {TARGET_SUBJECT_ID} correspond à ID Original LFW : {lfw_original_target_id}")
        df_subject = df[df['subject_id'] == lfw_original_target_id].copy()
        if len(df_subject) >= n_samples: df_subject_sampled = df_subject.sample(n=n_samples, random_state=42, replace=False)
        else: logging.warning(f"Sujet {lfw_original_target_id} a seulement {len(df_subject)} images < {n_samples}. Utilisation de toutes."); df_subject_sampled = df_subject.copy()
        def row_to_pil_image(row_pixels): return Image.fromarray(row_pixels.values.reshape((height, width)), mode='L')
        logging.info("Conversion des lignes de pixels en objets Image PIL pour le sujet cible...")
        pixel_columns = list(range(height * width))
        df_subject_sampled['userFaces'] = df_subject_sampled[pixel_columns].apply(row_to_pil_image, axis=1)
        df_subject_sampled = df_subject_sampled.reset_index(drop=True); df_subject_sampled['imageId'] = df_subject_sampled.index.astype(str) # ID unique string
        df_final = df_subject_sampled[['userFaces', 'imageId', 'subject_id']].copy()
        df_final.rename(columns={'subject_id': 'subject_number'}, inplace=True)
        logging.info(f"Données chargées pour sujet {lfw_original_target_id}: {len(df_final)} images.")
        return df_final, height, width, lfw_original_target_id
    except Exception as e: logging.critical(f"Erreur chargement/préparation LFW : {e}", exc_info=True); return None, None, None, None

def plot_image_variants(original_img_pil: Image.Image, variant_images_pil: list[Image.Image | None], variant_labels: list[str], figure_title: str, save_path: str):
    """Affiche l'image originale (fournie) et plusieurs variantes côte à côte et SAUVEGARDE."""
    if original_img_pil is None:
        logging.error("Image originale fournie à plot_image_variants est None.")
        return
    try:
        original_img_np = np.array(original_img_pil)
        h, w = original_img_np.shape[:2]
        logging.debug(f"Affichage: Taille image originale utilisée pour le tracé: ({h},{w})")
    except Exception as e:
        logging.error(f"Impossible de convertir l'image originale fournie en NumPy : {e}")
        return

    n_variants = len(variant_images_pil)
    n_cols = n_variants + 1
    fig = plt.figure(figsize=(2.0 * n_cols, 2.8))
    plt.suptitle(figure_title, size=14, y=0.98)

    ax = plt.subplot(1, n_cols, 1)
    ax.imshow(original_img_np, cmap=plt.cm.gray)
    ax.set_title("Originale\n(Prétraitée)", size=10) # Modifié pour indiquer prétraitée
    ax.set_xticks(()); ax.set_yticks(())

    for i, img_pil in enumerate(variant_images_pil):
        ax = plt.subplot(1, n_cols, i + 2)
        if img_pil is not None:
            try:
                img_np = np.array(img_pil)
                if img_np.shape[0] != h or img_np.shape[1] != w:
                     logging.warning(f"Discordance de taille pour variante '{variant_labels[i]}': Originale ({h},{w}), Variante {img_np.shape}. Tentative d'affichage.")
                ax.imshow(img_np, cmap=plt.cm.gray)
            except Exception as e:
                logging.error(f"Erreur conversion/affichage variante '{variant_labels[i]}': {e}")
                ax.text(0.5, 0.5, 'Erreur', ha='center', va='center')
                ax.imshow(np.zeros((h, w)), cmap=plt.cm.gray)
        else:
            ax.text(0.5, 0.5, 'Échec', ha='center', va='center')
            ax.imshow(np.zeros((h, w)), cmap=plt.cm.gray)

        ax.set_title(variant_labels[i], size=10)
        ax.set_xticks(()); ax.set_yticks(())

    plt.tight_layout(rect=[0, 0.01, 1, 0.90])
    try:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        logging.info(f"Graphique sauvegardé : {save_path}")
    except Exception as e:
        logging.error(f"Échec sauvegarde graphique {save_path}: {e}")
    plt.close(fig)

def decode_b64_to_pil(b64_string: str | None) -> Image.Image | None:
    """Décode une chaîne image base64 en objet PIL Image (niveaux de gris)."""
    if not isinstance(b64_string, str): return None
    try:
        img_bytes = base64.b64decode(b64_string)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('L')
        return img_pil
    except (binascii.Error, UnidentifiedImageError, IOError, ValueError) as e:
        return None
    except Exception as e:
        logging.error(f"Erreur inattendue décodage b64 : {e}", exc_info=False)
        return None

def numpy_to_pil(np_array: np.ndarray) -> Image.Image | None:
    """Convertit un tableau NumPy (supposé 2D grayscale) en objet PIL Image."""
    if not isinstance(np_array, np.ndarray): return None
    try:
        if np_array.dtype != np.uint8:
            np_array = np.clip(np_array, 0, 255).astype(np.uint8)
        return Image.fromarray(np_array, mode='L')
    except Exception as e:
        logging.error(f"Erreur conversion NumPy vers PIL : {e}")
        return None
# --- End Utility Functions ---


# --- Visualization Functions (Using Real Pipeline) ---

# Modifié: Accepte l'image originale prétraitée en argument
def visualize_k_impact_line(subject_id_str: str, lfw_id: int, k_values: list[int],
                             original_subject_images_pil: list[Image.Image],
                             original_focus_pil_proc: Image.Image, # Ajouté
                             output_dir: str):
    """Visualise l'impact de K (K-Same) en appelant directement k_same_pixel_individual."""
    logging.info(f"\n--- 2. Génération: Impact de K (K-Same Pixel) pour Sujet {lfw_id} ---")
    if not original_subject_images_pil:
        logging.error("Aucune image originale fournie pour la visualisation de K.")
        return
    if original_focus_pil_proc is None:
         logging.error("L'image originale focus prétraitée n'a pas été fournie pour K-Same.")
         return

    max_k = max(k_values) if k_values else 0
    if len(original_subject_images_pil) < max_k:
         logging.warning(f"Pas assez d'images ({len(original_subject_images_pil)}) pour le k maximum ({max_k}).")

    # img_original_focus_pil = original_subject_images_pil[0] # Plus nécessaire ici
    img_original_focus_id = f"subj_{lfw_id}_img_0" # Sera mis à jour

    variants_k_pil = []
    titles_k = []

    preprocessed_subject_data = []
    # original_focus_pil_proc = None # Maintenant passé en argument

    target_size = DEFAULT_IMAGE_SIZE
    logging.debug(f"Utilisation de target_size = {target_size} pour le prétraitement K-Same.")

    for i, img_pil in enumerate(original_subject_images_pil):
        img_id = f"subj_{lfw_id}_img_{i}"
        try:
            logging.debug(f"Appel de preprocess_image pour K-Same (image {i}) avec target_size={target_size}")
            preprocessed = anony_process_pipeline.preprocess_image(img_pil, resize_size=target_size, create_flattened=False)
            if preprocessed and 'grayscale_image' in preprocessed:
                grayscale_pil = preprocessed['grayscale_image']
                grayscale_np = np.array(grayscale_pil, dtype=np.uint8)
                logging.debug(f"Image {i} prétraitée pour K-Same. Shape NumPy: {grayscale_np.shape}")
                preprocessed_subject_data.append({'grayscale_np': grayscale_np, 'imageId': img_id})
                if i == 0:
                    img_original_focus_id = img_id
                    # original_focus_pil_proc = grayscale_pil # Plus nécessaire ici
            else:
                 logging.warning(f"Échec prétraitement image {i} sujet {lfw_id} pour K-Same.")
        except Exception as e:
             logging.error(f"Erreur prétraitement image {i} sujet {lfw_id} pour K-Same: {e}")

    if not preprocessed_subject_data:
        logging.error("Aucune image n'a pu être prétraitée pour K-Same. Visualisation K impossible.")
        return

    h_proc, w_proc = preprocessed_subject_data[0]['grayscale_np'].shape
    logging.info(f"Dimensions après prétraitement pour K-Same: ({h_proc}, {w_proc})")

    k_same_input_list = [(item['grayscale_np'], item['imageId']) for item in preprocessed_subject_data]

    logging.info(f"  Application de K-Same Pixel pour différentes valeurs de k...")
    for k_val in tqdm(k_values, desc="K-Values"):
        anonymized_img_pil = None
        num_valid_images = len(k_same_input_list)
        if num_valid_images < k_val:
            logging.warning(f"k={k_val} > nombre d'images valides ({num_valid_images}). Skip K-Same.")
            titles_k.append(f"k={k_val}\n(Skip)")
        else:
            try:
                k_same_output_tuples = ksp.k_same_pixel_individual(k_same_input_list, k=k_val)
                focus_img_result = None
                for anon_array, img_id in k_same_output_tuples:
                    if img_id == img_original_focus_id:
                        focus_img_result = anon_array
                        break
                if focus_img_result is not None:
                    logging.debug(f"Shape sortie K-Same (k={k_val}) pour image focus: {focus_img_result.shape}")
                    if focus_img_result.shape != (h_proc, w_proc):
                         logging.warning(f"Shape K-Same ({focus_img_result.shape}) != shape attendue ({h_proc},{w_proc}) pour k={k_val}")
                    anonymized_img_pil = numpy_to_pil(focus_img_result)
                    if anonymized_img_pil is None: logging.error(f"Échec conversion NumPy->PIL pour k={k_val}")
                    titles_k.append(f"k={k_val}")
                else:
                    logging.warning(f"Résultat K-Same non trouvé pour image focus (ID: {img_original_focus_id}) avec k={k_val}.")
                    titles_k.append(f"k={k_val}\n(Non trouvé)")
            except Exception as k_err:
                logging.error(f"Erreur durant k_same_pixel_individual pour k={k_val}: {k_err}", exc_info=False)
                titles_k.append(f"k={k_val}\n(Erreur)")
        variants_k_pil.append(anonymized_img_pil)

    if variants_k_pil:
        save_path = os.path.join(output_dir, f"subject_{lfw_id}_2_k_impact.pdf")
        # Utilise l'originale prétraitée passée en argument
        plot_image_variants(original_focus_pil_proc, variants_k_pil, titles_k,
                            f"Sujet {lfw_id}: Impact de K (K-Same)",
                            save_path=save_path)
    else:
        logging.error("Visualisation K échouée, aucune variante générée.")

# Modifié: Accepte l'image originale prétraitée en argument
def visualize_ratio_impact_line(subject_id_str: str, lfw_id: int, ratios: list[float], fixed_k: int,
                                 original_subject_images_pil: list[Image.Image],
                                 original_focus_pil_proc: Image.Image, # Ajouté
                                 output_dir: str):
    """Visualise l'impact du ratio PCA (sans bruit DP) en utilisant la pipeline."""
    logging.info(f"\n--- 3. Génération: Impact du Ratio PCA (k={fixed_k}, Pas de Bruit DP) pour Sujet {lfw_id} ---")
    if not original_subject_images_pil: logging.error("Aucune image originale fournie."); return
    if original_focus_pil_proc is None: logging.error("Image originale focus prétraitée non fournie."); return

    # img_original_focus_pil = original_subject_images_pil[0] # Plus nécessaire ici
    img_original_focus_id = f"subj_{lfw_id}_img_0" # Sera mis à jour

    variants_ratio_pil = []
    titles_ratio = []

    df_subject_visu_list = []
    for i, img_pil in enumerate(original_subject_images_pil):
         current_img_id = f"subj_{lfw_id}_img_{i}"
         df_subject_visu_list.append({
             'userFaces': img_pil, 'imageId': current_img_id, 'subject_number': subject_id_str
         })
         if i == 0: img_original_focus_id = current_img_id
    df_subject_visu = pd.DataFrame(df_subject_visu_list)

    target_size = DEFAULT_IMAGE_SIZE
    logging.debug(f"Utilisation de target_size = {target_size} pour la pipeline Ratio.")

    logging.info(f"  Reconstruction avec k={fixed_k}, Epsilon=Infini (Pas de DP) pour différents ratios...")
    for ratio in tqdm(ratios, desc="Ratios"):
        reconstructed_img_pil = None
        try:
            logging.debug(f"Appel de run_pipeline pour Ratio={ratio:.2f} avec image_size_override={target_size}")
            pipeline_output = anony_process_pipeline.run_pipeline(
                df_images=df_subject_visu.copy(), k_same_k_value=fixed_k,
                n_components_ratio=ratio, epsilon=NO_NOISE_EPSILON,
                image_size_override=target_size
            )
            if subject_id_str in pipeline_output:
                subject_results = pipeline_output[subject_id_str]
                try:
                    focus_idx = subject_results['imageIds'].index(img_original_focus_id)
                    b64_recon = subject_results['final_reconstructed_b64'][focus_idx]
                    reconstructed_img_pil = decode_b64_to_pil(b64_recon)
                    if reconstructed_img_pil: logging.debug(f"Image reconstruite (Ratio={ratio:.2f}). Taille PIL: {reconstructed_img_pil.size}")
                    elif b64_recon is not None: logging.warning(f"Échec décodage b64 pour ratio={ratio:.2f}")
                    titles_ratio.append(f"Ratio={ratio:.2f}")
                except ValueError:
                     logging.warning(f"Image focus ID {img_original_focus_id} non trouvée pour ratio={ratio:.2f}.")
                     titles_ratio.append(f"Ratio={ratio:.2f}\n(Non trouvé)")
                except IndexError:
                     logging.error(f"Index hors limites pour ratio={ratio:.2f}.")
                     titles_ratio.append(f"Ratio={ratio:.2f}\n(Erreur Index)")
            else:
                 logging.warning(f"Aucun résultat de pipeline pour sujet {subject_id_str} avec ratio={ratio:.2f}.")
                 titles_ratio.append(f"Ratio={ratio:.2f}\n(Échec Pipe)")
        except Exception as pipe_err:
            logging.error(f"Erreur pipeline pour ratio={ratio:.2f}: {pipe_err}", exc_info=True)
            titles_ratio.append(f"Ratio={ratio:.2f}\n(Erreur Pipe)")
        variants_ratio_pil.append(reconstructed_img_pil)

    if variants_ratio_pil:
        save_path = os.path.join(output_dir, f"subject_{lfw_id}_3_ratio_impact_k{fixed_k}_no_noise.pdf")
        # Utilise l'originale prétraitée passée en argument
        plot_image_variants(original_focus_pil_proc, variants_ratio_pil, titles_ratio,
                            f"Sujet {lfw_id}: Impact Ratio PCA (k={fixed_k}, Pas de Bruit)",
                            save_path=save_path)
    else:
        logging.error("Visualisation Ratio échouée, aucune variante générée.")

# Modifié: Accepte l'image originale prétraitée en argument
def visualize_epsilon_impact_line(subject_id_str: str, lfw_id: int, fixed_ratio: float, epsilons: list[float], fixed_k: int,
                                   original_subject_images_pil: list[Image.Image],
                                   original_focus_pil_proc: Image.Image, # Ajouté
                                   output_dir: str):
    """Visualise l'impact d'epsilon (avec ratio fixe) en utilisant la pipeline."""
    logging.info(f"\n--- 4. Génération: Impact d'Epsilon (k={fixed_k}, Ratio={fixed_ratio}) pour Sujet {lfw_id} ---")
    if not original_subject_images_pil: logging.error("Aucune image originale fournie."); return
    if original_focus_pil_proc is None: logging.error("Image originale focus prétraitée non fournie."); return

    # img_original_focus_pil = original_subject_images_pil[0] # Plus nécessaire ici
    img_original_focus_id = f"subj_{lfw_id}_img_0" # Sera mis à jour

    variants_eps_pil = []
    titles_eps = []

    df_subject_visu_list = []
    for i, img_pil in enumerate(original_subject_images_pil):
         current_img_id = f"subj_{lfw_id}_img_{i}"
         df_subject_visu_list.append({
             'userFaces': img_pil, 'imageId': current_img_id, 'subject_number': subject_id_str
         })
         if i == 0: img_original_focus_id = current_img_id
    df_subject_visu = pd.DataFrame(df_subject_visu_list)

    target_size = DEFAULT_IMAGE_SIZE
    logging.debug(f"Utilisation de target_size = {target_size} pour la pipeline Epsilon.")

    logging.info(f"  Reconstruction avec k={fixed_k}, Ratio={fixed_ratio} pour différents epsilons...")
    for eps in tqdm(epsilons, desc="Epsilons"):
        reconstructed_img_pil = None
        try:
            logging.debug(f"Appel de run_pipeline pour Epsilon={eps:.2f} avec image_size_override={target_size}")
            pipeline_output = anony_process_pipeline.run_pipeline(
                df_images=df_subject_visu.copy(), k_same_k_value=fixed_k,
                n_components_ratio=fixed_ratio, epsilon=eps,
                image_size_override=target_size
            )
            if subject_id_str in pipeline_output:
                subject_results = pipeline_output[subject_id_str]
                try:
                    focus_idx = subject_results['imageIds'].index(img_original_focus_id)
                    b64_recon = subject_results['final_reconstructed_b64'][focus_idx]
                    reconstructed_img_pil = decode_b64_to_pil(b64_recon)
                    if reconstructed_img_pil: logging.debug(f"Image reconstruite (Epsilon={eps:.2f}). Taille PIL: {reconstructed_img_pil.size}")
                    elif b64_recon is not None: logging.warning(f"Échec décodage b64 pour epsilon={eps:.2f}")
                    eps_label = f"Eps={eps:.1f}" if eps != NO_NOISE_EPSILON else NO_NOISE_EPSILON_LABEL
                    titles_eps.append(eps_label)
                except ValueError:
                     logging.warning(f"Image focus ID {img_original_focus_id} non trouvée pour epsilon={eps:.2f}.")
                     titles_eps.append(f"Eps={eps:.1f}\n(Non trouvé)")
                except IndexError:
                     logging.error(f"Index hors limites pour epsilon={eps:.2f}.")
                     titles_eps.append(f"Eps={eps:.1f}\n(Erreur Index)")
            else:
                 logging.warning(f"Aucun résultat de pipeline pour sujet {subject_id_str} avec epsilon={eps:.2f}.")
                 titles_eps.append(f"Eps={eps:.1f}\n(Échec Pipe)")
        except Exception as pipe_err:
            logging.error(f"Erreur pipeline pour epsilon={eps:.2f}: {pipe_err}", exc_info=True)
            titles_eps.append(f"Eps={eps:.1f}\n(Erreur Pipe)")
        variants_eps_pil.append(reconstructed_img_pil)

    if variants_eps_pil:
        save_path = os.path.join(output_dir, f"subject_{lfw_id}_4_epsilon_impact_k{fixed_k}_r{fixed_ratio:.2f}.pdf")
        # Utilise l'originale prétraitée passée en argument
        plot_image_variants(original_focus_pil_proc, variants_eps_pil, titles_eps,
                            f"Sujet {lfw_id}: Impact Epsilon (k={fixed_k}, Ratio={fixed_ratio:.2f})",
                            save_path=save_path)
    else:
        logging.error("Visualisation Epsilon échouée, aucune variante générée.")

# --- Main Execution ---
def main():
    """Orchestre le chargement et la visualisation, en sauvegardant les graphiques."""
    if not _modules_imported:
        print("Impossible d'exécuter la visualisation en raison d'erreurs d'importation.")
        return

    logging.info(f"--- DÉMARRAGE SCRIPT VISUALISATION PIPELINE (Sujet ID: {TARGET_SUBJECT_ID}) ---")
    logging.info(f"TAILLE D'IMAGE CIBLE UTILISÉE (DEFAULT_IMAGE_SIZE): {DEFAULT_IMAGE_SIZE}")
    logging.info(f"Paramètres fixes : K pour PEEP={FIXED_K_FOR_PEEP}, Ratio pour Epsilon={FIXED_RATIO_FOR_EPSILON}")
    logging.info(f"Visualisation K : {K_VALUES_TO_VISUALIZE}")
    logging.info(f"Visualisation Ratio : {RATIOS_TO_VISUALIZE}")
    logging.info(f"Visualisation Epsilon : {EPSILONS_TO_VISUALIZE}")

    # 1. Charger les données LFW (taille native après resize=0.4)
    df_subject, h_lfw, w_lfw, lfw_id = load_lfw_subject_data(MIN_FACES_PER_PERSON, N_SAMPLES_PER_PERSON, TARGET_SUBJECT_ID)
    if df_subject is None or h_lfw is None or w_lfw is None:
        logging.critical("Échec du chargement des données sujet. Sortie.")
        return

    original_images_pil = df_subject['userFaces'].tolist()
    subject_id_str = str(df_subject['subject_number'].iloc[0])

    if not original_images_pil:
        logging.error("Aucune image chargée pour le sujet. Sortie.")
        return
    if len(original_images_pil) < N_IMAGES_TO_SHOW_PER_SUBJECT:
        logging.warning(f"Moins d'images ({len(original_images_pil)}) que demandé ({N_IMAGES_TO_SHOW_PER_SUBJECT}). Utilisation de toutes les images disponibles.")

    images_to_visualize_pil = original_images_pil[:N_IMAGES_TO_SHOW_PER_SUBJECT]
    if not images_to_visualize_pil:
        logging.error("Aucune image sélectionnée pour la visualisation.")
        return

    # 2. Prétraiter l'image focus UNE SEULE FOIS pour l'affichage
    original_focus_pil_native = images_to_visualize_pil[0]
    original_focus_pil_proc = None
    try:
        logging.debug(f"Prétraitement de l'image focus pour affichage avec target_size={DEFAULT_IMAGE_SIZE}")
        preprocessed_focus = anony_process_pipeline.preprocess_image(
            original_focus_pil_native,
            resize_size=DEFAULT_IMAGE_SIZE,
            create_flattened=False
        )
        if preprocessed_focus and 'grayscale_image' in preprocessed_focus:
            original_focus_pil_proc = preprocessed_focus['grayscale_image']
            logging.info(f"Image focus prétraitée pour affichage. Taille: {original_focus_pil_proc.size}")
        else:
            logging.error("Échec du prétraitement de l'image focus pour l'affichage.")
            # Optionnel: utiliser l'image native si le prétraitement échoue ?
            # original_focus_pil_proc = original_focus_pil_native
            # logging.warning("Utilisation de l'image focus native pour l'affichage car prétraitement échoué.")
    except Exception as e:
        logging.error(f"Erreur lors du prétraitement de l'image focus pour affichage: {e}", exc_info=True)
        # Optionnel: fallback
        # original_focus_pil_proc = original_focus_pil_native
        # logging.warning("Utilisation de l'image focus native pour l'affichage suite à erreur.")

    if original_focus_pil_proc is None:
         logging.critical("Impossible de préparer l'image originale prétraitée pour l'affichage. Arrêt.")
         return


    # --- Exécuter les Visualisations ---

    # Viz 1: Afficher l'image originale (taille LFW native) - Optionnel, gardé pour info
    logging.info(f"\n--- 1. Affichage Image Originale Sujet {lfw_id} (Taille LFW: {w_lfw}x{h_lfw}) ---")
    save_path_orig_lfw = os.path.join(PLOT_OUTPUT_DIR, f"subject_{lfw_id}_1_original_lfw_size.pdf")
    plot_image_variants(original_focus_pil_native, [], [], f"Sujet {lfw_id}: Image Originale (Taille LFW)", save_path_orig_lfw)

    # Viz 1b: Afficher l'image originale prétraitée seule
    logging.info(f"\n--- 1b. Affichage Image Originale Sujet {lfw_id} (Prétraitée: {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]}) ---")
    save_path_orig_proc = os.path.join(PLOT_OUTPUT_DIR, f"subject_{lfw_id}_1b_original_preprocessed.pdf")
    plot_image_variants(original_focus_pil_proc, [], [], f"Sujet {lfw_id}: Image Originale (Prétraitée)", save_path_orig_proc)


    # Viz 2: Impact de K (affiche l'originale prétraitée)
    visualize_k_impact_line(subject_id_str, lfw_id, K_VALUES_TO_VISUALIZE,
                             original_images_pil, original_focus_pil_proc, # Passe l'originale prétraitée
                             PLOT_OUTPUT_DIR)

    # Viz 3: Impact du ratio (affiche l'originale prétraitée)
    visualize_ratio_impact_line(subject_id_str, lfw_id, RATIOS_TO_VISUALIZE, FIXED_K_FOR_PEEP,
                                 original_images_pil, original_focus_pil_proc, # Passe l'originale prétraitée
                                 PLOT_OUTPUT_DIR)

    # Viz 4: Impact d'epsilon (affiche l'originale prétraitée)
    visualize_epsilon_impact_line(subject_id_str, lfw_id, FIXED_RATIO_FOR_EPSILON, EPSILONS_TO_VISUALIZE, FIXED_K_FOR_PEEP,
                                   original_images_pil, original_focus_pil_proc, # Passe l'originale prétraitée
                                   PLOT_OUTPUT_DIR)


    logging.info(f"--- Script de Visualisation Terminé. Graphiques sauvegardés dans '{PLOT_OUTPUT_DIR}' ---")
    logging.info(f"Vérifiez les logs (niveau DEBUG activé) pour les détails sur les tailles d'image traitées.")

if __name__ == "__main__":
    main()
