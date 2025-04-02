import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Adjust PYTHONPATH for local modules (modify as needed) ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
# --- End PYTHONPATH Adjustment ---

# --- Import local modules ---
try:
    from src.modules import anony_process_pipeline
    from src.modules import utils_image
    _modules_imported = True
except ImportError as e:
    print(f"[ERROR] Failed to import local modules: {e}")
    print(f"Check your project structure and sys.path: {sys.path}")
    _modules_imported = False

if not _modules_imported:
      print("Exiting due to import errors.")
      sys.exit(1)
# --- End Imports ---

# --- Parameters ---
SELECTED_OPTIMAL_RATIO = 0.70
COMPROMISE_EPSILON = 0.500
TARGET_SUBJECT_ID = 1

# Parameters for NEW line visualizations
# Using 4 ratios for clarity
RATIOS_TO_VISUALIZE = [0.1, 0.4, SELECTED_OPTIMAL_RATIO, 0.9]
# Using 5 epsilon values, including compromise and 'no noise'
NO_NOISE_EPSILON = 1e6 # Represents reconstruction without DP noise
EPSILONS_TO_VISUALIZE = [0.2, COMPROMISE_EPSILON, 2.0, 5.0, NO_NOISE_EPSILON]
NO_NOISE_EPSILON_LABEL = "Inf (No DP)"

# Parameters for the initial optimal pipeline viz (optional)
N_IMAGES_TO_SHOW_ORIGINAL = 5
N_EIGENFACES_TO_SHOW = 6

# Dataset Parameters
MIN_FACES_PER_PERSON = 20
N_SAMPLES_PER_PERSON = 10

# --- Output Directory for Plots ---
PLOT_OUTPUT_DIR = f"visualizations_subject_{TARGET_SUBJECT_ID}"
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
logging.info(f"Plots will be saved to: {os.path.abspath(PLOT_OUTPUT_DIR)}")
# --- End Output Directory ---

# --- Utility Functions ---
def load_lfw_subject_data(min_faces: int, n_samples: int, target_subject_id: int) -> tuple[pd.DataFrame | None, int | None, int | None, int | None]:
    """Loads LFW, balances it, and returns data ONLY for the target subject."""
    logging.info(f"Loading LFW dataset (min_faces={min_faces}, n_samples={n_samples})...")
    try:
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4, color=False)
        height, width = lfw_people.images.shape[1], lfw_people.images.shape[2]
        logging.info(f"LFW images loaded with shape: ({height}, {width})")
        data = lfw_people.data
        if data.max() <= 1.0 + 1e-6: data = (data * 255).clip(0, 255).astype(np.uint8)
        else: data = data.clip(0, 255).astype(np.uint8)
        df = pd.DataFrame(data); df['subject_id'] = lfw_people.target
        grouped = df.groupby('subject_id'); sampled_subject_ids = []; original_subject_ids_in_order = []
        for subj_id, group in grouped:
            if len(group) >= n_samples: sampled_subject_ids.append(subj_id); original_subject_ids_in_order.append(subj_id)
        if not sampled_subject_ids: logging.error(f"No subjects found with >= {n_samples} images."); return None, None, None, None
        if TARGET_SUBJECT_ID < 0 or TARGET_SUBJECT_ID >= len(original_subject_ids_in_order): logging.error(f"TARGET_SUBJECT_ID {TARGET_SUBJECT_ID} out of bounds."); return None, None, None, None
        lfw_original_target_id = original_subject_ids_in_order[TARGET_SUBJECT_ID]
        logging.info(f"Analysis Subject ID {TARGET_SUBJECT_ID} corresponds to LFW Original Target ID: {lfw_original_target_id}")
        df_subject = df[df['subject_id'] == lfw_original_target_id].copy()
        if len(df_subject) >= n_samples: df_subject_sampled = df_subject.sample(n=n_samples, random_state=42, replace=False)
        else: logging.warning(f"Subject {lfw_original_target_id} has only {len(df_subject)} images < {n_samples}. Using all."); df_subject_sampled = df_subject.copy()
        def row_to_pil_image(row_pixels): return Image.fromarray(row_pixels.values.reshape((height, width)), mode='L')
        logging.info("Converting pixel rows to PIL Image objects for the target subject...")
        pixel_columns = list(range(height * width))
        df_subject_sampled['userFaces'] = df_subject_sampled[pixel_columns].apply(row_to_pil_image, axis=1)
        df_subject_sampled = df_subject_sampled.reset_index(drop=True); df_subject_sampled['imageId'] = df_subject_sampled.index
        df_final = df_subject_sampled[['userFaces', 'imageId', 'subject_id']].copy()
        df_final.rename(columns={'subject_id': 'subject_number'}, inplace=True)
        logging.info(f"Data loaded for subject {lfw_original_target_id}: {len(df_final)} images.")
        return df_final, height, width, lfw_original_target_id
    except Exception as e: logging.critical(f"LFW loading/preparation error: {e}", exc_info=True); return None, None, None, None


# --- Utility Functions (Plotting - kept plot_multiple_images and plot_image_variants) ---
def plot_multiple_images(images, titles, h, w, n_row, n_col, figure_title, save_path):
    """Helper function to plot images in a grid and SAVE to file."""
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.suptitle(figure_title, size=16)
    plot_count = min(len(images), n_row * n_col)
    for i in range(plot_count):
        ax = plt.subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        ax.set_title(titles[i], size=9)
        ax.set_xticks(()); ax.set_yticks(())
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try: plt.savefig(save_path, bbox_inches='tight'); logging.info(f"Plot saved to: {save_path}")
    except Exception as e: logging.error(f"Failed to save plot {save_path}: {e}")
    plt.close(fig)

def plot_image_variants(original_img, variant_images, variant_labels, h, w, figure_title, save_path):
    """Plots the original image and several variants side-by-side and SAVES to file."""
    # (Unchanged)
    n_variants = len(variant_images)
    n_cols = n_variants + 1
    fig = plt.figure(figsize=(2.0 * n_cols, 2.8))
    plt.suptitle(figure_title, size=14, y=0.98)
    ax = plt.subplot(1, n_cols, 1)
    ax.imshow(original_img.reshape((h, w)), cmap=plt.cm.gray)
    ax.set_title("Original", size=10)
    ax.set_xticks(()); ax.set_yticks(())
    for i, img in enumerate(variant_images):
        ax = plt.subplot(1, n_cols, i + 2)
        ax.imshow(img.reshape((h, w)), cmap=plt.cm.gray)
        ax.set_title(variant_labels[i], size=10)
        ax.set_xticks(()); ax.set_yticks(())
    plt.tight_layout(rect=[0, 0.01, 1, 0.90])
    try: plt.savefig(save_path, bbox_inches='tight'); logging.info(f"Plot saved to: {save_path}")
    except Exception as e: logging.error(f"Failed to save plot {save_path}: {e}")
    plt.close(fig)


# --- Placeholder Functions ---
def placeholder_get_pca_results(images_np, ratio):
    if not images_np: return None, None
    n_samples, n_features = len(images_np), images_np[0].size
    n_components_max = min(n_samples, n_features)
    n_components = max(1, min(n_components_max, int(round(ratio * n_components_max))))
    flat_images = np.array([img.flatten() for img in images_np])
    try: pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(flat_images); mean_face = np.mean(flat_images, axis=0); return pca, mean_face
    except Exception as e: logging.error(f"Placeholder PCA failed for ratio {ratio}: {e}"); return None, None
def placeholder_reconstruct(images_np, pca, mean_face):
    if pca is None or not images_np: return [np.zeros_like(img) for img in images_np]
    flat_images = np.array([img.flatten() for img in images_np])
    try:
        transformed = pca.transform(flat_images); reconstructed_flat = pca.inverse_transform(transformed); h, w = images_np[0].shape
        reconstructed_images = [(img.reshape((h, w))).clip(0, 255).astype(np.uint8) for img in reconstructed_flat]; return reconstructed_images
    except Exception as e: logging.error(f"Placeholder reconstruct failed: {e}"); return [np.zeros_like(img) for img in images_np]
def placeholder_reconstruct_with_noise(images_np, pca, mean_face, epsilon):
    reconstructed_clean = placeholder_reconstruct(images_np, pca, mean_face)
    if epsilon >= (NO_NOISE_EPSILON - 1):
        return reconstructed_clean
    noise_scale = 100 / max(epsilon, 0.1)
    noisy_images = []
    for img in reconstructed_clean:
         try: noise = np.random.normal(0, noise_scale, img.shape); noisy_img = (img + noise).clip(0, 255).astype(np.uint8); noisy_images.append(noisy_img)
         except Exception as e: logging.error(f"Placeholder noise addition failed for epsilon {epsilon}: {e}"); noisy_images.append(img) # Return clean image on error
    return noisy_images


# --- Visualization Functions ---
def visualize_optimal_pipeline(subject_id, lfw_id, optimal_ratio, compromise_epsilon, h, w, original_images_np, output_dir):
    """(Optional) Visualizes the optimal pipeline steps and SAVES plots."""
    logging.info("\n--- (Optional) 1. Generating: Optimal Pipeline Result Plots ---")
    if not original_images_np: print("No original images for optimal visualization."); return
    pca_optimal, mean_face_optimal = placeholder_get_pca_results(original_images_np, optimal_ratio)
    if pca_optimal is None: print("Failed to get PCA for optimal ratio."); return
    eigenfaces_optimal = [comp.reshape(h, w) for comp in pca_optimal.components_] if pca_optimal else []
    reconstructed_optimal_np = placeholder_reconstruct_with_noise(original_images_np, pca_optimal, mean_face_optimal, compromise_epsilon)
    n_originals = min(N_IMAGES_TO_SHOW_ORIGINAL, len(original_images_np))
    save_path_orig = os.path.join(output_dir, f"subject_{lfw_id}_1a_originals.png")
    plot_multiple_images(original_images_np[:n_originals], [f"Original {i+1}" for i in range(n_originals)], h, w, 1, n_originals, f"Subject LFW ID {lfw_id}: Original Images", save_path_orig)
    if mean_face_optimal is not None:
        save_path_mean = os.path.join(output_dir, f"subject_{lfw_id}_1b_mean_face_r{optimal_ratio:.2f}.png")
        plot_multiple_images([mean_face_optimal.reshape(h,w)], ["Mean Face"], h, w, 1, 1, f"Subject {lfw_id}: Mean Face (Ratio={optimal_ratio})", save_path_mean)
    if eigenfaces_optimal:
         n_eigen = min(N_EIGENFACES_TO_SHOW, len(eigenfaces_optimal))
         save_path_eigen = os.path.join(output_dir, f"subject_{lfw_id}_1c_eigenfaces_r{optimal_ratio:.2f}.png")
         plot_multiple_images(eigenfaces_optimal[:n_eigen], [f"Eigenface {i+1}" for i in range(n_eigen)], h, w, 1, n_eigen, f"Subject {lfw_id}: Top {n_eigen} Eigenfaces (Ratio={optimal_ratio})", save_path_eigen)
    if reconstructed_optimal_np:
         n_reconstructed = min(N_IMAGES_TO_SHOW_ORIGINAL, len(reconstructed_optimal_np))
         save_path_recon = os.path.join(output_dir, f"subject_{lfw_id}_1d_reconstructed_r{optimal_ratio:.2f}_e{compromise_epsilon:.3f}.png")
         plot_multiple_images(reconstructed_optimal_np[:n_reconstructed], [f"Recon {i+1}" for i in range(n_reconstructed)], h, w, 1, n_reconstructed, f"Subject {lfw_id}: Reconstructed (R={optimal_ratio}, E={compromise_epsilon})", save_path_recon)


# NEW function replacing the old ratio viz
def visualize_ratio_impact_line(subject_id, lfw_id, ratios, h, w, original_images_np, output_dir):
    """Visualizes impact of n_components_ratio (using NO noise DP) and SAVES plot."""
    logging.info("\n--- 2. Generating: Impact of n_components_ratio Plot (No DP Noise) ---")
    if not original_images_np: print("No original images for ratio visualization."); return

    original_img_focus = original_images_np[0]
    original_img_focus_list = [original_images_np[0]]
    reconstructions_ratio = []
    titles_ratio = []

    logging.info(f"  Reconstructing image with Epsilon={NO_NOISE_EPSILON} (effectively no DP noise) for different ratios...")
    for ratio in tqdm(ratios, desc="Ratios"):
        # Calculate PCA using all subject images for better fit
        pca_ratio, mean_face_ratio = placeholder_get_pca_results(original_images_np, ratio)
        if pca_ratio is None:
            recon_img = np.zeros_like(original_img_focus) # Placeholder on failure
            titles_ratio.append(f"Ratio={ratio:.2f}\n(Failed)")
        else:
            # Reconstruct WITHOUT noise (using large epsilon) only the focus image
            recon_img_list = placeholder_reconstruct_with_noise(original_img_focus_list, pca_ratio, mean_face_ratio, NO_NOISE_EPSILON)
            if not recon_img_list:
                logging.warning(f"Ratio recon failed for ratio {ratio}")
                recon_img = np.zeros_like(original_img_focus)
            else:
                recon_img = recon_img_list[0]
            titles_ratio.append(f"Ratio={ratio:.2f}")
        reconstructions_ratio.append(recon_img)
        # --- End Adaptation ---

    if reconstructions_ratio:
        save_path = os.path.join(output_dir, f"subject_{lfw_id}_2_ratio_impact_no_noise.png")
        plot_image_variants(original_img_focus, reconstructions_ratio, titles_ratio, h, w,
                            f"Subject {lfw_id}: Impact of PCA Ratio (No DP Noise)",
                            save_path=save_path)
    else:
        logging.error("Ratio impact visualization failed, no reconstructions generated.")


# NEW function replacing the old epsilon viz
def visualize_epsilon_impact_line(subject_id, lfw_id, optimal_ratio, epsilons, h, w, original_images_np, output_dir):
    """Visualizes impact of epsilon (using OPTIMAL ratio) and SAVES plot."""
    logging.info(f"\n--- 3. Generating: Impact of Epsilon Plot (Ratio={optimal_ratio}) ---")
    if not original_images_np: print("No original images for epsilon visualization."); return

    original_img_focus = original_images_np[0]

    original_img_focus_list = [original_images_np[0]]
    reconstructions_eps = []
    titles_eps = []

    # Calculate PCA ONCE using the optimal ratio with all images
    pca_optimal, mean_face_optimal = placeholder_get_pca_results(original_images_np, optimal_ratio)
    if pca_optimal is None:
        print(f"Failed to get PCA for optimal ratio {optimal_ratio}. Cannot visualize epsilon impact.")
        return
    # --- End Adaptation ---

    logging.info(f"  Reconstructing image with Ratio={optimal_ratio} for different epsilons...")
    for eps in tqdm(epsilons, desc="Epsilons"):
        # --- !!! ADAPTATION REQUISE !!! ---
        # Reconstruct WITH noise only the focus image
        recon_img_list = placeholder_reconstruct_with_noise(original_img_focus_list, pca_optimal, mean_face_optimal, eps)
        # --- End Adaptation ---
        if not recon_img_list:
            logging.warning(f"Epsilon recon failed for eps {eps}")
            recon_img = np.zeros_like(original_img_focus)
        else:
            recon_img = recon_img_list[0]
        reconstructions_eps.append(recon_img)

        eps_label = f"Eps={eps:.1f}" if eps < (NO_NOISE_EPSILON -1) else NO_NOISE_EPSILON_LABEL
        titles_eps.append(eps_label)


    if reconstructions_eps:
        save_path = os.path.join(output_dir, f"subject_{lfw_id}_3_epsilon_impact_r{optimal_ratio:.2f}.png")
        plot_image_variants(original_img_focus, reconstructions_eps, titles_eps, h, w,
                            f"Subject {lfw_id}: Impact of Epsilon (Ratio={optimal_ratio})",
                            save_path=save_path)
    else:
        logging.error("Epsilon impact visualization failed, no reconstructions generated.")


# --- Main Execution ---
def main():
    """Orchestrates the loading and visualization, saving plots."""
    if not _modules_imported: print("Cannot run visualization due to import errors."); return

    logging.info(f"Starting visualization for Subject ID: {TARGET_SUBJECT_ID}")
    logging.info(f"Optimal Params: Ratio={SELECTED_OPTIMAL_RATIO}, Epsilon={COMPROMISE_EPSILON}")
    logging.info(f"Ratio line viz params: Ratios={RATIOS_TO_VISUALIZE}")
    logging.info(f"Epsilon line viz params: Epsilons={EPSILONS_TO_VISUALIZE}")


    # 1. Load data once
    df_subject, h, w, lfw_id = load_lfw_subject_data(MIN_FACES_PER_PERSON, N_SAMPLES_PER_PERSON, TARGET_SUBJECT_ID)
    if df_subject is None: print("Failed to load subject data. Exiting."); return

    original_images_np = [np.array(img) for img in df_subject['userFaces'].tolist()]
    if not original_images_np: print("No images loaded for the subject. Exiting."); return

    # 2. Run Visualizations

    # Viz 1 (Optional): Show optimal pipeline results separately
    visualize_optimal_pipeline(TARGET_SUBJECT_ID, lfw_id, SELECTED_OPTIMAL_RATIO, COMPROMISE_EPSILON, h, w, original_images_np, PLOT_OUTPUT_DIR)

    # Viz 2: Generate and plot the ratio impact line
    visualize_ratio_impact_line(TARGET_SUBJECT_ID, lfw_id, RATIOS_TO_VISUALIZE, h, w, original_images_np, PLOT_OUTPUT_DIR)

    # Viz 3: Generate and plot the epsilon impact line
    visualize_epsilon_impact_line(TARGET_SUBJECT_ID, lfw_id, SELECTED_OPTIMAL_RATIO, EPSILONS_TO_VISUALIZE, h, w, original_images_np, PLOT_OUTPUT_DIR)


    logging.info(f"--- Visualization Script Finished. Plots saved in '{PLOT_OUTPUT_DIR}' ---")

if __name__ == "__main__":
    main()