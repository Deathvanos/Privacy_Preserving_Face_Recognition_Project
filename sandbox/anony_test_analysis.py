import os
import sys
import io
import base64
import logging
import time
import warnings
import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# --- Adjust PYTHONPATH for local modules ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
# --- End PYTHONPATH Adjustment ---

# --- Import local modules ---
try:
    from src.modules import anony_process_pipeline
    from src.modules import utils_image
    from src.config import IMAGE_SIZE
    _modules_imported = True
except ImportError as e:
    print(f"[ERROR] Failed to import local modules: {e}")
    print(f"Check your project structure and sys.path: {sys.path}")
    _modules_imported = False
except FileNotFoundError:
    print("[ERROR] The file 'src/config.py' was not found. Please check its existence.")
    _modules_imported = False

if not _modules_imported:
      sys.exit(1)
# --- End Imports ---

# --- Global Constants and Parameters ---
# Dataset Parameters (Specific to LFW)
MIN_FACES_PER_PERSON = 20
N_SAMPLES_PER_PERSON = 10

# Analysis Parameters - Phase 2
N_COMPONENTS_RATIO_RANGE = np.sort(np.round(np.arange(0.1, 1.0, 0.1), 2))
FIXED_EPSILON_PHASE1 = 10.0

# Analysis Parameters - Phase 3
EPSILON_RANGE = np.sort(np.round(np.concatenate([
    np.arange(0.2, 1.0, 0.1),
]), 3))

# --- THEORETICAL THRESHOLDS  ---
SSIM_PRIVACY_THRESHOLD = 0.45
MSE_UTILITY_THRESHOLD = 1500
# --- END THRESHOLDS ---

OUTPUT_DIR = "analysis_results_auto_v2" # New folder name
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --- End Constants ---

# --- Logging Configuration (Console and File with UTF-8 Encoding) ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
                    encoding='utf-8',
                    force=True)

log_filename = f"pipeline_analysis_auto_v2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(OUTPUT_DIR, log_filename)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

logger.info(f"Logging initialized. Output will be saved to: {os.path.abspath(log_filepath)}")
logger.info(f"Output directory for results and plots: {os.path.abspath(OUTPUT_DIR)}")
# --- End Logging Configuration ---


# --- Utility Functions ---
def load_lfw_dataframe_for_analysis(min_faces: int, n_samples: int) -> pd.DataFrame | None:
    """Loads and prepares the LFW DataFrame for analysis (robust version)."""
    logger.info(f"Loading LFW dataset (min_faces={min_faces}, n_samples={n_samples})...")
    try:
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4, color=False)
        height, width = lfw_people.images.shape[1], lfw_people.images.shape[2]
        logger.info(f"LFW images loaded with original shape (after resize): ({height}, {width})")
        data = lfw_people.data
        if data.max() <= 1.0 + 1e-6: data = (data * 255).clip(0, 255).astype(np.uint8)
        else: data = data.clip(0, 255).astype(np.uint8)
        df = pd.DataFrame(data); df['subject_id'] = lfw_people.target
        grouped = df.groupby('subject_id'); balanced_dfs = []
        logger.info("Sampling to balance the dataset...")
        for subject_id, group in tqdm(grouped, desc="Sampling per subject"):
            if len(group) >= n_samples: balanced_dfs.append(group.sample(n=n_samples, random_state=42, replace=False))
        if not balanced_dfs: logger.error(f"No subjects found with >= {n_samples} images."); return None
        df_balanced = pd.concat(balanced_dfs); logger.info(f"{len(balanced_dfs)} subjects sampled.")
        def row_to_pil_image(row_pixels): return Image.fromarray(row_pixels.values.reshape((height, width)), mode='L')
        logger.info("Converting pixel rows to PIL Image objects..."); pixel_columns = list(range(height * width))
        df_balanced['userFaces'] = df_balanced[pixel_columns].apply(row_to_pil_image, axis=1)
        df_balanced = df_balanced.reset_index(drop=True); df_balanced['imageId'] = df_balanced.index
        df_final = df_balanced[['userFaces', 'imageId', 'subject_id']].copy()
        df_final.rename(columns={'subject_id': 'subject_number'}, inplace=True)
        logger.info(f"Final DataFrame: {df_final.shape[0]} images ({df_final['subject_number'].nunique()} subjects).")
        return df_final
    except Exception as e: logger.critical(f"LFW loading/preparation error: {e}", exc_info=True); return None

def decode_b64_to_numpy(b64_string: str) -> np.ndarray | None:
    """Decodes a base64 image string into a NumPy array (grayscale)."""
    try:
        img_bytes = base64.b64decode(b64_string); img_pil = Image.open(io.BytesIO(img_bytes)).convert('L'); return np.array(img_pil)
    except Exception as e: logger.error(f"Base64 decode error: {e}. String(start): {b64_string[:50]}...", exc_info=False); return None

def calculate_average_metrics(original_b64_list: list, reconstructed_b64_list: list) -> tuple[float, float]:
    """Calculates average MSE and SSIM for a list of images (robust version)."""
    all_mse = []; all_ssim = []; num_images = len(original_b64_list)
    if num_images == 0 or len(reconstructed_b64_list) != num_images: logger.warning("Empty/mismatched lists."); return np.nan, np.nan
    processed_count = 0
    for i in range(num_images):
        img_orig_np = decode_b64_to_numpy(original_b64_list[i]); img_recon_np = decode_b64_to_numpy(reconstructed_b64_list[i])
        if img_orig_np is None or img_recon_np is None: logger.debug(f"Img {i}: decode failed."); continue
        if img_orig_np.shape != img_recon_np.shape: logger.warning(f"Img {i}: Shape mismatch."); continue
        try: all_mse.append(mean_squared_error(img_orig_np, img_recon_np))
        except Exception as e: logger.warning(f"Img {i}: MSE error: {e}."); continue
        try:
            min_dim = min(img_orig_np.shape); win_size = min(7, min_dim) if min_dim >= 7 else (3 if min_dim >= 3 else None)
            if win_size is None or win_size % 2 == 0: logger.warning(f"Img {i}: Too small for SSIM."); all_ssim.append(np.nan); continue
            all_ssim.append(ssim(img_orig_np, img_recon_np, data_range=255.0, win_size=win_size, channel_axis=None))
            processed_count += 1
        except ValueError as e: logger.warning(f"Img {i}: SSIM value error: {e}."); all_ssim.append(np.nan); continue
        except Exception as e: logger.error(f"Img {i}: SSIM unexpected error: {e}", exc_info=True); all_ssim.append(np.nan); continue
    avg_mse = np.nanmean(all_mse) if all_mse else np.nan; avg_ssim = np.nanmean(all_ssim) if all_ssim else np.nan
    if processed_count < num_images: logger.debug(f"{processed_count}/{num_images} pairs processed OK.")
    return avg_mse, avg_ssim
# --- End Utility Functions ---


# --- Phase 1: k_same_pixel Optimization

# --- End of Phase 1


# --- Phase 2: n_components_ratio Optimization ---
def run_phase2_analysis(df_images: pd.DataFrame) -> float | None:
    """
    Finds the optimal n_components_ratio.
    Selects the ratio tested *before* the one that gives minimum MSE.
    If min MSE is at the first ratio, selects the first ratio.
    """
    logger.info("--- STARTING PHASE 2: n_components_ratio Analysis ---")
    results_phase1 = []
    ratio_at_min_mse = None
    min_mse_value = np.inf

    if df_images is None or df_images.empty:
        logger.error("Image DataFrame is empty. Stopping Phase 2.")
        return None

    df_images_copy = df_images.copy()
    tested_ratios_list = list(N_COMPONENTS_RATIO_RANGE)

    for ratio in tqdm(tested_ratios_list, desc="Phase 2: Testing n_components_ratio"):
        logger.info(f"Testing ratio = {ratio:.2f} (with epsilon = {FIXED_EPSILON_PHASE1})")
        start_time = time.time(); pipeline_output = None
        try:
            pipeline_output = anony_process_pipeline.run_pipeline(
                df_images=df_images_copy, epsilon=FIXED_EPSILON_PHASE1, n_components_ratio=ratio
            )
        except Exception as e:
            logger.error(f"Pipeline execution error for ratio={ratio:.2f}: {e}", exc_info=True)
            results_phase1.append({'ratio': ratio, 'avg_mse': np.nan, 'avg_ssim': np.nan})
            continue
        finally:
             end_time = time.time(); logger.info(f"Pipeline (ratio={ratio:.2f}) finished in {end_time - start_time:.2f} sec.")

        subject_mses = []; subject_ssims = []; valid_subjects = 0
        if not pipeline_output:
             logger.warning(f"No pipeline output for ratio={ratio:.2f}."); results_phase1.append({'ratio': ratio, 'avg_mse': np.nan, 'avg_ssim': np.nan}); continue
        for subject_id, data in pipeline_output.items():
            original_b64 = data.get('grayscale', []); reconstructed_b64 = data.get('reconstructed', [])
            if not original_b64 or not reconstructed_b64: logger.warning(f"Subject {subject_id}, ratio {ratio:.2f}: Missing data."); continue
            avg_mse, avg_ssim = calculate_average_metrics(original_b64, reconstructed_b64)
            if not np.isnan(avg_mse): subject_mses.append(avg_mse)
            if not np.isnan(avg_ssim): subject_ssims.append(avg_ssim)
            if not np.isnan(avg_mse) and not np.isnan(avg_ssim): valid_subjects += 1

        overall_avg_mse = np.nanmean(subject_mses) if subject_mses else np.nan
        overall_avg_ssim = np.nanmean(subject_ssims) if subject_ssims else np.nan
        logger.info(f"Ratio={ratio:.2f} -> Global: Avg MSE ~ {overall_avg_mse:.4f}, Avg SSIM ~ {overall_avg_ssim:.4f} ({valid_subjects} valid subjects)")
        results_phase1.append({'ratio': ratio, 'avg_mse': overall_avg_mse, 'avg_ssim': overall_avg_ssim})

        if not np.isnan(overall_avg_mse) and overall_avg_mse < min_mse_value:
             min_mse_value = overall_avg_mse
             ratio_at_min_mse = ratio
    # --- End Ratio Loop ---

    if not results_phase1: logger.error("No results collected for Phase 2."); return None
    df_results_p1 = pd.DataFrame(results_phase1)
    logger.info("\n--- PHASE 2 COMPLETE RESULTS (MSE/SSIM vs n_components_ratio) ---")
    print(df_results_p1.round(4).to_string())

    # --- AUTOMATIC SELECTION & HIGHLIGHT ---
    selected_optimal_ratio = None # Initialize
    if ratio_at_min_mse is not None:
        try:
            min_mse_index = tested_ratios_list.index(ratio_at_min_mse)

            if min_mse_index > 0:
                selected_optimal_ratio = tested_ratios_list[min_mse_index - 1]
                logger.info(f"Minimum MSE occurred at ratio={ratio_at_min_mse:.2f}.")
                logger.info(f"Selecting the preceding ratio as optimal: {selected_optimal_ratio:.2f}")
            else:
                selected_optimal_ratio = tested_ratios_list[0]
                logger.warning(f"Minimum MSE occurred at the first tested ratio ({ratio_at_min_mse:.2f}). Selecting this first ratio as optimal ({selected_optimal_ratio:.2f}) according to the rule.")

        except ValueError:
             logger.error(f"Could not find ratio {ratio_at_min_mse} in the tested list, cannot apply selection logic.")
             selected_optimal_ratio = ratio_at_min_mse # Fallback to the min mse ratio itself
             logger.warning(f"Falling back to selecting the ratio with minimum MSE: {selected_optimal_ratio:.2f}")

        logger.info("\n**************************************************************************")
        logger.info(f"*** AUTOMATICALLY SELECTED OPTIMAL n_components_ratio (Phase 2) ***")
        logger.info(f"*** Based on Ratio Preceding Minimum MSE: ratio = {selected_optimal_ratio:.2f} ***")
        selected_metrics = df_results_p1[df_results_p1['ratio'] == selected_optimal_ratio].iloc[0]
        logger.info(f"*** -> Metrics at this ratio: MSE ~ {selected_metrics['avg_mse']:.4f}, SSIM ~ {selected_metrics['avg_ssim']:.4f} ***")
        logger.info("**************************************************************************")

    else:
        logger.warning("\nCould not automatically determine the ratio with minimum MSE. Optimal ratio not selected.")


    csv_path_p1 = os.path.join(OUTPUT_DIR, "phase1_n_components_ratio_analysis.csv")
    try: df_results_p1.to_csv(csv_path_p1, index=False); logger.info(f"Phase 2 results saved to: {csv_path_p1}")
    except Exception as e: logger.error(f"Error saving Phase 2 CSV ({csv_path_p1}): {e}", exc_info=True)

    # --- Plotting Phase 2 (Combined Plot) ---
    logger.info("Generating combined plot for Phase 2...")
    try:
        fig, ax1 = plt.subplots(figsize=(12, 7)) # Single plot setup
        fig.suptitle("Phase 2: MSE/SSIM vs n_components_ratio", fontsize=16)
        df_plot = df_results_p1.dropna(subset=['avg_mse', 'avg_ssim'])
        if df_plot.empty: logger.warning("No valid data to plot for Phase 2."); plt.close(fig); return selected_optimal_ratio

        ratios_plot = df_plot['ratio']
        color_mse = 'tab:red'; color_ssim = 'tab:blue'

        # Axis 1: MSE
        ax1.set_xlabel("n_components_ratio", fontsize=10)
        ax1.set_ylabel("Average MSE (Lower = Better Utility)", color=color_mse, fontsize=10)
        ln1 = ax1.plot(ratios_plot, df_plot['avg_mse'], color=color_mse, marker='o', linestyle='-', linewidth=2, label='MSE')
        ax1.tick_params(axis='y', labelcolor=color_mse, labelsize=9)
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)

        # Axis 2: SSIM
        ax2 = ax1.twinx()
        ax2.set_ylabel("Average SSIM (Higher = More Similar)", color=color_ssim, fontsize=10)
        ln2 = ax2.plot(ratios_plot, df_plot['avg_ssim'], color=color_ssim, marker='x', linestyle='--', linewidth=2, label='SSIM')
        ax2.tick_params(axis='y', labelcolor=color_ssim, labelsize=9)
        if not df_plot.empty: ax2.set_ylim(bottom=max(0, df_plot['avg_ssim'].min() - 0.1), top=min(1.0, df_plot['avg_ssim'].max() + 0.1))

        # Shared X Axis config
        ax1.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5); ax1.minorticks_on()
        ax1.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.3)

        ax1.set_xticks(tested_ratios_list); ax1.tick_params(axis='x', rotation=30, labelsize=9)
        ax1.set_xlim(left=min(tested_ratios_list)-0.05, right=max(tested_ratios_list)+0.05)

        ln_selected = []
        if selected_optimal_ratio is not None:
            ln_selected = [ax1.axvline(selected_optimal_ratio, color='green', linestyle='-', linewidth=2.5, label=f'Selected Ratio = {selected_optimal_ratio:.2f}')]

        if ratio_at_min_mse is not None and not np.isinf(min_mse_value):
             ax1.scatter([ratio_at_min_mse], [min_mse_value], marker='v', color='purple', s=80, zorder=5, label=f'Actual Min MSE at {ratio_at_min_mse:.2f}')

        # Legend
        lns = ln1 + ln2 + ln_selected
        if ratio_at_min_mse is not None:
             scatter_label = f'Actual Min MSE at {ratio_at_min_mse:.2f}'
             handles, labels = ax1.get_legend_handles_labels()
             if scatter_label not in labels:
                   lns.append(plt.Line2D([0], [0], marker='v', color='purple', linestyle='None', markersize=8)) # Dummy handle for legend
                   labs_final = [l.get_label() for l in lns]
                   labs_final.append(scatter_label) # Add the label text

             else: labs_final = [l.get_label() for l in lns]

        else: labs_final = [l.get_label() for l in lns]


        fig.legend(lns, labs_final, loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=3, fontsize=9)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plot_path_p1 = os.path.join(OUTPUT_DIR, "phase1_ratio_combined_plot.png") # New filename
        plt.savefig(plot_path_p1); logger.info(f"Phase 2 plot saved to: {plot_path_p1}")
        plt.close(fig)

    except Exception as e: logger.error(f"Error generating Phase 2 plot: {e}", exc_info=True)

    return selected_optimal_ratio
# --- End Phase 2 ---


# --- Phase 3: Epsilon Optimization ---
def run_phase3_analysis(df_images: pd.DataFrame, optimal_n_components_ratio: float):
    """Finds the best epsilon by analyzing the MSE/SSIM trade-off."""
    logger.info("--- STARTING PHASE 3: Epsilon Analysis ---")
    if optimal_n_components_ratio is None: logger.error("Optimal n_components_ratio not provided. Stopping Phase 3."); return
    logger.info(f"Using optimal n_components_ratio found: {optimal_n_components_ratio:.2f}")
    results_phase2 = []
    if df_images is None or df_images.empty: logger.error("Image DataFrame is empty. Stopping Phase 3."); return
    df_images_copy = df_images.copy()

    for epsilon in tqdm(EPSILON_RANGE, desc="Phase 3: Testing epsilon"):
        logger.info(f"Testing epsilon = {epsilon:.3f} (with ratio = {optimal_n_components_ratio:.2f})")
        start_time = time.time(); pipeline_output = None
        try:
            pipeline_output = anony_process_pipeline.run_pipeline(
                df_images=df_images_copy, epsilon=epsilon, n_components_ratio=optimal_n_components_ratio
            )
        except ValueError as e: logger.error(f"Value error for epsilon={epsilon:.3f}: {e}. Skipping."); results_phase2.append({'epsilon': epsilon, 'avg_mse': np.nan, 'avg_ssim': np.nan}); continue
        except Exception as e: logger.error(f"Pipeline execution error for epsilon={epsilon:.3f}: {e}", exc_info=True); results_phase2.append({'epsilon': epsilon, 'avg_mse': np.nan, 'avg_ssim': np.nan}); continue
        finally: end_time = time.time(); logger.info(f"Pipeline (epsilon={epsilon:.3f}) finished in {end_time - start_time:.2f} sec.")

        subject_mses = []; subject_ssims = []; valid_subjects = 0
        if not pipeline_output: logger.warning(f"No pipeline output for epsilon={epsilon:.3f}."); results_phase2.append({'epsilon': epsilon, 'avg_mse': np.nan, 'avg_ssim': np.nan}); continue
        for subject_id, data in pipeline_output.items():
            original_b64 = data.get('grayscale', []); reconstructed_b64 = data.get('reconstructed', [])
            if not original_b64 or not reconstructed_b64: logger.warning(f"Subject {subject_id}, epsilon {epsilon:.3f}: Missing data."); continue
            avg_mse, avg_ssim = calculate_average_metrics(original_b64, reconstructed_b64)
            if not np.isnan(avg_mse): subject_mses.append(avg_mse)
            if not np.isnan(avg_ssim): subject_ssims.append(avg_ssim)
            if not np.isnan(avg_mse) and not np.isnan(avg_ssim): valid_subjects += 1

        overall_avg_mse = np.nanmean(subject_mses) if subject_mses else np.nan
        overall_avg_ssim = np.nanmean(subject_ssims) if subject_ssims else np.nan
        logger.info(f"Epsilon={epsilon:.3f} -> Global: Avg MSE ~ {overall_avg_mse:.4f}, Avg SSIM ~ {overall_avg_ssim:.4f} ({valid_subjects} valid subjects)")
        results_phase2.append({'epsilon': epsilon, 'avg_mse': overall_avg_mse, 'avg_ssim': overall_avg_ssim})
    # --- End Epsilon Loop ---

    if not results_phase2: logger.error("No results collected for Phase 3."); return
    df_results_p2 = pd.DataFrame(results_phase2)
    logger.info("\n--- PHASE 3 COMPLETE RESULTS (MSE/SSIM vs epsilon) ---")
    print(df_results_p2.round(4).to_string())
    csv_filename_p2 = f"phase2_epsilon_analysis_ratio_{optimal_n_components_ratio:.2f}.csv"
    csv_path_p2 = os.path.join(OUTPUT_DIR, csv_filename_p2)
    try: df_results_p2.to_csv(csv_path_p2, index=False); logger.info(f"Phase 3 results saved to: {csv_path_p2}")
    except Exception as e: logger.error(f"Error saving Phase 3 CSV ({csv_path_p2}): {e}", exc_info=True)

    # --- AUTOMATED COMPROMISE EPSILON CALCULATION ---
    logger.info("\n--- Determining Automated Compromise Epsilon ---")
    compromise_epsilon = None; epsilon_util_priority = None; epsilon_priv_priority = None
    df_valid_p2 = df_results_p2.dropna(subset=['avg_mse', 'avg_ssim']).copy()
    if not df_valid_p2.empty:
        util_ok = df_valid_p2[df_valid_p2['avg_mse'] < MSE_UTILITY_THRESHOLD].copy()
        if not util_ok.empty:
            util_candidate_row = util_ok.loc[util_ok['avg_ssim'].idxmin()]
            epsilon_util_priority = util_candidate_row['epsilon']
            logger.info(f"Utility Priority Candidate: Epsilon={epsilon_util_priority:.3f} (MSE={util_candidate_row['avg_mse']:.2f}, SSIM={util_candidate_row['avg_ssim']:.4f})")
        else: logger.warning(f"No Epsilon found with MSE < {MSE_UTILITY_THRESHOLD}")
        priv_ok = df_valid_p2[df_valid_p2['avg_ssim'] < SSIM_PRIVACY_THRESHOLD].copy()
        if not priv_ok.empty:
            priv_candidate_row = priv_ok.loc[priv_ok['avg_mse'].idxmin()]
            epsilon_priv_priority = priv_candidate_row['epsilon']
            logger.info(f"Privacy Priority Candidate: Epsilon={epsilon_priv_priority:.3f} (MSE={priv_candidate_row['avg_mse']:.2f}, SSIM={priv_candidate_row['avg_ssim']:.4f})")
        else: logger.warning(f"No Epsilon found with SSIM < {SSIM_PRIVACY_THRESHOLD}")

        if epsilon_util_priority is not None and epsilon_priv_priority is not None:
            compromise_epsilon = (epsilon_util_priority + epsilon_priv_priority) / 2.0
            logger.info(f"Both candidates found. Compromise Epsilon set to midpoint: {compromise_epsilon:.3f}")
        elif epsilon_util_priority is not None: compromise_epsilon = epsilon_util_priority; logger.info(f"Only Utility Priority found. Compromise Epsilon set to: {compromise_epsilon:.3f}")
        elif epsilon_priv_priority is not None: compromise_epsilon = epsilon_priv_priority; logger.info(f"Only Privacy Priority found. Compromise Epsilon set to: {compromise_epsilon:.3f}")
        else: logger.warning("Neither priority candidates found. Cannot determine automated compromise epsilon.")
    else: logger.error("No valid (non-NaN) results in Phase 3 for compromise calculation.")

    # --- Plotting Phase 3 ---
    logger.info("Generating plot for Phase 3...")
    try:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        fig.suptitle(f'Phase 3: MSE/SSIM vs Epsilon (Ratio={optimal_n_components_ratio:.2f})', fontsize=14)
        df_plot_p2 = df_results_p2.dropna(subset=['avg_mse', 'avg_ssim'])
        if df_plot_p2.empty: logger.warning("No valid data to plot for Phase 3."); plt.close(fig); return
        epsilons_plot = df_plot_p2['epsilon']
        color_mse = 'tab:red'; color_ssim = 'tab:blue'
        ax1.set_xlabel('Epsilon (Higher = Less Privacy / Less Noise)', fontsize=10); ax1.set_ylabel('Avg MSE (Lower = Better Utility)', color=color_mse, fontsize=10)
        ln1 = ax1.plot(epsilons_plot, df_plot_p2['avg_mse'], color=color_mse, marker='o', linestyle='-', linewidth=2, label='MSE')
        ax1.tick_params(axis='y', labelcolor=color_mse, labelsize=9)
        ln_mse_thresh = ax1.axhline(y=MSE_UTILITY_THRESHOLD, color=color_mse, linestyle=':', linewidth=2.5, label=f'MSE Thr. ({MSE_UTILITY_THRESHOLD})')
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Avg SSIM (Lower = Better Privacy)', color=color_ssim, fontsize=10)
        ln2 = ax2.plot(epsilons_plot, df_plot_p2['avg_ssim'], color=color_ssim, marker='x', linestyle='--', linewidth=2, label='SSIM')
        ax2.tick_params(axis='y', labelcolor=color_ssim, labelsize=9)
        ln_ssim_thresh = ax2.axhline(y=SSIM_PRIVACY_THRESHOLD, color=color_ssim, linestyle=':', linewidth=2.5, label=f'SSIM Thr. ({SSIM_PRIVACY_THRESHOLD})')
        ax2.grid(True, which='major', axis='y', linestyle='-.', linewidth=0.5)
        if not df_plot_p2.empty: ax2.set_ylim(bottom=max(0, df_plot_p2['avg_ssim'].min() - 0.1), top=min(1.0, df_plot_p2['avg_ssim'].max() + 0.1))

        ln_compromise = []
        if compromise_epsilon is not None:
            ln_compromise = [ax1.axvline(x=compromise_epsilon, color='green', linestyle='-', linewidth=2.5, label=f'Compromise ε = {compromise_epsilon:.3f}')]
            logger.info(f"Plotting compromise epsilon line at: {compromise_epsilon:.3f}")
        else: logger.warning("No compromise epsilon determined, line not plotted.")

        ax1.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5); ax1.minorticks_on(); ax1.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.3)
        ax1.set_xticks(EPSILON_RANGE); ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.set_xlim(left=min(EPSILON_RANGE)-0.1, right=max(EPSILON_RANGE)+0.1)
        lns = ln1 + [ln_mse_thresh] + ln2 + [ln_ssim_thresh] + ln_compromise
        labs = [l.get_label() for l in lns]
        fig.legend(lns, labs, loc='lower center', bbox_to_anchor=(0.5, -0.18), fancybox=True, shadow=True, ncol=3, fontsize=9)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plot_filename_p2 = f"phase2_epsilon_tradeoff_plot_ratio_{optimal_n_components_ratio:.2f}.png"
        plot_path_p2 = os.path.join(OUTPUT_DIR, plot_filename_p2)
        plt.savefig(plot_path_p2); logger.info(f"Phase 3 plot saved to: {plot_path_p2}")
        plt.close(fig)
    except Exception as e: logger.error(f"Error generating Phase 3 plot: {e}", exc_info=True)

    # --- FINAL SUMMARY LOG ---
    logger.info("\n--- AUTOMATED TRADE-OFF SUMMARY (Phase 3) ---")
    if compromise_epsilon is not None:
         compromise_row = df_valid_p2.iloc[(df_valid_p2['epsilon'] - compromise_epsilon).abs().argsort()[:1]]
         if not compromise_row.empty:
              compromise_row = compromise_row.iloc[0]
              logger.info("\n**************************************************************************")
              logger.info(f"*** AUTOMATICALLY SELECTED COMPROMISE Epsilon (Phase 3) ***")
              logger.info(f"*** Based on midpoint/available candidates: Epsilon ~ {compromise_epsilon:.3f} ***")
              logger.info(f"*** -> Metrics near this Epsilon: SSIM ~ {compromise_row['avg_ssim']:.4f}, MSE ~ {compromise_row['avg_mse']:.2f} ***")
              logger.info(f"*** -> Compare with Thresholds: SSIM (<{SSIM_PRIVACY_THRESHOLD}? {'Yes' if compromise_row['avg_ssim'] < SSIM_PRIVACY_THRESHOLD else 'NO'}), MSE (<{MSE_UTILITY_THRESHOLD}? {'Yes' if compromise_row['avg_mse'] < MSE_UTILITY_THRESHOLD else 'NO'}) ***")
              logger.info("**************************************************************************")
         else: logger.warning("Could not find data point near calculated compromise epsilon to display metrics.")
    else:
         logger.warning("\nNo suitable compromise Epsilon could be automatically determined based on the defined thresholds and logic.")
         logger.warning("Please review the Phase 3 results table and plot to make a manual selection or adjust thresholds/logic.")
# --- End Phase 3 ---


# --- Main Execution Function ---
def main():
    """Orchestrates the execution of the analysis phases."""
    start_global_time = time.time()
    logger.info("==========================================================")
    logger.info("===== STARTING AUTOMATED PIPELINE ANALYSIS SCRIPT =====")
    logger.info("==========================================================")
    df_lfw = load_lfw_dataframe_for_analysis(min_faces=MIN_FACES_PER_PERSON, n_samples=N_SAMPLES_PER_PERSON)
    if df_lfw is None: logger.critical("LFW data loading failed. Stopping script."); return

    # Phase 2
    auto_optimal_ratio = run_phase2_analysis(df_lfw)

    if auto_optimal_ratio is not None:
        # Phase 3
        run_phase3_analysis(df_lfw, auto_optimal_ratio)
    else:
        logger.error("Phase 2 failed or did not find an optimal ratio. Phase 3 cancelled.")

    end_global_time = time.time()
    logger.info("==========================================================")
    logger.info("===== ANALYSIS FINISHED =====")
    logger.info(f"Total execution time: {end_global_time - start_global_time:.2f} seconds.")
    logger.info(f"Results, plots, and logs saved in directory: {os.path.abspath(OUTPUT_DIR)}")
    logger.info("==========================================================")
    logger.info("\n--- IDEAS FOR FUTURE ANALYSES ---")

    logger.info("  - Threshold Validation: Adjust thresholds based on user studies or ML model performance.")
    logger.info("  - ML Model Testing: Directly measure recognition performance vs. epsilon.")
    logger.info("  - Visual Inspection: Save & compare original vs. reconstructed images.")
    logger.info("  - Per-Subject Analysis: Study metric variability across subjects.")
    logger.info("  - Parameter Sensitivity: Test impact of N_SAMPLES_PER_PERSON or thresholds.")
    logger.info("  - Alternative Metrics/Logic: Explore different metrics or compromise selection methods.")
    logger.info("  - SSIM vs MSE Plot: Visualize the Pareto front directly.")
    logger.info("==========================================")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main()