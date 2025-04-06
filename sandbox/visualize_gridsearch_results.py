# -*- coding: utf-8 -*-
import os
import sys
import logging
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
# warnings.filterwarnings("ignore", message=".*legend.*") # Ignore KDE+Scatter legend warnings if needed

# --- Global Constants ---
CSV_FILENAME = "grid_search_3d_results.csv"
ANALYSIS_DIR = "analysis_results_grid_search_3d"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# === THRESHOLDS FOR ANALYSIS (Updated as requested) ===
SSIM_THRESHOLD_MAX = 0.45 # Updated
MSE_THRESHOLD_MAX = 1500  # Updated
# =====================================================

# === Parameters for scatter plot centering/zoom ===
# Define the half-width/height of the window around the threshold intersection point
# Adjust these values to zoom more or less
DELTA_MSE_ZOOM = 1000 # Window of +/- 1000 around MSE=1500
DELTA_SSIM_ZOOM = 0.15 # Window of +/- 0.15 around SSIM=0.45
# ==================================================

# --- Logging Configuration ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]: logger.removeHandler(handler)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.info(f"Analysis directory (CSV read & plot save): {os.path.abspath(ANALYSIS_DIR)}")
logger.info(f"Thresholds used for analysis: SSIM < {SSIM_THRESHOLD_MAX}, MSE < {MSE_THRESHOLD_MAX}")
# --- End Logging ---

# --- Plotting Functions ---

def plot_ssim_vs_mse_density_candidates(df: pd.DataFrame, ssim_max: float, mse_max: float, output_dir: str):
    """
    Generates a combined plot: density map (KDE) for all points
    and an overlaid scatter plot for candidate points (meeting thresholds).
    Candidates are colored by k, size varies with epsilon.
    The plot is centered on the threshold intersection and has no main title.
    Saves the plot as PDF.
    """
    logger.info("Generating Density + Candidates plot (SSIM vs MSE) - Centered...")
    # ... (plotting logic remains the same until savefig) ...
    if df.empty:
        logger.warning("DataFrame is empty. Density + Candidates plot not generated.")
        return

    df_plot = df.dropna(subset=['avg_mse', 'avg_ssim', 'k', 'epsilon']).copy()
    if df_plot.empty:
        logger.warning("No valid (non-NaN) data for Density + Candidates plot.")
        return

    # Identify candidates
    df_plot['Target Zone'] = (df_plot['avg_ssim'] < ssim_max) & (df_plot['avg_mse'] < mse_max)
    candidates = df_plot[df_plot['Target Zone']].copy()
    n_candidates = len(candidates)
    logger.info(f"Number of candidates found in the target zone: {n_candidates}")

    plt.figure(figsize=(12, 8))
    ax = plt.gca() # Get current axes

    # 1. Draw density map (KDE) for all points in the background
    sns.kdeplot(
        data=df_plot, x='avg_mse', y='avg_ssim', fill=True,
        cmap="Blues", alpha=0.5, ax=ax, warn_singular=False
    )

    # 2. Draw candidate points on top
    if not candidates.empty:
        scatter = sns.scatterplot(
            data=candidates, x='avg_mse', y='avg_ssim', hue='k', size='epsilon',
            palette='viridis', sizes=(40, 250), alpha=0.8, legend='full', ax=ax
        )
        # Adjust legend for candidates
        handles, labels = scatter.get_legend_handles_labels()
        plt.legend(handles=handles, labels=labels, title='Candidates (k / ε)', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    else:
        logger.warning("No candidates to display in the target zone.")
        # Create an informative legend if no candidates
        ax.legend(title='No candidates found', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 3. Add threshold lines
    ax.axhline(y=ssim_max, color='red', linestyle='--', linewidth=1.5, label=f'SSIM Threshold ({ssim_max:.2f})')
    ax.axvline(x=mse_max, color='blue', linestyle='--', linewidth=1.5, label=f'MSE Threshold ({mse_max})')

    # 4. Labels (Main title removed)
    # ax.set_title('Result Density and Candidates (SSIM vs MSE)') # Title removed
    ax.set_xlabel(f'Avg MSE (Threshold={mse_max})')
    ax.set_ylabel(f'Avg SSIM (Threshold={ssim_max:.2f})')
    ax.grid(True, linestyle=':', alpha=0.5)

    # 5. Center the plot on the threshold intersection
    x_center = mse_max
    y_center = ssim_max
    x_lim_lower = max(0, x_center - DELTA_MSE_ZOOM) # Ensure >= 0
    x_lim_upper = x_center + DELTA_MSE_ZOOM
    y_lim_lower = max(0, y_center - DELTA_SSIM_ZOOM) # Ensure >= 0
    y_lim_upper = y_center + DELTA_SSIM_ZOOM

    ax.set_xlim(x_lim_lower, x_lim_upper)
    ax.set_ylim(y_lim_lower, y_lim_upper)
    logger.info(f"Plot centered on MSE={x_center}, SSIM={y_center}. X Limits=[{x_lim_lower:.0f}, {x_lim_upper:.0f}], Y Limits=[{y_lim_lower:.2f}, {y_lim_upper:.2f}]")

    # Re-add threshold text labels (position adjusted for visibility within zoom)
    ax.text(min(x_lim_upper * 0.98, ax.get_xlim()[1]), ssim_max, f' SSIM={ssim_max:.2f}', color='red', va='center', ha='right', backgroundcolor='white', alpha=0.7)
    ax.text(mse_max, min(y_lim_upper * 0.98, ax.get_ylim()[1]), f' MSE={mse_max} ', color='blue', va='top', ha='center', rotation=90, backgroundcolor='white', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for external legend

    # --- Modification for PDF Output ---
    plot_filename_base = "density_candidates_ssim_vs_mse_centered"
    plot_filename_pdf = os.path.join(output_dir, f"{plot_filename_base}.pdf")
    try:
        plt.savefig(plot_filename_pdf, format='pdf', bbox_inches='tight') # Save as PDF
        logger.info(f"Density + Candidates plot (Centered) saved as PDF: {plot_filename_pdf}")
        plt.close()
    except Exception as e:
        logger.error(f"Error saving Density + Candidates plot as PDF: {e}", exc_info=True)
        plt.close()
    # --- End Modification ---


def plot_metric_trends_dual_axis(df: pd.DataFrame, param_x: str, output_dir: str, ssim_max: float, mse_max: float):
    """
    Generates a plot showing the average trend of SSIM (left axis, blue)
    and MSE (right axis, red) against the specified parameter (param_x).
    Calculated on the entire provided dataset. Includes threshold lines.
    Saves the plot as PDF.
    """
    logger.info(f"Generating trend plot for parameter: {param_x}...")
    # ... (plotting logic remains the same until savefig) ...
    if df.empty:
        logger.warning(f"DataFrame is empty. Trend plot for '{param_x}' not generated.")
        return
    if param_x not in df.columns:
        logger.error(f"Parameter '{param_x}' not found in DataFrame.")
        return

    trends = df.groupby(param_x)[['avg_ssim', 'avg_mse']].mean().reset_index()

    if trends.empty:
        logger.warning(f"No valid aggregated data for trend plot '{param_x}'.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_ssim = 'tab:blue'
    ax1.set_xlabel(param_x.replace('_', ' ').title()) # X label
    ax1.set_ylabel('Avg SSIM', color=color_ssim)
    line1 = ax1.plot(trends[param_x], trends['avg_ssim'], color=color_ssim, marker='o', linestyle='-', label='Avg SSIM')
    ax1.tick_params(axis='y', labelcolor=color_ssim)
    ax1.axhline(y=ssim_max, color=color_ssim, linestyle=':', linewidth=2, label=f'SSIM Threshold ({ssim_max:.2f})')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6) # Grid for primary y-axis only

    ax2 = ax1.twinx()
    color_mse = 'tab:red'
    ax2.set_ylabel('Avg MSE', color=color_mse)
    line2 = ax2.plot(trends[param_x], trends['avg_mse'], color=color_mse, marker='s', linestyle='--', label='Avg MSE')
    ax2.tick_params(axis='y', labelcolor=color_mse)
    ax2.axhline(y=mse_max, color=color_mse, linestyle=':', linewidth=2, label=f'MSE Threshold ({mse_max})')
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
    ax2.ticklabel_format(style='plain', axis='y')

    plt.title(f'Avg SSIM & MSE Trend vs {param_x.replace("_", " ").title()}')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    handles_thresholds = [ax1.get_lines()[1], ax2.get_lines()[1]] # Get threshold lines
    labels_thresholds = [h.get_label() for h in handles_thresholds]
    ax1.legend(lines + handles_thresholds, labels + labels_thresholds, loc='best')
    fig.tight_layout() # Auto-adjust layout

    # --- Modification for PDF Output ---
    plot_filename_base = f"trend_ssim_mse_vs_{param_x}"
    plot_filename_pdf = os.path.join(output_dir, f"{plot_filename_base}.pdf")
    try:
        plt.savefig(plot_filename_pdf, format='pdf', bbox_inches='tight') # Save as PDF
        logger.info(f"Trend plot for '{param_x}' saved as PDF: {plot_filename_pdf}")
        plt.close(fig) # Close figure to free memory
    except Exception as e:
        logger.error(f"Error saving trend plot for '{param_x}' as PDF: {e}", exc_info=True)
        plt.close(fig)
    # --- End Modification ---


def filter_and_display_candidates(df: pd.DataFrame, ssim_max: float, mse_max: float):
    """
    Filters the DataFrame to find candidates meeting the thresholds
    and prints a sorted summary.
    (Function kept)
    """
    logger.info(f"Filtering candidates meeting thresholds (SSIM<{ssim_max}, MSE<{mse_max})...")
    # ... (code de la fonction inchangé) ...
    if df.empty:
        logger.warning("DataFrame is empty. Cannot filter.")
        return pd.DataFrame() # Return empty DataFrame

    df_filtered = df.copy()
    candidates = df_filtered[
        (df_filtered['avg_ssim'] < ssim_max) &
        (df_filtered['avg_mse'] < mse_max)
    ].copy()

    if candidates.empty:
        logger.warning(f"No parameter combinations found meeting thresholds.")
    else:
        logger.info(f"Number of candidate combinations found: {len(candidates)}")
        # Sort candidates, e.g., by SSIM ascending then MSE ascending
        candidates_sorted = candidates.sort_values(by=['avg_ssim', 'avg_mse'], ascending=[True, True])

        logger.info(f"\n--- Top {min(15, len(candidates_sorted))} Candidates (sorted by SSIM, then MSE) ---")
        # Print table to log/console
        with pd.option_context('display.max_rows', 15, 'display.max_columns', None, 'display.width', 1000):
            print(candidates_sorted.round(4)) # Display with 4 decimal places for readability
        logger.info("--- End of candidate list ---")
        return candidates_sorted # Return sorted candidates
    return pd.DataFrame() # Return empty DataFrame if no candidates


# --- Main Execution Logic ---
def main_visualize():
    """Main function to load data and generate visualizations."""
    logger.info(f"--- STARTING VISUALIZATION SCRIPT (Centered Scatter + Trends) - PDF Output ---") # Updated title
    csv_path = os.path.join(ANALYSIS_DIR, CSV_FILENAME)

    # 1. Load data from CSV
    try:
        logger.info(f"Loading results from: {csv_path}")
        df_results = pd.read_csv(csv_path)
        logger.info(f"Data loaded successfully. Initial shape: {df_results.shape}")
        required_cols = ['k', 'n_component_ratio', 'epsilon', 'avg_mse', 'avg_ssim']
        if not all(col in df_results.columns for col in required_cols):
            logger.error(f"Missing columns in CSV. Expected: {required_cols}. Found: {list(df_results.columns)}")
            return
        logger.info("NaN count per column before cleaning:\n" + str(df_results.isnull().sum()))
        # Drop rows with NaN in essential columns
        df_results.dropna(subset=['avg_ssim', 'avg_mse', 'k', 'epsilon', 'n_component_ratio'], inplace=True)
        logger.info(f"Shape after dropping rows with NaNs: {df_results.shape}")

    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        logger.error("Please run the Grid Search script first or check the path.")
        return
    except Exception as e:
        logger.error(f"Error loading or reading CSV file: {e}", exc_info=True)
        return

    # 2. Generate Density + Candidates plot (SSIM vs MSE) - Centered
    #    Uses all valid data.
    plot_ssim_vs_mse_density_candidates(df_results, SSIM_THRESHOLD_MAX, MSE_THRESHOLD_MAX, ANALYSIS_DIR)

    # 3. Generate Average Trend plots (calculated on ALL valid data)
    params_to_plot_trends = ['k', 'n_component_ratio', 'epsilon']
    for param in params_to_plot_trends:
        plot_metric_trends_dual_axis(df_results, param, ANALYSIS_DIR, SSIM_THRESHOLD_MAX, MSE_THRESHOLD_MAX)

    # 4. Filter and display promising candidates (globally)
    logger.info("\n=== Candidate Analysis (All k) ===")
    filter_and_display_candidates(df_results, SSIM_THRESHOLD_MAX, MSE_THRESHOLD_MAX)


    logger.info("--- VISUALIZATION SCRIPT (Centered Scatter + Trends) FINISHED ---")
    logger.info(f"Plots and candidate analysis generated.")
    logger.info(f"Check PDF files in: {os.path.abspath(ANALYSIS_DIR)}") # Updated message
    logger.info("Remember: Validate candidates with visual inspection and ML evaluation!")

if __name__ == "__main__":
    main_visualize()
