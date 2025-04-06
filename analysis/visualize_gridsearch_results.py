import os
import sys
import logging
import warnings
import math
import io # Required for in-memory buffer
import base64 # Required for encoding

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# --- Initial Configuration ---
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# --- Default Configuration (can be overridden by function arguments) ---
DEFAULT_SSIM_THRESHOLD = 0.45
DEFAULT_MSE_THRESHOLD = 1500
DELTA_MSE_ZOOM = 1000 # Window of +/- 1000 around MSE threshold for zoom
DELTA_SSIM_ZOOM = 0.15 # Window of +/- 0.15 around SSIM threshold for zoom
# Note: CSV_FILENAME and ANALYSIS_DIR are removed as globals, now passed as arguments

# --- Logging Configuration ---
# Configure logger once globally
logger = logging.getLogger(__name__) # Use module-specific logger name
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# Avoid adding handlers multiple times if script/module is reloaded
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    logger.info("Logger configured for console output.")
# File logging can be added within the main execution function if needed

# --- Plotting Functions (Modified to return Base64 string) ---

def plot_ssim_vs_mse_density_candidates(
    df: pd.DataFrame,
    ssim_threshold: float,
    mse_threshold: float,
    output_dir: str
) -> str | None: # Modified return type annotation
    """
    Generates a combined plot: density map (KDE) for all points
    and an overlaid scatter plot for candidate points (meeting thresholds).
    Candidates are colored by k, size varies with epsilon.
    Highlights the candidate closest to the threshold intersection.
    The plot is centered on the threshold intersection and has no main title.
    Saves the plot as PDF to the specified output directory.
    Returns a base64 encoded PNG image string for HTML embedding.

    Args:
        df: DataFrame with grid search results.
        ssim_threshold: Maximum SSIM value for candidate selection.
        mse_threshold: Maximum MSE value for candidate selection.
        output_dir: Directory to save the plot PDF file.

    Returns:
        A base64 encoded string for embedding the plot in HTML (data URI),
        or None if plotting failed.
    """
    logger.info(f"Generating Density + Candidates plot (SSIM vs MSE) - Thresholds (SSIM<{ssim_threshold}, MSE<{mse_threshold})...")
    if df.empty:
        logger.warning("DataFrame is empty. Plot not generated.")
        return None # Return None if no data

    df_plot = df.dropna(subset=['avg_mse', 'avg_ssim', 'k', 'epsilon', 'n_component_ratio']).copy()
    if df_plot.empty:
        logger.warning("No valid (non-NaN) data for Density + Candidates plot.")
        return None # Return None if no valid data

    # ... (rest of the plotting logic remains the same) ...
    # Identify candidates
    df_plot['Target Zone'] = (df_plot['avg_ssim'] < ssim_threshold) & (df_plot['avg_mse'] < mse_threshold)
    candidates = df_plot[df_plot['Target Zone']].copy()
    n_candidates = len(candidates)
    logger.info(f"Number of candidates found in the target zone: {n_candidates}")

    # --- Find the candidate closest to the intersection point ---
    closest_point = None
    min_dist = float('inf')
    if not candidates.empty:
        candidates['distance_to_target'] = candidates.apply(
            lambda row: math.sqrt(
                ((row['avg_mse'] - mse_threshold) ** 2) +
                ((row['avg_ssim'] - ssim_threshold) ** 2)
            ),
            axis=1
        )
        closest_idx = candidates['distance_to_target'].idxmin()
        closest_point = candidates.loc[closest_idx]
        min_dist = closest_point['distance_to_target']
        logger.info(f"Closest candidate found (distance={min_dist:.4f}):")
        closest_params_str = (
            f"  k={closest_point['k']:.0f}, "
            f"ratio={closest_point['n_component_ratio']:.3f}, "
            f"eps={closest_point['epsilon']:.3f}"
        )
        closest_metrics_str = f"(MSE={closest_point['avg_mse']:.3f}, SSIM={closest_point['avg_ssim']:.3f})"
        logger.info(closest_params_str)
        logger.info(f"  Metrics: {closest_metrics_str}")
        candidates['epsilon_legend'] = candidates['epsilon'].round(2)
    else:
        logger.warning("No candidates found, cannot determine closest point.")

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8)) # Use fig object

    # 1. Density map
    sns.kdeplot(data=df_plot, x='avg_mse', y='avg_ssim', fill=True, cmap="Blues", alpha=0.5, ax=ax, warn_singular=False)

    # 2. Candidate points
    legend_handles = []
    legend_labels = []
    if not candidates.empty:
        scatter = sns.scatterplot(
            data=candidates, x='avg_mse', y='avg_ssim', hue='k',
            size='epsilon_legend',
            palette='viridis', sizes=(40, 250), alpha=0.8, legend='full', ax=ax
        )
        handles, labels = scatter.get_legend_handles_labels()
        legend_handles.extend(handles)
        legend_labels.extend(labels)
        if ax.legend_: ax.legend_.remove() # Remove default scatter legend
    else:
        logger.warning("No candidates to display in the target zone.")

    # 3. Threshold lines
    line_ssim = ax.axhline(y=ssim_threshold, color='red', linestyle='--', linewidth=1.5, label=f'SSIM Threshold ({ssim_threshold:.2f})')
    line_mse = ax.axvline(x=mse_threshold, color='blue', linestyle='--', linewidth=1.5, label=f'MSE Threshold ({mse_threshold})')
    legend_handles.extend([line_mse, line_ssim])
    legend_labels.extend([f'MSE Threshold ({mse_threshold})', f'SSIM Threshold ({ssim_threshold:.2f})'])

    # 4. Highlight closest point and annotate
    if closest_point is not None:
        star_marker = ax.scatter(
            closest_point['avg_mse'], closest_point['avg_ssim'],
            marker='*', s=300, color='red', edgecolor='black', zorder=10,
            label='Closest Point'
        )
        legend_handles.append(star_marker)
        legend_labels.append('Closest Point')
        annotation_text = (
            f"Closest:\n"
            f" k={closest_point['k']:.0f}\n"
            f" ratio={closest_point['n_component_ratio']:.3f}\n"
            f" eps={closest_point['epsilon']:.3f}\n"
            f" MSE={closest_point['avg_mse']:.3f}\n"
            f" SSIM={closest_point['avg_ssim']:.3f}"
        )
        ax.annotate(
            annotation_text,
            xy=(closest_point['avg_mse'], closest_point['avg_ssim']),
            xytext=(15, -15), textcoords='offset points',
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=1.0', fc='yellow', alpha=0.6),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2')
        )

    # 5. Labels, Grid, Centering
    ax.set_xlabel(f'Avg MSE (Threshold={mse_threshold})')
    ax.set_ylabel(f'Avg SSIM (Threshold={ssim_threshold:.2f})')
    ax.grid(True, linestyle=':', alpha=0.5)

    x_center = mse_threshold
    y_center = ssim_threshold
    x_lim_lower = max(0, x_center - DELTA_MSE_ZOOM)
    x_lim_upper = x_center + DELTA_MSE_ZOOM
    y_lim_lower = max(0, y_center - DELTA_SSIM_ZOOM)
    y_lim_upper = y_center + DELTA_SSIM_ZOOM

    ax.set_xlim(x_lim_lower, x_lim_upper)
    ax.set_ylim(y_lim_lower, y_lim_upper)
    logger.info(f"Plot centered on MSE={x_center}, SSIM={y_center}. X Limits=[{x_lim_lower:.0f}, {x_lim_upper:.0f}], Y Limits=[{y_lim_lower:.2f}, {y_lim_upper:.2f}]")

    ax.text(min(x_lim_upper * 0.98, ax.get_xlim()[1]), ssim_threshold, f' SSIM={ssim_threshold:.2f}', color='red', va='center', ha='right', backgroundcolor='white', alpha=0.7)
    ax.text(mse_threshold, min(y_lim_upper * 0.98, ax.get_ylim()[1]), f' MSE={mse_threshold} ', color='blue', va='top', ha='center', rotation=90, backgroundcolor='white', alpha=0.7)

    # 6. Final legend
    valid_legend_items = []
    processed_labels = set()
    for h, l in zip(legend_handles, legend_labels):
        if h is None: continue
        current_label = l
        # Keep unique labels, plus threshold and closest point labels
        if current_label not in processed_labels or any(thr_label in l for thr_label in ['Threshold', 'Closest']):
            valid_legend_items.append((h, current_label))
            processed_labels.add(current_label)

    if valid_legend_items:
        valid_handles, valid_labels = zip(*valid_legend_items)
        ax.legend(handles=valid_handles, labels=valid_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    else:
        ax.legend(title='Legend (Empty)', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Use fig object

    # --- Save plot to PDF (optional, can be kept) ---
    os.makedirs(output_dir, exist_ok=True)
    plot_filename_base = "density_candidates_ssim_vs_mse_centered_highlighted"
    plot_filename_pdf = os.path.join(output_dir, f"{plot_filename_base}.pdf")
    try:
        fig.savefig(plot_filename_pdf, format='pdf', bbox_inches='tight') # Use fig object
        logger.info(f"Density + Candidates plot saved as PDF: {os.path.abspath(plot_filename_pdf)}")
    except Exception as e:
        logger.error(f"Error saving Density + Candidates plot as PDF: {e}", exc_info=True)

    # --- Generate Base64 encoded PNG for return ---
    plot_html_embed = None
    try:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight') # Save to buffer as PNG
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        img_buffer.close()
        plot_html_embed = f"data:image/png;base64,{img_base64}"
        logger.info("Generated Base64 string for Density + Candidates plot.")
    except Exception as e:
        logger.error(f"Error generating Base64 for Density + Candidates plot: {e}", exc_info=True)

    plt.close(fig) # Close the figure associated with fig object
    return plot_html_embed # Return the base64 string (or None)


def plot_metric_trends_dual_axis(
    df: pd.DataFrame,
    param_x: str,
    ssim_threshold: float,
    mse_threshold: float,
    output_dir: str
) -> str | None: # Modified return type annotation
    """
    Generates a plot showing the average trend of SSIM (left axis, blue)
    and MSE (right axis, red) against the specified parameter (param_x).
    Includes threshold lines based on provided values.
    Saves the plot as PDF to the specified output directory.
    Returns a base64 encoded PNG image string for HTML embedding.

    Args:
        df: DataFrame with grid search results.
        param_x: Name of the parameter column for the x-axis.
        ssim_threshold: SSIM threshold value to display.
        mse_threshold: MSE threshold value to display.
        output_dir: Directory to save the plot PDF file.

    Returns:
        A base64 encoded string for embedding the plot in HTML (data URI),
        or None if plotting failed.
    """
    logger.info(f"Generating trend plot for parameter: {param_x}...")
    if df.empty:
        logger.warning(f"DataFrame is empty. Trend plot for '{param_x}' not generated.")
        return None
    if param_x not in df.columns:
        logger.error(f"Parameter '{param_x}' not found in DataFrame.")
        return None

    # Ensure metrics are numeric before grouping
    df_plot = df.copy()
    df_plot['avg_ssim'] = pd.to_numeric(df_plot['avg_ssim'], errors='coerce')
    df_plot['avg_mse'] = pd.to_numeric(df_plot['avg_mse'], errors='coerce')
    df_plot = df_plot.dropna(subset=['avg_ssim', 'avg_mse', param_x])

    trends = df_plot.groupby(param_x)[['avg_ssim', 'avg_mse']].mean().reset_index()

    if trends.empty:
        logger.warning(f"No valid aggregated data for trend plot '{param_x}'.")
        return None

    # Create figure and axes explicitly
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_ssim = 'tab:blue'
    ax1.set_xlabel(param_x.replace('_', ' ').title())
    ax1.set_ylabel('Avg SSIM', color=color_ssim)
    line1 = ax1.plot(trends[param_x], trends['avg_ssim'], color=color_ssim, marker='o', linestyle='-', label='Avg SSIM')
    ax1.tick_params(axis='y', labelcolor=color_ssim)
    ax1.axhline(y=ssim_threshold, color=color_ssim, linestyle=':', linewidth=2, label=f'SSIM Threshold ({ssim_threshold:.2f})')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color_mse = 'tab:red'
    ax2.set_ylabel('Avg MSE', color=color_mse)
    line2 = ax2.plot(trends[param_x], trends['avg_mse'], color=color_mse, marker='s', linestyle='--', label='Avg MSE')
    ax2.tick_params(axis='y', labelcolor=color_mse)
    ax2.axhline(y=mse_threshold, color=color_mse, linestyle=':', linewidth=2, label=f'MSE Threshold ({mse_threshold})')
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
    ax2.ticklabel_format(style='plain', axis='y')

    plt.title(f'Avg SSIM & MSE Trend vs {param_x.replace("_", " ").title()}')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # Correctly fetch threshold lines for legend
    handles_thresholds = []
    labels_thresholds = []
    if len(ax1.get_lines()) > 1:
        handles_thresholds.append(ax1.get_lines()[1])
        labels_thresholds.append(ax1.get_lines()[1].get_label())
    if len(ax2.get_lines()) > 1:
        handles_thresholds.append(ax2.get_lines()[1])
        labels_thresholds.append(ax2.get_lines()[1].get_label())

    ax1.legend(lines + handles_thresholds, labels + labels_thresholds, loc='best')
    fig.tight_layout() # Use fig object

    # --- Save plot to PDF (optional, can be kept) ---
    os.makedirs(output_dir, exist_ok=True)
    plot_filename_base = f"trend_ssim_mse_vs_{param_x}"
    plot_filename_pdf = os.path.join(output_dir, f"{plot_filename_base}.pdf")
    try:
        fig.savefig(plot_filename_pdf, format='pdf', bbox_inches='tight') # Use fig object
        logger.info(f"Trend plot for '{param_x}' saved as PDF: {os.path.abspath(plot_filename_pdf)}")
    except Exception as e:
        logger.error(f"Error saving trend plot for '{param_x}' as PDF: {e}", exc_info=True)

    # --- Generate Base64 encoded PNG for return ---
    plot_html_embed = None
    try:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight') # Save to buffer as PNG
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        img_buffer.close()
        plot_html_embed = f"data:image/png;base64,{img_base64}"
        logger.info(f"Generated Base64 string for trend plot '{param_x}'.")
    except Exception as e:
        logger.error(f"Error generating Base64 for trend plot '{param_x}': {e}", exc_info=True)

    plt.close(fig) # Close the figure associated with fig object
    return plot_html_embed # Return the base64 string (or None)

# --- Filter function remains unchanged ---
def filter_and_display_candidates(
    df: pd.DataFrame,
    ssim_threshold: float,
    mse_threshold: float
) -> pd.DataFrame:
    """
    Filters the DataFrame to find candidates meeting the thresholds
    and prints a sorted summary.

    Args:
        df: DataFrame with grid search results.
        ssim_threshold: Maximum SSIM value for candidate selection.
        mse_threshold: Maximum MSE value for candidate selection.

    Returns:
        DataFrame containing only the candidate rows, sorted. Empty if no candidates.
    """
    logger.info(f"Filtering candidates meeting thresholds (SSIM<{ssim_threshold}, MSE<{mse_threshold})...")
    if df.empty:
        logger.warning("DataFrame is empty. Cannot filter.")
        return pd.DataFrame()

    # Ensure metrics are numeric for filtering
    df_filtered = df.copy()
    df_filtered['avg_ssim'] = pd.to_numeric(df_filtered['avg_ssim'], errors='coerce')
    df_filtered['avg_mse'] = pd.to_numeric(df_filtered['avg_mse'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['avg_ssim', 'avg_mse'])

    # Calculate distance if not present
    if 'distance_to_target' not in df_filtered.columns and not df_filtered.empty:
         logger.debug("Calculating 'distance_to_target' for candidate sorting.")
         # Check required columns exist before calculating distance
         if 'avg_mse' in df_filtered.columns and 'avg_ssim' in df_filtered.columns:
            df_filtered['distance_to_target'] = df_filtered.apply(
                lambda row: math.sqrt(
                    ((row['avg_mse'] - mse_threshold) ** 2) +
                    ((row['avg_ssim'] - ssim_threshold) ** 2)
                ) if pd.notna(row['avg_mse']) and pd.notna(row['avg_ssim']) else float('inf'), # Handle potential NaNs just in case
                axis=1
            )
         else:
             logger.warning("Cannot calculate 'distance_to_target', missing 'avg_mse' or 'avg_ssim'.")
             # Assign a dummy value or handle appropriately if sorting is critical
             df_filtered['distance_to_target'] = float('inf')


    # Filter using provided thresholds
    candidates = df_filtered[
        (df_filtered['avg_ssim'] < ssim_threshold) &
        (df_filtered['avg_mse'] < mse_threshold)
    ].copy()

    if candidates.empty:
        logger.warning(f"No parameter combinations found meeting thresholds.")
        return pd.DataFrame()
    else:
        logger.info(f"Number of candidate combinations found: {len(candidates)}")
        # Sort candidates (ensure distance_to_target exists before sorting by it)
        sort_columns = ['avg_ssim', 'avg_mse']
        if 'distance_to_target' in candidates.columns:
            sort_columns.insert(0, 'distance_to_target')

        candidates_sorted = candidates.sort_values(by=sort_columns, ascending=[True]*len(sort_columns))


        logger.info(f"\n--- Top {min(15, len(candidates_sorted))} Candidates (sorted by distance, SSIM, then MSE) ---")
        with pd.option_context('display.max_rows', 15, 'display.max_columns', None, 'display.width', 1000):
            # Define formatters
            formatters = {
                'n_component_ratio': '{:.3f}'.format,
                'epsilon': '{:.3f}'.format,
                'avg_mse': '{:.3f}'.format,
                'avg_ssim': '{:.3f}'.format,
            }
            if 'distance_to_target' in candidates_sorted.columns:
                 formatters['distance_to_target'] = '{:.4f}'.format

            # Define columns to display, ensuring they exist
            display_cols_base = ['k', 'n_component_ratio', 'epsilon', 'avg_mse', 'avg_ssim']
            if 'distance_to_target' in candidates_sorted.columns:
                display_cols_base.append('distance_to_target')
            display_cols = [col for col in display_cols_base if col in candidates_sorted.columns]

            print(candidates_sorted[display_cols].to_string(formatters=formatters)) # Apply formatting only to existing columns

        logger.info("--- End of candidate list ---")
        return candidates_sorted


# --- MODULAR Main Execution Logic (Modified to capture return values) ---
def main_visualize(
    input_csv_path: str,
    output_dir: str,
    ssim_threshold: float = DEFAULT_SSIM_THRESHOLD,
    mse_threshold: float = DEFAULT_MSE_THRESHOLD
):
    """
    Main function to load data, generate visualizations (returning embeddable strings),
    and analyze candidates.

    Args:
        input_csv_path: Path to the input CSV file.
        output_dir: Path to the directory where PDF plots will be saved.
        ssim_threshold: Maximum SSIM value.
        mse_threshold: Maximum MSE value.
    """
    logger.info(f"--- STARTING MODULAR VISUALIZATION SCRIPT ---")
    logger.info(f"Input CSV: {os.path.abspath(input_csv_path)}")
    logger.info(f"Output Directory: {os.path.abspath(output_dir)}")
    logger.info(f"Thresholds: SSIM < {ssim_threshold}, MSE < {mse_threshold}")

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory '{output_dir}': {e}")
        return

    # 1. Load data
    try:
        logger.info(f"Loading results from: {input_csv_path}")
        df_results = pd.read_csv(input_csv_path)
        logger.info(f"Data loaded successfully. Initial shape: {df_results.shape}")

        required_cols = ['k', 'n_component_ratio', 'epsilon', 'avg_mse', 'avg_ssim']
        missing_cols = [col for col in required_cols if col not in df_results.columns]
        if missing_cols:
            logger.error(f"Missing required columns in CSV: {missing_cols}. Expected: {required_cols}.")
            return

        logger.info("NaN count per required column before cleaning:\n" + str(df_results[required_cols].isnull().sum()))
        initial_rows = len(df_results)
        df_results.dropna(subset=required_cols, inplace=True)
        dropped_rows = initial_rows - len(df_results)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows due to NaN values in required columns.")
        logger.info(f"Shape after dropping rows with NaNs in required columns: {df_results.shape}")

        if df_results.empty:
            logger.error("DataFrame is empty after removing NaN values. Cannot proceed.")
            return

    except FileNotFoundError:
        logger.error(f"Input CSV file not found: {input_csv_path}")
        return
    except Exception as e:
        logger.error(f"Error loading or processing CSV file '{input_csv_path}': {e}", exc_info=True)
        return

    # --- Store returned plot data ---
    plot_data_html = {} # Dictionary to hold the base64 strings

    # 2. Generate Density + Candidates plot (capture return value)
    density_plot_html = plot_ssim_vs_mse_density_candidates(
        df_results,
        ssim_threshold=ssim_threshold,
        mse_threshold=mse_threshold,
        output_dir=output_dir
    )
    if density_plot_html:
        plot_data_html['density_candidates'] = density_plot_html
        logger.info("Captured Base64 string for Density + Candidates plot.")
        # Example: print(f"Density Plot HTML Embed:\n<img src=\"{density_plot_html[:100]}...\">\n") # Print snippet for verification
    else:
        logger.warning("Density + Candidates plot generation failed or returned None.")


    # 3. Generate Average Trend plots (capture return values)
    params_to_plot_trends = ['k', 'n_component_ratio', 'epsilon']
    for param in params_to_plot_trends:
        if param in df_results.columns:
            trend_plot_html = plot_metric_trends_dual_axis(
                df_results,
                param,
                ssim_threshold=ssim_threshold,
                mse_threshold=mse_threshold,
                output_dir=output_dir
            )
            if trend_plot_html:
                 plot_data_html[f'trend_{param}'] = trend_plot_html
                 logger.info(f"Captured Base64 string for trend plot '{param}'.")
                 # Example: print(f"Trend {param} Plot HTML Embed:\n<img src=\"{trend_plot_html[:100]}...\">\n") # Print snippet
            else:
                 logger.warning(f"Trend plot generation for '{param}' failed or returned None.")
        else:
            logger.warning(f"Column '{param}' not found for trend plotting.")

    # 4. Filter and display promising candidates
    logger.info("\n=== Candidate Analysis ===")
    candidate_df = filter_and_display_candidates( # Capture the DataFrame if needed later
        df_results,
        ssim_threshold=ssim_threshold,
        mse_threshold=mse_threshold
    )

    logger.info("--- MODULAR VISUALIZATION SCRIPT FINISHED ---")
    logger.info(f"PDF plots (if saved successfully) are in: {os.path.abspath(output_dir)}")
    logger.info(f"Base64 encoded plot strings captured: {list(plot_data_html.keys())}")
    logger.info("These strings can be used in an HTML GUI (e.g., <img src='...'/>).")

    # Optionally return the captured data if this function is called by another module/GUI backend
    # return plot_data_html, candidate_df


if __name__ == "__main__":
    # --- Configuration for this specific run ---
    csv_location = os.path.join("analysis_results_grid_search_modular", "grid_search_results.csv")
    results_save_directory = "plots_analysis_output"
    ssim_limit = 0.50
    mse_limit = 1800

    # Call the main function
    main_visualize(
        input_csv_path=csv_location,
        output_dir=results_save_directory,
        ssim_threshold=ssim_limit,
        mse_threshold=mse_limit
    )