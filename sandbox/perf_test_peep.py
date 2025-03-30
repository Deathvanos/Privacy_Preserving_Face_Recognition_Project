import os
import pickle
from matplotlib.gridspec import GridSpec

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from src.modules.image_preprocessing import preprocess_image
from src.modules.peep import Peep
from src.config import IMAGE_SIZE
from src.modules.noise_generator import NoiseGenerator
from src.modules.utils_image import image_numpy_to_pillow


def calculate_mse(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE) between two images.

    MSE represents the average squared difference between pixel values, measuring
    the absolute difference between the original and reconstructed images.
    Lower values indicate better image reconstruction quality.

    Args:
        imageA: First image (original)
        imageB: Second image (reconstructed)

    Returns:
        float: MSE value
    """
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def normalize_for_display(eigenface: np.ndarray) -> np.ndarray:
    """
    Normalizes an eigenface for better visualization.
    This ensures eigenfaces with both positive and negative values are properly displayed.
    """
    # Find the absolute maximum value for centered normalization
    abs_max = np.max(np.abs(eigenface))
    if abs_max < 1e-10:  # Avoid division by zero
        return np.zeros_like(eigenface)

    # Normalize so values are between -1 and 1
    normalized = eigenface / abs_max

    # Transform from [-1, 1] to [0, 1] for display
    display_image = (normalized + 1) / 2
    return display_image


def perf_test(image_folder: str, output_folder: str, epsilon_values: list, n_components_ratios: list,
              target_subject=None, image_size=IMAGE_SIZE) -> dict:
    """
    Performs performance tests comparing noise on eigenfaces vs. noise on projection vectors.
    Generates comparison plots for the experimental results section of the paper.

    This function implements the experiment described in Section V of the paper.
    Images are resized to the dimensions specified in IMAGE_SIZE (typically 100x100 pixels)
    and normalized to values between 0 and 1 before processing.

    The function analyzes the trade-off between privacy (controlled by epsilon) and
    utility (measured by MSE and SSIM) across different PCA component ratios.

    Args:
        image_folder: Path to folder with images (YaleFace dataset)
        output_folder: Path for saving results
        epsilon_values: List of epsilon values to test (e.g., 0.5, 0.75, 1.0)
        n_components_ratios: List of PCA component ratios to test (e.g., 0.55, 0.7, 0.85, 1.0)
        target_subject: If provided, only visualize this subject but compute metrics for all
        image_size: Size to which images are resized before processing

    Returns:
        Dictionary containing results for both methods (eigenface noise and projection noise)
    """

    if not os.path.isdir(image_folder):
        raise ValueError(f"The provided image folder '{image_folder}' is not a directory.")

    # Create output folders for essential visualizations
    comparison_folder = os.path.join(output_folder, "method_comparison")
    method_folder = os.path.join(output_folder, "noise_methods")
    os.makedirs(comparison_folder, exist_ok=True)
    os.makedirs(method_folder, exist_ok=True)

    # Results dictionaries for both methods
    results_eigenface = {}  # Noise on eigenfaces
    results_projection = {}  # Noise on projection vectors
    errors = []

    # Organize data by subject first to avoid reprocessing
    subject_images = {}
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        try:
            parts = filename.split("_")
            if len(parts) < 4:
                raise ValueError(f"Filename '{filename}' has an invalid format.")
            subject_number = int(parts[1])
            image_path = os.path.join(image_folder, filename)

            with Image.open(image_path) as img:
                # This function resizes the image to image_size (100x100 pixels by default)
                # and creates a normalized, flattened version for further processing
                processed_data = preprocess_image(img, resize_size=image_size, create_flattened=True)
                if processed_data and processed_data['flattened_image'] is not None:
                    if subject_number not in subject_images:
                        subject_images[subject_number] = []
                    subject_images[subject_number].append(
                        (processed_data['flattened_image'], processed_data['resized_image']))
                else:
                    errors.append(f"Skipping {filename} due to preprocessing error.")

        except (IOError, OSError, ValueError) as e:
            errors.append(f"Error processing {filename}: {e}")
            continue

    print(f"Found data for {len(subject_images)} subjects")
    if target_subject is not None:
        if target_subject not in subject_images:
            print(
                f"Warning: Target subject {target_subject} not found in data. Available subjects: {list(subject_images.keys())}")
            if subject_images:
                target_subject = list(subject_images.keys())[0]
                print(f"Using first available subject: {target_subject}")
        else:
            print(f"Will visualize subject {target_subject} only")

    for epsilon in tqdm(epsilon_values, desc="Epsilon values"):
        results_eigenface[epsilon] = {}
        results_projection[epsilon] = {}

        for n_components_ratio in tqdm(n_components_ratios, desc="n_components ratios", leave=False):
            results_eigenface[epsilon][n_components_ratio] = {'avg_mse': 0, 'avg_ssim': 0, 'example_images': {}}
            results_projection[epsilon][n_components_ratio] = {'avg_mse': 0, 'avg_ssim': 0, 'example_images': {}}

            mse_values_eigenface = []
            ssim_values_eigenface = []
            mse_values_projection = []
            ssim_values_projection = []

            for subject, images_data in subject_images.items():
                if not images_data:
                    continue

                flattened_images, resized_images = zip(*images_data)
                flattened_images_array = np.array(flattened_images)
                # Ensure we don't exceed the maximum number of components
                max_possible_components = min(flattened_images_array.shape[0] - 1, flattened_images_array.shape[1])
                n_components = int(n_components_ratio * flattened_images_array.shape[0])
                n_components = max(1, min(n_components, max_possible_components))

                try:
                    # Create and run the Peep object
                    peep = Peep(epsilon=epsilon, image_size=image_size)
                    peep.run(flattened_images_array, method='bounded', n_components=n_components)

                    # Get original eigenfaces and mean face
                    original_eigenfaces = peep.get_eigenfaces(format='numpy')

                    # Improved eigenface visualization
                    original_eigenfaces_pil = []
                    for eigenface in original_eigenfaces:
                        # Normalize for better display
                        display_eigenface = normalize_for_display(eigenface)
                        original_eigenfaces_pil.append(image_numpy_to_pillow(display_eigenface))
                    mean_face = peep.get_mean_face()

                    # ==========================================================
                    # METHOD 1: Adding noise to eigenfaces
                    # ==========================================================
                    # Create copies of eigenfaces to add noise
                    noised_eigenfaces = []
                    noised_eigenfaces_pil = []

                    for i in range(len(original_eigenfaces)):
                        eigenface = original_eigenfaces[i].copy()

                        # Apply noise directly to eigenface
                        noise_gen = NoiseGenerator(eigenface.reshape(1, -1), epsilon)
                        noise_gen.normalize_images()
                        noise_gen.add_laplace_noise(peep.sensitivity * 0.4)  # Reduced sensitivity for eigenfaces
                        noised_eigenface = noise_gen.get_noised_eigenfaces()

                        # Reshape to image dimensions
                        noised_eigenface_reshaped = noised_eigenface.reshape(image_size)
                        noised_eigenfaces.append(noised_eigenface_reshaped)

                        # Create PIL version for visualization with improved normalization
                        display_noised_eigenface = normalize_for_display(noised_eigenface_reshaped)
                        noised_eigenfaces_pil.append(image_numpy_to_pillow(display_noised_eigenface))

                    # Reconstruct images using noised eigenfaces
                    # This requires modifying the PCA components in the PCA object
                    temp_pca = peep.pca_object.pca
                    original_components = temp_pca.components_.copy()

                    # Replace components with noised eigenfaces (flattened)
                    for i in range(len(noised_eigenfaces)):
                        temp_pca.components_[i] = noised_eigenfaces[i].flatten()

                    # Reconstruct images with noised eigenfaces
                    reconstructed_with_noised_eigenfaces = temp_pca.inverse_transform(peep.projected_vectors)

                    # Convert to PIL images for visualization
                    reconstructed_eigenface_pil = []
                    for img in reconstructed_with_noised_eigenfaces:
                        img_reshaped = img.reshape(image_size)
                        img_norm = np.clip((img_reshaped - np.min(img_reshaped)) / (np.max(img_reshaped) - np.min(img_reshaped) + 1e-10), 0, 1)
                        reconstructed_eigenface_pil.append(image_numpy_to_pillow(img_norm))
                    # Restore original components
                    temp_pca.components_ = original_components

                    # ==========================================================
                    # METHOD 2: Adding noise to projection vectors
                    # ==========================================================
                    # Get original projection vectors
                    proj_vectors = peep.projected_vectors.copy()

                    # Store min/max for denormalization later
                    min_vals = np.min(proj_vectors, axis=0)
                    max_vals = np.max(proj_vectors, axis=0)

                    # Apply noise to projection vectors
                    noise_generator = NoiseGenerator(proj_vectors, epsilon)
                    noise_generator.normalize_images()
                    noise_generator.add_laplace_noise(peep.sensitivity * 0.5)  # Sensitivity for projections
                    noised_projections = noise_generator.get_noised_eigenfaces()

                    # Denormalize to restore original scale
                    for i in range(noised_projections.shape[1]):
                        if max_vals[i] - min_vals[i] > 0:  # Avoid division by zero
                            noised_projections[:, i] = noised_projections[:, i] * (max_vals[i] - min_vals[i]) + min_vals[i]

                    # Reconstruct images from noised projections
                    reconstructed_with_noised_projections = peep.pca_object.reconstruct_image(noised_projections)

                    # Convert to PIL images for visualization
                    reconstructed_projection_pil = []
                    for img in reconstructed_with_noised_projections:
                        img_reshaped = img.reshape(image_size)
                        img_norm = np.clip((img_reshaped - np.min(img_reshaped)) / (np.max(img_reshaped) - np.min(img_reshaped) + 1e-10), 0, 1)
                        reconstructed_projection_pil.append(image_numpy_to_pillow(img_norm))

                    # ==========================================================
                    # Store example images for visualization of each step
                    # But only for the target subject (if specified)
                    # ==========================================================
                    should_visualize = (target_subject is None) or (subject == target_subject)

                    if should_visualize and subject not in results_eigenface[epsilon][n_components_ratio]['example_images']:
                        results_eigenface[epsilon][n_components_ratio]['example_images'][subject] = {
                            'original': resized_images[0],
                            'eigenface': original_eigenfaces_pil[0] if original_eigenfaces_pil else None,
                            'noised_eigenface': noised_eigenfaces_pil[0] if noised_eigenfaces_pil else None,
                            'reconstructed': reconstructed_eigenface_pil[0],
                            'proj_vectors': proj_vectors[0],
                            'mean_face': mean_face
                        }

                        results_projection[epsilon][n_components_ratio]['example_images'][subject] = {
                            'original': resized_images[0],
                            'eigenface': original_eigenfaces_pil[0] if original_eigenfaces_pil else None,
                            'proj_vectors': proj_vectors[0],
                            'noised_proj_vectors': noised_projections[0],
                            'reconstructed': reconstructed_projection_pil[0],
                            'mean_face': mean_face
                        }

                    for i in range(len(resized_images)):
                        original_np = np.array(resized_images[i]).astype(float) / 255.0

                        # Method 1: Eigenface noise
                        reconstructed_eigenface_np = np.array(reconstructed_eigenface_pil[i]).astype(float) / 255.0
                        mse_eigenface = calculate_mse(original_np, reconstructed_eigenface_np)
                        ssim_eigenface = ssim(original_np, reconstructed_eigenface_np, data_range=1.0)
                        mse_values_eigenface.append(mse_eigenface)
                        ssim_values_eigenface.append(ssim_eigenface)

                        # Method 2: Projection vector noise
                        reconstructed_proj_np = np.array(reconstructed_projection_pil[i]).astype(float) / 255.0
                        mse_proj = calculate_mse(original_np, reconstructed_proj_np)
                        ssim_proj = ssim(original_np, reconstructed_proj_np, data_range=1.0)
                        mse_values_projection.append(mse_proj)
                        ssim_values_projection.append(ssim_proj)

                except Exception as e:
                    errors.append(f"Error processing subject {subject} with epsilon {epsilon}, n_components_ratio {n_components_ratio}: {e}")
                    continue

            results_eigenface[epsilon][n_components_ratio]['avg_mse'] = np.mean(mse_values_eigenface) if mse_values_eigenface else float('nan')
            results_eigenface[epsilon][n_components_ratio]['avg_ssim'] = np.mean(ssim_values_eigenface) if ssim_values_eigenface else float('nan')
            results_projection[epsilon][n_components_ratio]['avg_mse'] = np.mean(mse_values_projection) if mse_values_projection else float('nan')
            results_projection[epsilon][n_components_ratio]['avg_ssim'] = np.mean(ssim_values_projection) if ssim_values_projection else float('nan')

    # ==========================================================
    # Generate essential comparison visualizations
    # ==========================================================
    for n_components_ratio in n_components_ratios:
        epsilon_values_plot = sorted(epsilon_values)

        plt.figure(figsize=(10, 6))
        avg_mse_eigenface = [results_eigenface[epsilon][n_components_ratio]['avg_mse'] for epsilon in epsilon_values_plot]
        avg_mse_projection = [results_projection[epsilon][n_components_ratio]['avg_mse'] for epsilon in epsilon_values_plot]
        plt.plot(epsilon_values_plot, avg_mse_eigenface, 'r-o', label='Noise on Eigenfaces')
        plt.plot(epsilon_values_plot, avg_mse_projection, 'b-s', label='Noise on Projection Vectors')
        plt.xlabel("Epsilon (ε)")
        plt.ylabel("Average MSE")
        plt.title(f"MSE Comparison - n_components_ratio={n_components_ratio:.2f}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(comparison_folder, f"mse_comparison_n{n_components_ratio:.2f}.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        avg_ssim_eigenface = [results_eigenface[epsilon][n_components_ratio]['avg_ssim'] for epsilon in epsilon_values_plot]
        avg_ssim_projection = [results_projection[epsilon][n_components_ratio]['avg_ssim'] for epsilon in epsilon_values_plot]
        plt.plot(epsilon_values_plot, avg_ssim_eigenface, 'r-o', label='Noise on Eigenfaces')
        plt.plot(epsilon_values_plot, avg_ssim_projection, 'b-s', label='Noise on Projection Vectors')
        plt.xlabel("Epsilon (ε)")
        plt.ylabel("Average SSIM")
        plt.title(f"SSIM Comparison - n_components_ratio={n_components_ratio:.2f}")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(comparison_folder, f"ssim_comparison_n{n_components_ratio:.2f}.png"))
        plt.close()

    # 2. ESSENTIAL: Visual comparison of noised images for different epsilon values
    # Find a subject with data across all combinations
    target_subject_found = False
    for subject in subject_images.keys():
        if target_subject is not None and subject != target_subject:
            continue

        has_all_data = True
        for epsilon in epsilon_values:
            for n_ratio in n_components_ratios:
                if subject not in results_eigenface[epsilon][n_ratio]['example_images'] or \
                        subject not in results_projection[epsilon][n_ratio]['example_images']:
                    has_all_data = False
                    break

        if has_all_data:
            target_subject = subject
            target_subject_found = True
            break

    if not target_subject_found:
        print("Warning: No subject has data for all epsilon and n_components combinations")
        return {"eigenface": results_eigenface, "projection": results_projection}

    # 3. ESSENTIAL: Create separate comparative visualizations for each epsilon value
    # to better show the deformation process of eigenfaces and projections

    # Create a dedicated folder for the detailed visualizations
    detailed_viz_folder = os.path.join(output_folder, "detailed_noise_effects")
    os.makedirs(detailed_viz_folder, exist_ok=True)

    for n_components_ratio in n_components_ratios:
        for i, epsilon in enumerate(sorted(epsilon_values)):
            if (target_subject in results_eigenface[epsilon][n_components_ratio]['example_images'] and
                    target_subject in results_projection[epsilon][n_components_ratio]['example_images']):
                # ======= SOLUTION 1: CORRECTION POUR L'ERREUR TIGHT_LAYOUT =======
                # METHOD 1: Eigenface Noise Visualization
                fig_ef = plt.figure(figsize=(16, 4))

                # Créer un layout de GridSpec pour mieux contrôler les positions des sous-graphiques
                from matplotlib.gridspec import GridSpec
                gs = GridSpec(1, 4, figure=fig_ef, width_ratios=[1, 1, 1, 1])

                # Gather data for METHOD 1 (eigenface noise)
                original_img = results_eigenface[epsilon][n_components_ratio]['example_images'][target_subject]['original']
                original_eigenface = results_eigenface[epsilon][n_components_ratio]['example_images'][target_subject]['eigenface']
                noised_eigenface = results_eigenface[epsilon][n_components_ratio]['example_images'][target_subject]['noised_eigenface']
                reconstructed_eigenface = results_eigenface[epsilon][n_components_ratio]['example_images'][target_subject]['reconstructed']

                # Créer des sous-graphiques avec GridSpec
                ax1 = fig_ef.add_subplot(gs[0, 0])
                ax2 = fig_ef.add_subplot(gs[0, 1])
                ax3 = fig_ef.add_subplot(gs[0, 2])
                ax4 = fig_ef.add_subplot(gs[0, 3])

                # Display original image
                ax1.imshow(original_img, cmap='gray')
                ax1.set_title("Original Image (100x100)")
                ax1.axis('off')

                # Show the original eigenface
                ax2.imshow(original_eigenface, cmap='gray')
                ax2.set_title("Original Eigenface")
                ax2.axis('off')

                # Show the noised eigenface
                ax3.imshow(noised_eigenface, cmap='gray')
                ax3.set_title("Noised Eigenface")
                ax3.axis('off')

                # Show the reconstructed image
                ax4.imshow(reconstructed_eigenface, cmap='gray')
                ax4.set_title("Reconstructed Image")
                ax4.axis('off')

                plt.suptitle(f"Method 1: Eigenface Noise (ε={epsilon}, n_ratio={n_components_ratio:.2f})", fontsize=16)
                fig_ef.tight_layout()
                plt.savefig(os.path.join(detailed_viz_folder, f"eigenface_noise_e{epsilon}_n{n_components_ratio:.2f}.png"))
                plt.close(fig_ef)

                # METHOD 2: Projection Vector Noise Visualization - AMÉLIORÉ POUR LA VISUALISATION
                fig_pv = plt.figure(figsize=(16, 4))

                # Utiliser GridSpec pour mieux contrôler la disposition
                gs_pv = GridSpec(1, 4, figure=fig_pv, width_ratios=[1, 1, 1, 1])

                # Gather data for METHOD 2 (projection vector noise)
                original_img = results_projection[epsilon][n_components_ratio]['example_images'][target_subject]['original']
                eigenface = results_projection[epsilon][n_components_ratio]['example_images'][target_subject]['eigenface']
                proj_vectors_original = results_projection[epsilon][n_components_ratio]['example_images'][target_subject]['proj_vectors']
                noised_proj_vectors = results_projection[epsilon][n_components_ratio]['example_images'][target_subject]['noised_proj_vectors']
                reconstructed_projection = results_projection[epsilon][n_components_ratio]['example_images'][target_subject]['reconstructed']

                # Créer des sous-graphiques avec GridSpec
                ax1_pv = fig_pv.add_subplot(gs_pv[0, 0])
                ax2_pv = fig_pv.add_subplot(gs_pv[0, 1])
                ax3_pv = fig_pv.add_subplot(gs_pv[0, 2])
                ax4_pv = fig_pv.add_subplot(gs_pv[0, 3])

                # Original Image
                ax1_pv.imshow(original_img, cmap='gray')
                ax1_pv.set_title("Original Image (100x100)")
                ax1_pv.axis('off')

                # ====== AMÉLIORATION DE LA VISUALISATION DES VECTEURS DE PROJECTION =======
                # Afficher l'eigenface original
                ax2_pv.imshow(eigenface, cmap='gray')
                ax2_pv.set_title("Original Eigenface")
                ax2_pv.axis('off')

                # Créer une visualisation améliorée pour les vecteurs de projection
                # Limiter à 10 dimensions pour une meilleure lisibilité
                max_dims = min(10, len(proj_vectors_original))
                index = np.arange(max_dims)
                bar_width = 0.35

                # Normaliser les vecteurs pour une meilleure visualisation
                orig_norm = proj_vectors_original[:max_dims] / (np.max(np.abs(proj_vectors_original[:max_dims])) + 1e-10)
                noised_norm = noised_proj_vectors[:max_dims] / (np.max(np.abs(noised_proj_vectors[:max_dims])) + 1e-10)

                ax3_pv.bar(index, orig_norm, bar_width,
                           label='Original', color='blue', alpha=0.7)
                ax3_pv.bar(index + bar_width, noised_norm, bar_width,
                           label='Noised', color='red', alpha=0.7)

                ax3_pv.set_xlabel('Dimension')
                ax3_pv.set_title('Projection Vectors\nComparison')
                ax3_pv.set_xticks(index + bar_width / 2)
                ax3_pv.set_xticklabels([f'{i + 1}' for i in range(max_dims)])
                ax3_pv.legend(loc='upper right', fontsize='small')
                ax3_pv.grid(alpha=0.3)

                # Afficher l'image reconstruite
                ax4_pv.imshow(reconstructed_projection, cmap='gray')
                ax4_pv.set_title("Reconstructed Image")
                ax4_pv.axis('off')

                plt.suptitle(f"Method 2: Projection Vector Noise (ε={epsilon}, n_ratio={n_components_ratio:.2f})",
                             fontsize=16)
                fig_pv.tight_layout()
                plt.savefig(os.path.join(detailed_viz_folder, f"projection_noise_e{epsilon}_n{n_components_ratio:.2f}.png"))
                plt.close(fig_pv)

                # COMPARISON: Direct comparison of both methods for this epsilon and ratio
                fig_comp = plt.figure(figsize=(16, 8))
                gs_comp = GridSpec(2, 4, figure=fig_comp)

                # Row 1: Method 1 (Eigenface Noise)
                ax1_comp = fig_comp.add_subplot(gs_comp[0, 0])
                ax2_comp = fig_comp.add_subplot(gs_comp[0, 1])
                ax3_comp = fig_comp.add_subplot(gs_comp[0, 2])
                ax4_comp = fig_comp.add_subplot(gs_comp[0, 3])

                ax1_comp.imshow(original_img, cmap='gray')
                ax1_comp.set_title("Original Image")
                ax1_comp.axis('off')

                ax2_comp.imshow(original_eigenface, cmap='gray')
                ax2_comp.set_title("Original Eigenface")
                ax2_comp.axis('off')

                ax3_comp.imshow(noised_eigenface, cmap='gray')
                ax3_comp.set_title("Noised Eigenface")
                ax3_comp.axis('off')

                ax4_comp.imshow(reconstructed_eigenface, cmap='gray')
                ax4_comp.set_title("Reconstructed (Method 1)")
                ax4_comp.axis('off')

                # Row 2: Method 2 (Projection Vector Noise)
                ax5_comp = fig_comp.add_subplot(gs_comp[1, 0])
                ax6_comp = fig_comp.add_subplot(gs_comp[1, 1])
                ax7_comp = fig_comp.add_subplot(gs_comp[1, 2])
                ax8_comp = fig_comp.add_subplot(gs_comp[1, 3])

                ax5_comp.imshow(original_img, cmap='gray')
                ax5_comp.set_title("Original Image")
                ax5_comp.axis('off')

                # Visualisation améliorée des vecteurs de projection
                max_dims = min(10, len(proj_vectors_original))
                index = np.arange(max_dims)

                # Premier graphique - vecteurs originaux
                ax6_comp.bar(index, proj_vectors_original[:max_dims], color='blue', alpha=0.7)
                ax6_comp.set_title("Original Projection")
                ax6_comp.set_xticks([])
                ax6_comp.set_yticks([])

                # Deuxième graphique - vecteurs bruités
                ax7_comp.bar(index, noised_proj_vectors[:max_dims], color='red', alpha=0.7)
                ax7_comp.set_title("Noised Projection")
                ax7_comp.set_xticks([])
                ax7_comp.set_yticks([])

                # Image reconstruite
                ax8_comp.imshow(reconstructed_projection, cmap='gray')
                ax8_comp.set_title("Reconstructed (Method 2)")
                ax8_comp.axis('off')

                plt.suptitle(f"Comparison of Noise Application Methods (ε={epsilon}, n_ratio={n_components_ratio:.2f})", fontsize=16)
                fig_comp.tight_layout()
                plt.savefig(os.path.join(detailed_viz_folder, f"methods_comparison_e{epsilon}_n{n_components_ratio:.2f}.png"))
                plt.close(fig_comp)
        # End loop for detailed visualization

    return {"eigenface": results_eigenface, "projection": results_projection}


if __name__ == "__main__":
    output = perf_test(image_folder="../data/database", output_folder="perf_results",
                       epsilon_values=[0.5, 1, 2, 4, 8],
                       n_components_ratios=[0.55, 0.7, 0.85, 1.0],
                       target_subject=None, image_size=IMAGE_SIZE)
    print("Tests de performance terminés.")
