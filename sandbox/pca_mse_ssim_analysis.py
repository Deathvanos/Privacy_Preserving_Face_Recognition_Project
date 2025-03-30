import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.filters import sobel
from src.modules.eigenface import EigenfaceGenerator
from src.modules.utils_image import load_images, calculate_mse

# Configuration
DATASET_FOLDER = "../data/database"  # Update with the correct path
IMAGE_SIZE = (100, 100)  # Target image size
RATIOS = [0.4, 0.55, 0.7, 0.85, 1]  # Ratios for PCA component selection
OUTPUT_FOLDER = "output_graphs"  # Folder to save images
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def preprocess_images(image_folder):
    images = load_images(image_folder)
    processed_images = [np.array(img.resize(IMAGE_SIZE).convert("L")).flatten() for img in images]
    return np.array(processed_images)


def compute_metrics(original_images, reconstructed_images):
    mse_values, ssim_values = [], []
    for orig, recon in zip(original_images, reconstructed_images):
        mse_values.append(calculate_mse(orig, recon, IMAGE_SIZE))
        ssim_values.append(ssim(orig.reshape(IMAGE_SIZE), recon.reshape(IMAGE_SIZE), data_range=1.0))
    return np.mean(mse_values), np.mean(ssim_values)


def plot_analysis_graphs(results_df):
    plt.figure()
    plt.plot(results_df["ratio"], results_df["MSE"], marker='o', label='MSE', color='blue')
    plt.xlabel("PCA Component Ratio")
    plt.ylabel("MSE", color='blue')
    plt.title("MSE & SSIM vs PCA Component Ratio")
    plt.grid()
    plt.legend(loc='upper left')

    ax2 = plt.gca().twinx()
    ax2.plot(results_df["ratio"], results_df["SSIM"], marker='s', linestyle='--', label='SSIM', color='red')
    ax2.set_ylabel("SSIM", color='red')
    ax2.legend(loc='upper right')

    plt.savefig(os.path.join(OUTPUT_FOLDER, "figure1-2_combined_mse_ssim.png"))


def run_pca_experiment(images, ratios):
    results = {"ratio": [], "MSE": [], "SSIM": []}
    reconstructed_images_dict = {}
    num_images = images.shape[0]
    example_idx = 0

    fig, axes = plt.subplots(2, max(len(ratios), 3), figsize=(15, 8))
    original_image = images[example_idx].reshape(IMAGE_SIZE)
    edge_image = sobel(original_image)

    # Centrage des images originales et eigenface en haut
    axes[0, (max(len(ratios), 3) - 2) // 2].imshow(original_image, cmap='gray')
    axes[0, (max(len(ratios), 3) - 2) // 2].set_title("Original Image")
    axes[0, (max(len(ratios), 3) - 2) // 2].axis('off')

    axes[0, (max(len(ratios), 3) - 2) // 2 + 1].imshow(edge_image, cmap='gray')
    axes[0, (max(len(ratios), 3) - 2) // 2 + 1].set_title("Edges Detected")
    axes[0, (max(len(ratios), 3) - 2) // 2 + 1].axis('off')

    for j in range(max(len(ratios), 3)):
        if j < (max(len(ratios), 3) - 2) // 2 or j > (max(len(ratios), 3) - 2) // 2 + 1:
            axes[0, j].axis('off')  # Désactiver les cases vides

    # Deuxième ligne : Reconstructions
    for i, ratio in enumerate(ratios):
        n_components = int(num_images * ratio)
        eigenface_generator = EigenfaceGenerator(images, n_components)
        eigenface_generator.generate()
        reconstructed_images = eigenface_generator.reconstruct_image(eigenface_generator.pca.transform(images))

        reconstructed_image = reconstructed_images[example_idx].reshape(IMAGE_SIZE)

        axes[1, i].imshow(reconstructed_image, cmap='gray')
        axes[1, i].set_title(f"Reconstructed {ratio}")
        axes[1, i].axis('off')

        mse, ssim_value = compute_metrics(images, reconstructed_images)
        results["ratio"].append(ratio)
        results["MSE"].append(mse)
        results["SSIM"].append(ssim_value)
        reconstructed_images_dict[ratio] = reconstructed_images

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "figure3_eigenface_all_results.png"))

    return pd.DataFrame(results), reconstructed_images_dict


if __name__ == "__main__":
    images = preprocess_images(DATASET_FOLDER)
    results_df, reconstructed_images_dict = run_pca_experiment(images, RATIOS)
    plot_analysis_graphs(results_df)