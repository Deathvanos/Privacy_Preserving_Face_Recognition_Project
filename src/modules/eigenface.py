import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from src.config import IMAGE_SIZE

class EigenfaceGenerator:
    """
    Génère des eigenfaces à partir d'un ensemble d'images via l'analyse en composantes principales (PCA).

    Args:
        images (np.ndarray): Tableau 2D (nombre d'images x taille d'image aplatie).
        n_components (int, optional): Nombre de composantes principales (eigenfaces) à générer. Défaut: 5.
    """
    def __init__(self, images: np.ndarray, n_components: int = 5) -> None:
        if not isinstance(images, np.ndarray) or images.ndim != 2:
            raise ValueError("`images` doit être un tableau NumPy 2D (nombre d'images x taille_aplatie).")
        self.images = images
        self.n_components = n_components if n_components is not None else min(images.shape)
        self.pca = None
        self.eigenfaces = None
        self.mean_face = None
        self.image_shape = IMAGE_SIZE  # On utilise une taille fixée par la config
        self.original_data = None  # Pour éventuellement la reconstruction

    def generate(self) -> None:
        """Exécute la PCA pour générer les eigenfaces."""
        if not self.images.any():
            raise ValueError("Aucune donnée d'image fournie.")
        if self.images.shape[0] < 2:
            raise ValueError("Au moins deux images sont requises pour générer les eigenfaces.")

        self.original_data = self.images
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(self.images)
        self.eigenfaces = [component.reshape(self.image_shape) for component in self.pca.components_]
        self.mean_face = self.pca.mean_.reshape(self.image_shape)

    def get_eigenfaces(self) -> list:
        """Renvoie la liste des eigenfaces générées."""
        if self.eigenfaces is None:
            self.generate()
        return self.eigenfaces

    def get_mean_face(self) -> np.ndarray:
        """Renvoie l'image moyenne calculée."""
        if self.mean_face is None:
            self.generate()
        return self.mean_face

    def get_pca_object(self) -> PCA:
        """Renvoie l'objet PCA utilisé pour la décomposition."""
        if self.pca is None:
            self.generate()
        return self.pca

    def reconstruct_image(self, projected_data: np.ndarray) -> np.ndarray:
        """
        Reconstruit une image à partir de sa projection sur le sous-espace des eigenfaces.

        Args:
            projected_data (np.ndarray): Coefficients dans l'espace eigenface.

        Returns:
            np.ndarray: Image reconstruite (aplatie).
        """
        if self.pca is None:
            raise ValueError("Les eigenfaces doivent être générées avant la reconstruction.")
        if projected_data.ndim == 1:
            projected_data = projected_data.reshape(1, -1)
        return self.pca.inverse_transform(projected_data)

    def plot_eigenfaces(self, output_folder: str, subject, filename: str = "eigenfaces", show_plot: bool = False) -> None:
        """
        Affiche et sauvegarde les eigenfaces générées.

        Args:
            output_folder (str): Dossier où enregistrer la figure.
            subject (int ou str): Identifiant du sujet pour le nom du fichier.
            filename (str, optional): Nom de base du fichier. Défaut: "eigenfaces".
            show_plot (bool, optional): Afficher la figure. Défaut: False.
        """
        if self.eigenfaces is None:
            self.generate()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure(figsize=(12, 6))
        num_eigenfaces = len(self.eigenfaces)
        cols = (num_eigenfaces + 1) // 2  # Calcul du nombre de colonnes
        for i, eigenface in enumerate(self.eigenfaces):
            plt.subplot(2, cols, i + 1)
            plt.imshow(eigenface, cmap='gray')
            plt.title(f'Eigenface {i + 1}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{filename}_{subject}.png"))
        if show_plot:
            plt.show()
        plt.close()

    def plot_mean_face(self, output_folder: str, subject, show_plot: bool = False) -> None:
        """
        Affiche et sauvegarde l'image moyenne.

        Args:
            output_folder (str): Dossier de sauvegarde.
            subject (int ou str): Identifiant du sujet pour le nom du fichier.
            show_plot (bool, optional): Afficher la figure. Défaut: False.
        """
        if self.mean_face is None:
            self.generate()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure()
        plt.imshow(self.mean_face, cmap='gray')
        plt.title("Mean face")
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f"mean_face_{subject}.png"))
        if show_plot:
            plt.show()
        plt.close()

    def plot_explained_variance(self, output_folder: str, show_plot: bool = False) -> None:
        """
        Trace et sauvegarde le cumul de la variance expliquée par composante.

        Args:
            output_folder (str): Dossier de sauvegarde.
            show_plot (bool, optional): Afficher la figure. Défaut: False.
        """
        if self.pca is None:
            self.generate()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        plt.figure()
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel("Nombre de composantes")
        plt.ylabel("Variance cumulée expliquée")
        plt.title('Explained Variance Ratio')
        plt.savefig(os.path.join(output_folder, "explained_variance.png"))
        if show_plot:
            plt.show()
        plt.close()
