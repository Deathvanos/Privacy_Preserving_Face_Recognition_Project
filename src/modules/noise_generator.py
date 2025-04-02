import numpy as np
from functools import reduce
import operator


class NoiseGenerator:
    """
    Ajoute du bruit Laplacien aux images (ou projections) pour garantir la confidentialité différentielle.

    Args:
        eigenfaces_images (list ou np.ndarray): Images ou projections sur eigenfaces.
        epsilon (float): Paramètre de confidentialité (epsilon).

    Attributes:
        eigenfaces (np.ndarray): Données d'entrée converties en tableau NumPy.
        epsilon (float): Paramètre de confidentialité.
        img_shape (tuple): Dimensions d'une image.
        img_size (int): Taille aplatie d'une image.
        nb_images (int): Nombre d'images.
        eigenfaces_normalized (np.ndarray): Images normalisées.
        noised_eigenfaces (np.ndarray): Images après ajout de bruit.
    """

    def __init__(self, eigenfaces_images, epsilon: float) -> None:
        if not isinstance(eigenfaces_images, (list, np.ndarray)):
            raise TypeError("eigenfaces_images doit être une liste ou un tableau NumPy.")
        if not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon doit être un nombre.")
        if epsilon <= 0:
            raise ValueError("epsilon doit être supérieur à 0.")

        self.eigenfaces = np.array(eigenfaces_images)
        self.epsilon = epsilon

        if self.eigenfaces.ndim > 2:
            self.img_shape = self.eigenfaces[0].shape
            self.img_size = reduce(operator.mul, self.img_shape)
            self.nb_images = len(self.eigenfaces)
            self.is_projection = False
        elif self.eigenfaces.ndim == 2:
            self.img_shape = None
            self.img_size = self.eigenfaces.shape[1]
            self.nb_images = self.eigenfaces.shape[0]
            self.is_projection = True
        else:
            raise ValueError("eigenfaces_images a un nombre de dimensions non supporté.")

        self.eigenfaces_normalized = None
        self.noised_eigenfaces = None

    def flatten_images(self) -> None:
        """Aplati les images si elles ne le sont pas déjà."""
        if not self.is_projection:
            self.eigenfaces = self.eigenfaces.reshape(self.nb_images, self.img_size)

    def normalize_images(self) -> None:
        """Normalise les images dans l'intervalle [0, 1]."""
        images_copy = self.eigenfaces.copy()
        if self.is_projection:
            self.eigenfaces_normalized = self._normalize(images_copy)
        else:
            self.eigenfaces_normalized = np.array([self._normalize(img) for img in images_copy])

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalise une image ou projection.

        Args:
            image (np.ndarray): Image à normaliser.

        Returns:
            np.ndarray: Image normalisée.
        """
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val == 0:
            return np.zeros_like(image)
        return (image - min_val) / (max_val - min_val)

    def add_laplace_noise(self, sensitivity: float) -> None:
        """
        Ajoute du bruit Laplacien aux images normalisées.

        Args:
            sensitivity (float): Sensibilité de la requête.
        """
        if self.eigenfaces_normalized is None:
            self.normalize_images()
        if not isinstance(sensitivity, (int, float)):
            raise TypeError("La sensibilité doit être un nombre.")
        if sensitivity <= 0:
            raise ValueError("La sensibilité doit être > 0")
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=self.eigenfaces_normalized.shape)
        self.noised_eigenfaces = self.eigenfaces_normalized + noise

    def get_noised_eigenfaces(self) -> np.ndarray:
        """
        Renvoie les images avec bruit ajouté.

        Returns:
            np.ndarray: Images bruitées.
        """
        if self.noised_eigenfaces is None:
            raise ValueError("Le bruit n'a pas encore été ajouté. Veuillez appeler add_laplace_noise().")
        return self.noised_eigenfaces
