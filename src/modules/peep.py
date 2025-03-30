import numpy as np
from PIL import Image

from src.config import IMAGE_SIZE
from src.modules.eigenface import EigenfaceGenerator
from src.modules.noise_generator import NoiseGenerator
from src.modules.utils_image import image_numpy_to_pillow, image_pillow_to_bytes


class Peep:
    """
    Implémente les eigenfaces différentiellement privées pour la reconnaissance faciale.

    Args:
        epsilon (int, optional): Paramètre de confidentialité. Défaut: 9.
        image_size (tuple, optional): Taille cible pour le redimensionnement. Défaut: IMAGE_SIZE.
    """

    def __init__(self, epsilon: int = 9, image_size=IMAGE_SIZE) -> None:
        if not isinstance(epsilon, (int, float)):
            raise TypeError("Epsilon doit être un nombre.")
        if epsilon <= 0:
            raise ValueError("Epsilon doit être supérieur à 0.")
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size doit être un tuple de longueur 2 (hauteur, largeur).")

        self.resize_size = image_size
        self.epsilon = float(epsilon)
        self.pca_object = None
        self.projected_vectors = None
        self.noised_vectors = None
        self.sensitivity = None
        self.max_components = None

    def _generate_eigenfaces(self, images_data: np.ndarray, n_components: int = None) -> None:
        """
        Génère les eigenfaces à partir des données d'image prétraitées.

        Args:
            images_data (np.ndarray): Tableau 2D (nombre d'images x taille aplatie).
            n_components (int, optional): Nombre de composantes pour la PCA.
        """
        if not isinstance(images_data, np.ndarray):
            raise ValueError("images_data doit être un tableau NumPy.")
        if images_data.ndim != 2:
            raise ValueError("images_data doit être un tableau 2D (nombre d'images x pixels).")

        num_images = images_data.shape[0]
        self.max_components = num_images

        if n_components is None:
            n_components = min(num_images - 1, self.max_components)

        if n_components <= 0:
            raise ValueError("Pas assez d'images pour générer les eigenfaces (doit être > 1).")
        if n_components > images_data.shape[1]:
            raise ValueError("n_components ne peut dépasser le nombre de caractéristiques (pixels).")

        try:
            self.pca_object = EigenfaceGenerator(images_data, n_components=n_components)
            self.pca_object.generate()
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la génération des eigenfaces: {e}")

    def _project_images(self, images_data: np.ndarray) -> None:
        """
        Projette les images dans le sous-espace des eigenfaces.

        Args:
            images_data (np.ndarray): Données d'image (2D).
        """
        if self.pca_object is None:
            raise ValueError("Les eigenfaces doivent être générées avant la projection.")
        self.projected_vectors = self.pca_object.pca.transform(images_data)

    def _calculate_sensitivity(self, method: str = 'bounded', unbounded_bound_type: str = 'l2') -> None:
        """
        Calcule la sensibilité de la transformation PCA.

        Args:
            method (str, optional): Méthode de calcul ('bounded' ou 'unbounded'). Défaut: 'bounded'.
            unbounded_bound_type (str, optional): Type de borne pour 'unbounded' ('l2' ou 'empirical'). Défaut: 'l2'.
        """
        if self.pca_object is None:
            raise ValueError("Les eigenfaces doivent être générées avant de calculer la sensibilité.")

        if method == 'bounded':
            max_image_diff_norm = np.sqrt(2)
            sensitivity = max_image_diff_norm * np.linalg.norm(self.pca_object.pca.components_, ord=2)
        elif method == 'unbounded':
            if unbounded_bound_type == 'l2':
                max_image_norm = np.sqrt(self.resize_size[0] * self.resize_size[1])
                sensitivity = (2 * (max_image_norm ** 2)) / len(self.projected_vectors)
            elif unbounded_bound_type == 'empirical':
                max_diff = 0
                for i in range(len(self.projected_vectors)):
                    for j in range(i + 1, len(self.projected_vectors)):
                        diff = np.linalg.norm(self.projected_vectors[i] - self.projected_vectors[j])
                        max_diff = max(max_diff, diff)
                sensitivity = max_diff
            else:
                raise ValueError("unbounded_bound_type invalide. Choisissez 'l2' ou 'empirical'.")
        else:
            raise ValueError("Méthode de sensibilité invalide. Choisissez 'bounded' ou 'unbounded'.")

        self.sensitivity = sensitivity

    def set_epsilon(self, epsilon: float) -> None:
        """
        Met à jour la valeur de epsilon.

        Args:
            epsilon (float): Nouvelle valeur d'epsilon.
        """
        if not isinstance(epsilon, (int, float)):
            raise TypeError("Epsilon doit être un nombre.")
        if epsilon <= 0:
            raise ValueError("Epsilon doit être supérieur à 0.")
        self.epsilon = float(epsilon)

    def _add_laplace_noise(self) -> None:
        """
        Ajoute du bruit Laplacien aux vecteurs projetés.
        """
        if self.projected_vectors is None:
            raise ValueError("Les images doivent être projetées avant d'ajouter du bruit.")
        if self.sensitivity is None:
            raise ValueError("La sensibilité doit être calculée avant d'ajouter du bruit.")

        noise_generator = NoiseGenerator(self.projected_vectors, self.epsilon)
        noise_generator.normalize_images()
        noise_generator.add_laplace_noise(self.sensitivity)
        self.noised_vectors = noise_generator.get_noised_eigenfaces()

    def run(self, images_data: np.ndarray, method: str = 'bounded', unbounded_bound_type: str = 'l2',
            n_components: int = None) -> bool:
        """
        Exécute l'ensemble du processus : génération des eigenfaces, projection, calcul de la sensibilité et ajout de bruit.

        Args:
            images_data (np.ndarray): Tableau 2D d'images aplaties.
            method (str): Méthode de calcul de sensibilité.
            unbounded_bound_type (str): Type de borne pour 'unbounded'.
            n_components (int, optional): Nombre de composantes pour la PCA.

        Returns:
            bool: True si le processus est terminé avec succès.
        """
        self._generate_eigenfaces(images_data, n_components)
        self._project_images(images_data)
        self._calculate_sensitivity(method, unbounded_bound_type)
        self._add_laplace_noise()
        return True

    def get_eigenfaces(self, format: str = 'numpy'):
        """
        Récupère les eigenfaces générées.

        Args:
            format (str, optional): Format désiré ('numpy', 'pillow', etc.). Défaut: 'numpy'.

        Returns:
            Selon le format choisi, retourne les eigenfaces.
        """
        if self.pca_object is None:
            raise ValueError("Les eigenfaces n'ont pas été générées.")
        if format == 'numpy':
            return self.pca_object.get_eigenfaces()
        elif format == 'pillow':
            return [image_numpy_to_pillow(face, resized_size=self.resize_size) for face in
                    self.pca_object.get_eigenfaces()]
        else:
            raise ValueError("Format non supporté.")

    def get_mean_face(self, format: str = 'numpy'):
        """
        Récupère l'image moyenne.

        Args:
            format (str, optional): Format désiré ('numpy', 'pillow', etc.). Défaut: 'numpy'.

        Returns:
            L'image moyenne dans le format spécifié.
        """
        if self.pca_object is None:
            raise ValueError("Les eigenfaces n'ont pas été générées.")
        mean_face = self.pca_object.get_mean_face()
        if format == 'numpy':
            return mean_face
        elif format == 'pillow':
            return image_numpy_to_pillow(mean_face, resized_size=self.resize_size)
        else:
            raise ValueError("Format non supporté.")

    def get_pca_object(self) -> EigenfaceGenerator:
        """
        Renvoie l'objet PCA associé.

        Returns:
            EigenfaceGenerator: Instance de la classe EigenfaceGenerator.
        """
        if self.pca_object is None:
            raise ValueError("L'objet PCA n'est pas encore créé.")
        return self.pca_object

    def image_pillow_to_bytes(self, image: Image.Image) -> str:
        """
        Convertit une image PIL en bytes encodés en base64.

        Args:
            image (PIL.Image.Image): Image à convertir.

        Returns:
            str: Image encodée en base64.
        """
        return image_pillow_to_bytes(image)
