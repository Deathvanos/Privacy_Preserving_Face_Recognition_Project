import base64
import io
import os
import PIL
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from werkzeug.datastructures import FileStorage



def load_images(image_folder: str, subject_prefix: str = None,
                image_extensions: tuple = (".png", ".jpg", ".jpeg")) -> list:
    """
    Charge toutes les images d'un dossier en fonction d'un préfixe facultatif.

    Args:
        image_folder (str): Chemin du dossier.
        subject_prefix (str, optional): Préfixe pour filtrer les images.
        image_extensions (tuple, optional): Extensions d'image autorisées.

    Returns:
        list: Liste d'images PIL.
    """
    images = []
    for filename in os.listdir(image_folder):
        filename_split = filename.split("_")
        if filename_split[3].endswith(image_extensions) and (
                subject_prefix is None or filename_split[1] == subject_prefix):
            with Image.open(os.path.join(image_folder, filename)) as img:
                images.append(img.copy())
    return images


def resize_images(images: list, size: tuple) -> list:
    """Redimensionne une liste d'images."""
    return [img.resize(size) for img in images]


def crop_images(images: list, box: tuple) -> list:
    """Découpe une liste d'images selon une boîte (box)."""
    return [img.crop(box) for img in images]


def convert_to_grayscale(images: list) -> list:
    """Convertit une liste d'images en niveaux de gris."""
    return [img.convert('L') for img in images]


def normalize_images(images: list) -> list:
    """Normalise une liste d'images (tableaux) dans l'intervalle [0, 1]."""
    return [np.array(img) / 255.0 for img in images]


def flip_images(images: list, horizontal: bool = True) -> list:
    """Retourne une liste d'images retournées horizontalement ou verticalement."""
    if horizontal:
        return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
    else:
        return [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]


def rotate_images(images: list, angle: float) -> list:
    """Fait pivoter une liste d'images d'un angle donné."""
    return [img.rotate(angle) for img in images]


def plot_images(images: list, titles: list = None) -> None:
    """Affiche une liste d'images avec éventuellement des titres."""
    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()


def plot_histograms(images: list) -> None:
    """Affiche les histogrammes des niveaux de gris pour chaque image."""
    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.hist(np.array(img).ravel(), bins=256)
        plt.title(f"Image {i + 1}")
    plt.show()


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calcule et renvoie plusieurs métriques de performance.

    Args:
        y_true: Valeurs réelles.
        y_pred: Prédictions.

    Returns:
        dict: Dictionnaire des métriques (accuracy, precision, recall, f1).
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro")
    }
    return metrics


def create_folders(paths: list) -> None:
    """Crée les dossiers donnés s'ils n'existent pas."""
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def save_data(data, filename: str) -> None:
    """Enregistre des données dans un fichier .npy."""
    np.save(filename, data)


def load_data(filename: str):
    """Charge des données depuis un fichier .npy."""
    return np.load(filename)


def calculate_mse(image_a: np.ndarray, image_b: np.ndarray, image_size: tuple) -> float:
    """
    Calcule l'erreur quadratique moyenne (MSE) entre deux images.

    Args:
        image_a (np.ndarray): Première image.
        image_b (np.ndarray): Seconde image.
        image_size (tuple): Taille d'origine de l'image (hauteur, largeur).

    Returns:
        float: Valeur du MSE.
    """
    if image_a.ndim == 1:
        image_a = image_a.reshape(image_size)
    if image_b.ndim == 1:
        image_b = image_b.reshape(image_size)
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])
    return err


def calculate_ssim(image_a: np.ndarray, image_b: np.ndarray, data_range: float = None) -> float:
    """
    Calcule l'indice de similarité structurelle (SSIM) entre deux images.

    Args:
        image_a (np.ndarray): Première image.
        image_b (np.ndarray): Seconde image.
        data_range (float, optional): Plage de données. Défaut: None.

    Returns:
        float: Valeur du SSIM.
    """
    if image_a.ndim == 3:
        image_a = np.dot(image_a[..., :3], [0.2989, 0.5870, 0.1140])
    if image_b.ndim == 3:
        image_b = np.dot(image_b[..., :3], [0.2989, 0.5870, 0.1140])
    return ssim(image_a, image_b, data_range=data_range)




def filestorage_image_to_pil(element: FileStorage|list[FileStorage]) -> PIL.Image.Image|list[PIL.Image.Image]:
    """Converts a FileStorage Image or a list of FileStorage Image to PIL image(s)."""
    if element is None:
        raise ValueError("no element for filestorage_image_to_pil()")
    if isinstance(element, list):
        return [Image.open(io.BytesIO(image.read())) for image in element]
    else:
        return Image.open(io.BytesIO(element.read()))


def filestorage_image_to_numpy(element: FileStorage | list[FileStorage]) -> np.ndarray | list[np.ndarray]:
    """Converts a FileStorage Image or a list of FileStorage Image to numpy array(s) with proper color channels."""
    if element is None:
        raise ValueError("no element for filestorage_image_to_numpy()")
    if isinstance(element, list):
        return [np.array(Image.open(io.BytesIO(image.read())).convert('RGB')) for image in element]
    else:
        return np.array(Image.open(io.BytesIO(element.read())).convert('RGB'))

def pillow_image_to_bytes(element: PIL.Image.Image|list[PIL.Image.Image]) -> str|list[str]:
    """Converts a PIL image or a list of PIL image to a bytes string image(s)."""
    if element is None:
        raise ValueError("no element for pillow_image_to_bytes()")
    def convert(image):
        if not isinstance(image, Image.Image):
            raise ValueError("'image' must be a valid PIL Image object.")
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()
    if isinstance(element, list):
        return [convert(image) for image in element]
    else:
        return convert(element)


def numpy_image_to_pillow(element: np.ndarray | list[np.ndarray], resized_size: (int, int) = None,
                          list_mode: bool = False) -> Image.Image | list[Image.Image]:
    """Converts a NumPy array or a list of NumPy array to a PIL image(s)."""
    if element is None:
        raise ValueError("no element for numpy_image_to_pillow()")

    def convert(image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("'image' must be a valid NumPy array.")
        elif image.ndim == 1:
            if resized_size is None:
                raise ValueError("'resized_size' must be provided because the image is one-dimensional.")
            image = image.reshape(resized_size)

        # Assurez-vous que les valeurs sont dans la plage correcte
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Gérez les canaux de couleur
        if image.ndim == 2:
            return Image.fromarray(image, mode='L')  # Image en niveaux de gris
        elif image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image, mode='RGB')  # Image en couleur
        else:
            raise ValueError("Unsupported image shape: {}".format(image.shape))

    if isinstance(element, list) or list_mode:
        return [convert(image) for image in element]
    else:
        return convert(element)

