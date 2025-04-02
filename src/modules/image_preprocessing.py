import numpy as np
from PIL import Image


def flatten_image(img: Image.Image) -> np.ndarray:
    """
    Aplati une image (PIL ou tableau NumPy) en un vecteur 1D.

    Args:
        img (PIL.Image.Image or np.ndarray): Image à aplatir.

    Returns:
        np.ndarray: Image aplatie.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    return img.flatten()


def preprocess_image(img: Image.Image, resize_size: tuple, create_grayscale: bool = True,
                     create_normalized: bool = True, create_flattened: bool = True) -> dict:
    """
    Prétraite une image en la redimensionnant et en appliquant diverses conversions.

    Args:
        img (PIL.Image.Image): L'image source.
        resize_size (tuple): La taille cible (largeur, hauteur).
        create_grayscale (bool, optional): Convertir en niveaux de gris. Défaut: True.
        create_normalized (bool, optional): Normaliser les pixels entre 0 et 1. Défaut: True.
        create_flattened (bool, optional): Aplatir l'image. Défaut: True.

    Returns:
        dict: Dictionnaire contenant les versions prétraitées de l'image.
    """
    try:
        processed_data = {}
        resized_img = img.resize(resize_size)
        if resized_img.size != resize_size:
            print(f"ERROR: Redimensionnement incorrect: {resized_img.size} au lieu de {resize_size}")
            return {}
        processed_data['resized_image'] = resized_img

        # Conversion en niveaux de gris
        if create_grayscale:
            grayscale_img = resized_img.convert("L")
            processed_data['grayscale_image'] = grayscale_img
        else:
            grayscale_img = resized_img

        # Normalisation
        if create_normalized:
            normalized_img = np.array(grayscale_img) / 255.0
            processed_data['normalized_image'] = normalized_img
        else:
            normalized_img = np.array(grayscale_img)

        # Aplatissement
        if create_flattened:
            flattened_image = normalized_img.flatten()
            processed_data['flattened_image'] = flattened_image

        return processed_data

    except Exception as e:
        print(f"Erreur lors du prétraitement: {e}")
        return {}
