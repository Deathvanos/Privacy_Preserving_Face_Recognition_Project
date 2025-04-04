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


def load_images(image_folder, subject_prefix=None, image_extensions=(".png", ".jpg", ".jpeg")):
    images = []
    for filename in os.listdir(image_folder):
        filename_split = filename.split("_")
        if filename_split[3].endswith(image_extensions) and (subject_prefix is None or filename_split[1] == subject_prefix):
            with Image.open(os.path.join(image_folder, filename)) as img:
                images.append(img.copy())
    return images


def resize_images(images, size):
    return [img.resize(size) for img in images]


def crop_images(images, box):
    return [img.crop(box) for img in images]


def convert_to_grayscale(images):
    return [img.convert('L') for img in images]


def normalize_images(images):
    return [np.array(img) / 255.0 for img in images]


def flip_images(images, horizontal=True):
    if horizontal:
        return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
    else:
        return [img.transpose(Image.FLIP_TOP_BOTTOM) for img in images]


def rotate_images(images, angle):
    return [img.rotate(angle) for img in images]


def plot_images(images, titles=None):
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


def plot_histograms(images):
    num_images = len(images)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    plt.figure(figsize=(12, 6))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.hist(np.array(img).ravel(), bins=256)
        plt.title(f"Image {i+1}")
    plt.show()


def calculate_metrics(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro")
    }
    return metrics


def create_folders(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def save_data(data, filename):
    np.save(filename, data)


def load_data(filename):
    return np.load(filename)




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


# TO DELETE
def image_numpy_to_pillow(image, resized_size=None):
    """Converts a NumPy array to a PIL Image."""
    if image is None or type(image) != np.ndarray:
        raise ValueError("'image' must be a valid NumPy array.")
    elif image.ndim == 1:
        if resized_size is None:
            raise ValueError("'resized_size' must be provided because the image is one-dimensional.")
        image = image.reshape(resized_size)
    return Image.fromarray((image * 255).astype(np.uint8))

def calculate_mse(imageA, imageB, image_size):
    if imageA.ndim == 1:
        imageA = imageA.reshape(image_size)
    if imageB.ndim == 1:
        imageB = imageB.reshape(image_size)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_ssim(imageA, imageB, data_range=None):
    if imageA.ndim == 3:
        imageA = np.dot(imageA[...,:3], [0.2989, 0.5870, 0.1140])  # Convertit en niveaux de gris
    if imageB.ndim == 3:
        imageB = np.dot(imageB[...,:3], [0.2989, 0.5870, 0.1140])  # Convertit en niveaux de gris
        if data_range is None:
            data_range = imageB.max() - imageB.min()
    return ssim(imageA, imageB, data_range=data_range)
