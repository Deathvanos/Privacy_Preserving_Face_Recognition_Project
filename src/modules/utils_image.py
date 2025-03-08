import base64
import io
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def image_pillow_to_bytes(image_pil):
    """Converts a PIL Image to bytes."""
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='PNG')  # Or 'JPEG', etc.
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def image_numpy_to_pillow(image_array, resize_size):
    """Converts a NumPy array to a PIL Image."""
    # Ensure the image_array is within the valid range [0, 1]
    image_array = np.clip(image_array, 0, 1)
    # Scale to [0, 255] and convert to uint8
    image_array = (image_array * 255).astype(np.uint8)
    # Ensure the array is reshaped correctly if it's flattened
    if image_array.ndim == 1:
        image_array = image_array.reshape(resize_size)
    return Image.fromarray(image_array)