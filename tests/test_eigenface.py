import unittest
import numpy as np
from sklearn.decomposition import PCA
from src.modules.eigenface import EigenfaceGenerator
from src.modules.utils_image import load_images  # Import the load_images function

class TestEigenfaceGenerator(unittest.TestCase):

    def setUp(self):
        image_folder = "data/dataset-yalefaces"
        # Use the load_images function from utils.py
        self.images = load_images(image_folder, subject_prefix="subject01")
        self.generator = EigenfaceGenerator(self.images, n_components=len(self.images))

    def test_generate(self):
        self.generator.generate()
        self.assertIsNotNone(self.generator.eigenfaces)
        self.assertIsNotNone(self.generator.mean_face)
        self.assertEqual(len(self.generator.eigenfaces), len(self.images))

    def test_get_eigenfaces(self):
        eigenfaces = self.generator.get_eigenfaces()
        self.assertIsInstance(eigenfaces, list)
        self.assertEqual(len(eigenfaces), len(self.images))

    def test_get_mean_face(self):
        mean_face = self.generator.get_mean_face()
        self.assertIsInstance(mean_face, np.ndarray)

    def test_get_pca_object(self):
        pca_object = self.generator.get_pca_object()
        self.assertIsInstance(pca_object, PCA)

if __name__ == '__main__':
    unittest.main()