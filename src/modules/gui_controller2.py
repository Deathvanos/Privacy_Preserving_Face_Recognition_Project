import os
import pickle

import PIL
from numpy import ndarray
from poetry.console.commands import self
from werkzeug.datastructures import FileStorage
import numpy as np
import modules.anony_process_pipeline as pipeline

from config import IMAGE_SIZE
from modules.gui_controller import GUIController
from modules.image_preprocessing import preprocess_image
from modules.utils_image import pillow_image_to_bytes, filestorage_image_to_numpy, numpy_image_to_pillow
from modules.database_controller import DatabaseController

class GUIController2:
    path = r"data\temp_gui_controller.pkl"

    def __init__(self, images: list[FileStorage]):
        if not images:
            raise Exception('No images')
        if not all(isinstance(image, FileStorage) for image in images):
            raise Exception('All images must be of type FileStorage')
        # Images Attributs
        self.image_size: (int, int) = (100,) * 2
        self.images_source: list[np.ndarray] = filestorage_image_to_numpy(images)
        self.step1 = self.step2 = self.step3 = self.step4 = self.step5 = None
        self.next_step = 1



    #-----------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------#
    #-------------------------# DATABASE MANAGEMENT WORKFLOW #--------------------------#
    #-----------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------#
    @classmethod
    def get_user_data(cls, user_id: int):
        return DatabaseController().get_user(user_id)

    @classmethod
    def delete_user(cls, user_id: int):
        return DatabaseController().delete_user(user_id)

    #-----------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------#
    #-----------------------------# CREATE USER WORKFLOW #------------------------------#
    #-----------------------------------------------------------------------------------#

    def can_run_step(self, step:int):
        return 0 <= step <= self.next_step

    @classmethod
    def initialize_new_user(cls, files: list[FileStorage], image_size:(int, int)=None,) -> (dict, int):
        """Step 1 : Preprocessing"""
        # TODO: add image_size to the process
        # Check input images format
        if not files: return {'error': 'No files uploaded'}, 400
        if not all(isinstance(file, FileStorage) for file in files): return {'error': 'Uploaded files are invalid'}, 400
        # Init GUI Controller
        ctrl = GUIController2(files)
        ##### STEP 1 : Preprocessing #####
        try: ctrl.step1, ctrl.image_size = pipeline.run_preprocessing(filestorage_list=files, image_size_override=image_size)
        except Exception as e:  return {'error': str(e)}, 400
        # Init Pickle file to save GUI Controller
        ctrl.next_step = 2
        ctrl.save_into_pickle()
        # Return validation of the process
        return {}, 200

    @classmethod
    def apply_k_same_pixel(cls, k_same_value:int=4):
        """Step 2 : K-Same-Pixel"""
        # Retrieve GUI Controller
        ctrl = GUIController2.load_pickle_file()
        if not ctrl: return {'error': 'Please run the fist step before this one'}, 400
        if not ctrl.can_run_step(2):  return {'error': 'Step not ready to run'}, 400
        # Check input images format
        if k_same_value is None: return {'error': 'k_same_value parameter is missing'}, 400
        ##### Step 2 : K-Same-Pixel #####
        try:  ctrl.step2 = pipeline.run_k_same_anonymization(ctrl.step1, k_same_value)
        except Exception as e:  return {'error': str(e)}, 400
        # Update Pickle GUI Controller
        ctrl.next_step = 3
        ctrl.save_into_pickle()
        # Return validation of the process
        images = ctrl.step2[list(ctrl.step2.keys())[0]]
        images = [img['flattened_anonymized_image'] for img in ctrl.step2["upload_subject_1"]]
        images = pillow_image_to_bytes(numpy_image_to_pillow(images, ctrl.image_size, True))
        return {'images': images}, 200

    @classmethod
    def generate_pca_components(cls, n_components: int=None):
        """Step 3 : PEEP Eigenface"""
        # Retrieve GUI Controller
        ctrl = GUIController2.load_pickle_file()
        if not ctrl: return {'error': 'Please run the fist step before this one'}, 400
        if not ctrl.can_run_step(3):  return {'error': 'Step not ready to run'}, 400
        # Check input images format
        user = list(ctrl.step2.keys())[0]
        images = [img['flattened_anonymized_image'] for img in ctrl.step2[user]]
        images = np.array(images, dtype=np.float32)
        n_samples, n_features = images.shape
        # TODO: change with anony_test_analysis.py to generate real optimal number + graph et mettre cette partie dans une fonction à part dans le back
        n_components_ratio = 0.8
        n_components_optimal = min(max(1, int(n_components_ratio * n_samples)), n_features)
        print(f"n_components_optimal: {n_components_optimal}")
        if n_components is None:
            n_components = n_components_optimal
        else:
            try: n_components = int(n_components)
            except: return {'error': 'n_components must be an Integer'}, 400
        ##### Step 3 : PEEP Eigenface #####
        try:  ctrl.step3 = pipeline.run_eigenface(images, n_components)
        except Exception as e:  return {'error': str(e)}, 400
        # Update Pickle GUI Controller
        ctrl.next_step = 4
        ctrl.save_into_pickle()
        # Return validation of the process
        return {}, 200

    @classmethod
    def apply_differential_privacy(cls, epsilon: float):
        """Step 4 & 5 : Differential privacy & Reconstruction"""
        # Retrieve GUI Controller
        ctrl = GUIController2.load_pickle_file()
        if not ctrl: return {'error': 'Please run the fist step before this one'}, 400
        if not ctrl.can_run_step(4):  return {'error': 'Step not ready to run'}, 400
        # Check input images format
        if epsilon is None: return {'error': 'epsilon parameter is missing'}, 400
        try: epsilon = float(epsilon)
        except: return {'error': 'epsilon must be a float'}, 400
        try:
            ##### Step 4 : Differential privacy #####
            pca, mean_face, projection = ctrl.step3
            ctrl.step4 = pipeline.run_add_noise(projection, epsilon)
            ##### Step 5 : Reconstruction #####
            ctrl.step5 = pipeline.run_reconstruction(pca, ctrl.step4, ctrl.image_size)
        except Exception as e:
            return {'error': str(e)}, 400
        # Update Pickle GUI Controller
        ctrl.next_step = 5
        ctrl.save_into_pickle()
        # Return validation of the process
        return {'images': ctrl.step5}, 200

    @classmethod
    def save_user_in_db(cls):
        # Retrieve GUI Controller
        ctrl = GUIController2.load_pickle_file()
        if not ctrl: return {'error': 'Please run the fist step before this one'}, 400
        if not ctrl.can_run_step(4):  return {'error': 'Step not ready to run'}, 400
        # Save user
        try:
            db = DatabaseController()
            user_id = db.add_user(np.array(ctrl.step5))
        except Exception as e:
            return {'error': str(e)}, 400
        # Destroy Pickle GUI Controller
        GUIController.delete_temp_file()
        # Return validation of the process
        return {'user_id': user_id}, 200


    #-----------------------------------------------------------------------------------#
    #-----------------------------# PICKLE SYSTEM PART #------------------------------#
    #-----------------------------------------------------------------------------------#

    def save_into_pickle(self):
        # Create pickle file
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'wb') as file:
            # Save data
            pickle.dump(self, file)

    @classmethod
    def load_pickle_file(cls):
        try:
            with open(GUIController2.path, 'rb') as file:
                obj = pickle.load(file)
            return obj
        except FileNotFoundError:
            return None
        except Exception as e:
            raise e

    @classmethod
    def delete_pickle_file(cls):
        if os.path.exists(GUIController2.path):
            os.remove(GUIController2.path)

