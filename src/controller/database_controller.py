import os
import io
import json
import base64
import shutil
import sqlite3
import kagglehub
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.other import lfw_dataset_process as utils



class DatabaseController:

    _table_name = "noised_user_components"
    _column_id = "id"
    _column_data = "array"

    # Dataset path
    lfw_folder = r"data\dataset-lfw_reconstructed"
    yalefaces_folder = r"data\dataset-yalefaces"

    def __init__(self, path=r"data\gui_database.db"):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Log in the database
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        # Create noised user table
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                {self._column_id} INTEGER PRIMARY KEY AUTOINCREMENT,
                {self._column_data} TEXT
            )
        ''')
    def __del__(self):
        # Close database connexions
        self.conn.commit()
        self.conn.close()

    def add_user(self, noised_vectors: np.ndarray):
        json_array = json.dumps(noised_vectors.tolist())
        self.cursor.execute(f"INSERT INTO {self._table_name} ({self._column_data}) VALUES (?)",(json_array,))
        self.conn.commit()
        return self.cursor.lastrowid # Send user id

    def get_user(self, id) -> np.ndarray:
        self.cursor.execute(f"SELECT {self._column_data} FROM {self._table_name} WHERE {self._column_id} = (?)", (id,))
        result = self.cursor.fetchone()
        if result:
            retrieved_json = result[0]
            retrieved_array = np.array(json.loads(retrieved_json))
            return retrieved_array # Send user data
        return None # No user with this id

    def update_user(self, id, new_noised_vectors: np.ndarray):
        json_array = json.dumps(new_noised_vectors.tolist())
        self.cursor.execute(f"UPDATE {self._table_name} SET {self._column_data} = (?) WHERE {self._column_id} = (?)",(json_array, id))
        self.conn.commit()
        return self.cursor.rowcount # Number of affected rows

    def delete_user(self, id):
        self.cursor.execute(f"DELETE FROM {self._table_name} WHERE {self._column_id} = (?)",(id,))
        self.conn.commit()
        return self.cursor.rowcount # Number of affected rows

    def get_user_id_list(self):
        self.cursor.execute(f"SELECT {self._column_id} FROM {self._table_name}")
        result = self.cursor.fetchall()
        return [row[0] for row in result] # Send list of user IDs

    def get_user_vectors(self):
        self.cursor.execute(f"SELECT {self._column_data} FROM {self._table_name}")
        result = self.cursor.fetchall()
        return [row[0] for row in result]

    def get_table(self):
        self.cursor.execute(f"SELECT * FROM {self._table_name}")
        result = self.cursor.fetchall()
        return result


    def reset_database(self):
        # Close database connexions
        self.conn.commit()
        self.conn.close()
        # Delete database file
        os.remove(self.path)
        # Restart database creation
        self.__init__(self.path)
        return


    def process_dataset(self, images64_dict):
        # Create a new db
        self.reset_database()
        # Save lwf user in database.
        for user, images64 in images64_dict.items():
            images64 = np.array(images64['final_reconstructed_b64'])
            self.add_user(images64)


    def load_lfw_dataset(self, directory=lfw_folder):
        # Download  and processe lfw dataset
        utils.anony_process_pipeline.set_reconstructed_dir(directory)
        df = utils.load_lfw_dataframe(min_faces_per_person=20, n_samples_per_person=20)
        images64_dict = utils.anony_process_pipeline.run_pipeline(df_images=df, k_same_k_value=10, epsilon=0.0124, n_components_ratio=0.19)
        self.process_dataset(images64_dict)

    def load_yalefaces_dataset(self, directory=yalefaces_folder):
        # Download latest version
        path = kagglehub.dataset_download("olgabelitskaya/yale-face-database")
        images_base64 = []
        images_dict = []
        # Prepare the move
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
        # Move the folder into the project
        for idx, filename in enumerate(os.listdir(path)):
            if filename in ["data", "Readme.txt"]:
                continue
            # for each image
            input_path = os.path.join(path, filename)
            try:
                img = plt.imread(input_path)
                img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                # Save in .png
                img_pil = Image.fromarray(img_uint8)
                output_path = os.path.join(directory, filename + ".png")
                img_pil.save(output_path)
                # Convertir et sauvegarder en .png
                img_pil = Image.fromarray(img_uint8)
                output_path = os.path.join(directory, filename + ".png")
                img_pil.save(output_path)
                # Convert in base64 image for folder save
                buffered = io.BytesIO()
                img_pil.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                images_base64.append(img_base64)
                # Create dataframe dict for noised database save
                subject_number = filename.split('.')[0][-2:]
                images_dict.append({
                    "subject_number": int(subject_number),
                    "userFaces": img_pil,
                    "id": idx
                })
            except Exception as e:
                print(f"Erreur avec {filename} : {e}")
        images_df = pd.DataFrame(images_dict)
        # Preprocess images
        temps_folder = "temp"
        utils.anony_process_pipeline.set_reconstructed_dir(temps_folder)
        images64_dict = utils.anony_process_pipeline.run_pipeline(df_images=images_df, k_same_k_value=10, epsilon=0.0024, n_components_ratio=0.19)
        shutil.rmtree(temps_folder)
        self.process_dataset(images64_dict)









