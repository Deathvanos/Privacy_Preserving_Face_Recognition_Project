import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
from typing import Union, Tuple, List

from src.modules.image_preprocessing import preprocess_image
from src.config import IMAGE_SIZE
from src.modules.peep import Peep

class Main:
    """
    Classe principale pour charger, prétraiter les images et créer des objets Peep.

    Args:
        image_size (tuple, optional): Taille cible pour le redimensionnement. Défaut: IMAGE_SIZE.
        image_extensions (tuple, optional): Extensions d'image autorisées. Défaut: (".png", ".jpg", ".jpeg").
    """
    def __init__(self, image_size: tuple = IMAGE_SIZE, image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> None:
        self.image_size = image_size
        self.image_extensions = image_extensions
        self.peep_objects = {}
        self.errors: List[str] = []

    def load_lfw_dataframe(self, min_faces_per_person: int = 20, n_samples_per_person: int = 20) -> Tuple[dict, list]:
        """
        Charge et équilibre le dataset LFW People pour obtenir exactement n_samples_per_person images par personne.
        Le DataFrame résultant contiendra les colonnes suivantes :
          - userFaces: Image PIL en niveaux de gris
          - imageId: Identifiant unique (index du DataFrame)
          - subject_number: Identifiant numérique du sujet
        Puis, ce DataFrame est passé à load_and_process_from_dataframe pour générer les objets Peep.

        Args:
            min_faces_per_person (int): Nombre minimum d'images par personne pour être inclus.
            n_samples_per_person (int): Nombre d'images à échantillonner par personne.

        Returns:
            Tuple[dict, list]: self.peep_objects et self.errors.
        """
        from sklearn.datasets import fetch_lfw_people

        # Charger le dataset LFW People
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
        height, width = lfw_people.images.shape[1], lfw_people.images.shape[2]

        # Vérifier la plage des données et convertir en uint8 correctement
        data = lfw_people.data
        if data.max() <= 1.0:
            # Si les valeurs sont dans [0,1], les ramener à [0,255]
            data = (data * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)

        # Créer un DataFrame à partir des données aplaties
        df = pd.DataFrame(data)
        df['subject_number'] = lfw_people.target
        df['target_names'] = df['subject_number'].apply(lambda x: lfw_people.target_names[x])

        # Équilibrer le dataset : échantillonner n_samples_per_person images par personne
        grouped = df.groupby('target_names')
        balanced_dfs = []
        for name, group in tqdm(grouped, desc="Sampling each person"):
            if len(group) >= n_samples_per_person:
                sampled_group = group.sample(n=n_samples_per_person, random_state=42)
                balanced_dfs.append(sampled_group)
        df_balanced = pd.concat(balanced_dfs)

        # Fonction pour reconstruire une image PIL à partir d'une ligne
        def row_to_image(row):
            # Les pixels se trouvent dans les colonnes de 0 à (height*width - 1)
            pixel_values = row.iloc[:height * width].values.astype(np.uint8)
            img_array = pixel_values.reshape((height, width))
            return Image.fromarray(img_array, mode='L')

        # Créer la colonne userFaces et un identifiant imageId
        df_balanced['userFaces'] = df_balanced.apply(row_to_image, axis=1)
        df_balanced['imageId'] = df_balanced.index

        # Optionnel : supprimer les colonnes des pixels et target_names
        columns_to_drop = list(range(height * width)) + ['target_names']
        df_balanced.drop(columns=columns_to_drop, inplace=True)

        # Utiliser la méthode existante pour générer les objets Peep
        peep_objects, errors = self.load_and_process_from_dataframe(df_balanced)
        return peep_objects, errors

    def load_and_process_from_folder(self, image_folder: str, subject_prefix: str = None,
                                     target_subject: int = None, epsilon: float = 9.0,
                                     method: str = 'bounded', unbounded_bound_type: str = 'l2',
                                     n_components: int = None) -> Tuple[dict, list]:
        """
        Charge, prétraite les images d'un dossier et crée les objets Peep par sujet.

        Args:
            image_folder (str): Chemin du dossier d'images.
            subject_prefix (str, optional): Préfixe pour filtrer les sujets.
            target_subject (int, optional): Numéro de sujet cible.
            epsilon (float, optional): Paramètre de confidentialité. Défaut: 9.0.
            method (str, optional): Méthode de calcul de sensibilité ('bounded' ou 'unbounded').
            unbounded_bound_type (str, optional): Type de borne pour 'unbounded' ('l2' ou 'empirical').
            n_components (int, optional): Nombre de composantes pour la PCA.

        Returns:
            Tuple[dict, list]: Dictionnaire d'objets Peep et liste d'erreurs.
        """
        if not os.path.isdir(image_folder):
            raise ValueError(f"Le dossier '{image_folder}' n'est pas valide.")

        subject_data = {}

        for filename in tqdm(os.listdir(image_folder), desc="Chargement et traitement"):
            if not filename.lower().endswith(self.image_extensions):
                continue

            try:
                parts = filename.split("_")
                if len(parts) < 4:
                    raise ValueError(f"Nom de fichier invalide: '{filename}'.")
                subject_number = int(parts[1])
                if target_subject is not None and subject_number != target_subject:
                    continue
                if subject_prefix is not None and str(subject_number) != subject_prefix:
                    continue

                image_path = os.path.join(image_folder, filename)
                with Image.open(image_path) as img:
                    processed_data = preprocess_image(img, resize_size=self.image_size, create_flattened=True)
                    if processed_data and processed_data.get('flattened_image') is not None:
                        subject_data.setdefault(subject_number, []).append(processed_data['flattened_image'])
                    else:
                        self.errors.append(f"Image {filename} ignorée (erreur de prétraitement).")

            except (IOError, OSError, ValueError) as e:
                self.errors.append(f"Erreur avec {filename}: {e}")
                continue

        self.peep_objects = self._create_peep_objects(subject_data, epsilon, method, unbounded_bound_type, n_components)
        return self.peep_objects, self.errors

    def load_and_process_from_dataframe(self, df: pd.DataFrame, target_subject: int = None,
                                        epsilon: float = 9.0, method: str = 'bounded',
                                        unbounded_bound_type: str = 'l2', n_components: int = None) -> Tuple[dict, list]:
        """
        Charge et prétraite les images à partir d'un DataFrame et crée les objets Peep.

        Args:
            df (pd.DataFrame): DataFrame avec colonnes 'userFaces', 'imageId' et 'subject_number'.
            target_subject (int, optional): Sujet cible.
            epsilon (float, optional): Paramètre de confidentialité.
            method (str, optional): Méthode de sensibilité.
            unbounded_bound_type (str, optional): Type de borne pour 'unbounded'.
            n_components (int, optional): Nombre de composantes pour la PCA.

        Returns:
            Tuple[dict, list]: Dictionnaire d'objets Peep et liste d'erreurs.
        """
        required_columns = {'userFaces', 'imageId', 'subject_number'}
        if not required_columns.issubset(df.columns):
            raise ValueError("Le DataFrame doit contenir 'userFaces', 'imageId' et 'subject_number'.")

        subject_data = {}
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Traitement du DataFrame"):
            try:
                img = row['userFaces']
                subject_number = row['subject_number']
                if target_subject is not None and subject_number != target_subject:
                    continue
                if not isinstance(img, Image.Image):
                    raise ValueError(f"Image invalide à l'index {index}.")
                processed_data = preprocess_image(img, resize_size=self.image_size, create_flattened=True)
                if processed_data and processed_data.get('flattened_image') is not None:
                    subject_data.setdefault(subject_number, []).append(processed_data['flattened_image'])
                else:
                    self.errors.append(f"Image ID {row['imageId']} ignorée (erreur de prétraitement).")

            except ValueError as e:
                self.errors.append(f"Erreur à l'index {index}: {e}")
                continue

        self.peep_objects = self._create_peep_objects(subject_data, epsilon, method, unbounded_bound_type, n_components)
        return self.peep_objects, self.errors

    def _create_peep_objects(self, subject_data: dict, epsilon: float, method: str,
                             unbounded_bound_type: str, n_components: int) -> dict:
        """
        Crée les objets Peep pour chaque sujet.

        Args:
            subject_data (dict): Dictionnaire {sujet: [images aplaties]}.
            epsilon (float): Paramètre de confidentialité.
            method (str): Méthode de calcul de sensibilité.
            unbounded_bound_type (str): Type de borne pour 'unbounded'.
            n_components (int): Nombre de composantes pour la PCA.

        Returns:
            dict: Objets Peep indexés par sujet.
        """
        peep_objects = {}
        for subject, images in subject_data.items():
            if not images:
                print(f"Aucun donnée pour le sujet {subject}. Ignoré.")
                continue
            try:
                images_array = np.array(images)
                peep = Peep(epsilon=epsilon, image_size=self.image_size)
                peep.run(images_array, method=method, unbounded_bound_type=unbounded_bound_type, n_components=n_components)
                peep_objects[subject] = peep
            except ValueError as e:
                print(f"Erreur pour le sujet {subject}: {e}")
                self.errors.append(f"Sujet {subject}: {e}")
                continue
        return peep_objects

    def get_peep_object(self, subject: int) -> Union[Peep, None]:
        """
        Récupère l'objet Peep pour un sujet donné.

        Args:
            subject (int): Numéro du sujet.

        Returns:
            Peep ou None: Objet Peep si trouvé.
        """
        return self.peep_objects.get(subject)

    def clear_errors(self) -> None:
        """Réinitialise la liste des erreurs."""
        self.errors = []

def main() -> None:
    """
    Fonction principale pour démontrer l'utilisation de la classe Main.
    """

    main_obj = Main()
    peep_objects, errors = main_obj.load_lfw_dataframe()
    if errors:
        print("Erreurs rencontrées:")
        for error in errors:
            print(error)
    else:
        print("Traitement terminé sans erreurs.")

if __name__ == "__main__":
    main()
