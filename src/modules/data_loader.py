# src/modules/data_loader.py

import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib # Pour sauvegarder/charger le LabelEncoder
from typing import Tuple, List, Optional, Dict, Any

# --- Fonctions Utilitaires ---

def save_label_encoder(encoder: LabelEncoder, filepath: str):
    """Sauvegarde un objet LabelEncoder dans un fichier."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True) # Crée le dossier parent si besoin
        joblib.dump(encoder, filepath)
        print(f"LabelEncoder sauvegardé dans : {filepath}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du LabelEncoder : {e}")

def load_label_encoder(filepath: str) -> Optional[LabelEncoder]:
    """Charge un objet LabelEncoder depuis un fichier."""
    try:
        if os.path.exists(filepath):
            encoder = joblib.load(filepath)
            print(f"LabelEncoder chargé depuis : {filepath}")
            return encoder
        else:
            print(f"Erreur: Fichier LabelEncoder non trouvé à : {filepath}")
            return None
    except Exception as e:
        print(f"Erreur lors du chargement du LabelEncoder : {e}")
        return None

# --- Fonction Principale de Chargement ---

def load_anonymized_images_flat(
    data_dir: str,
    img_width: int,
    img_height: int,
    color_mode: str = 'grayscale' # 'grayscale' ou 'rgb'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[LabelEncoder]]:
    """
    Charge les images LFW anonymisées depuis un dossier plat.

    Les noms de fichiers doivent suivre le pattern: 'reconstructed_<subject_id>_<num_img>.png'
    où <subject_id> est un entier.

    Args:
        data_dir: Chemin vers le dossier contenant les images .png.
        img_width: Largeur cible des images.
        img_height: Hauteur cible des images.
        color_mode: 'grayscale' pour niveaux de gris (1 canal), 'rgb' pour couleur (3 canaux).

    Returns:
        Tuple contenant:
        - X: np.ndarray des données image (None en cas d'erreur).
        - y: np.ndarray des labels entiers encodés (None en cas d'erreur).
        - label_encoder: L'objet LabelEncoder fitté (None en cas d'erreur).
        Retourne (None, None, None) si aucune image n'est chargée ou en cas d'erreur majeure.
    """
    images = []
    labels_original = [] # Stocke les subject_id (strings) extraits des noms de fichiers
    required_parts = 3 # 'reconstructed', '<subject_id>', '<num_img>.png'

    print(f"Chargement des images depuis : {data_dir}")
    print(f"Format attendu : {img_width}x{img_height}, mode: {color_mode}")

    if not os.path.isdir(data_dir):
        print(f"Erreur: Le dossier spécifié n'existe pas : {data_dir}")
        return None, None, None

    try:
        filenames = [f for f in os.listdir(data_dir) if f.lower().endswith('.png')]
        if not filenames:
            print(f"Erreur: Aucun fichier .png trouvé dans {data_dir}")
            return None, None, None

        print(f"Trouvé {len(filenames)} fichiers PNG.")

        processed_files = 0
        skipped_files = 0
        for filename in filenames:
            try:
                parts = filename.split('_')
                # Vérifie le format du nom de fichier
                if len(parts) < required_parts or not parts[1].isdigit():
                    print(f"Attention: Format de nom de fichier inattendu, fichier ignoré : {filename}")
                    skipped_files += 1
                    continue

                subject_id = parts[1] # L'étiquette est le subject_id

                # Chargement de l'image
                img_path = os.path.join(data_dir, filename)
                with Image.open(img_path) as img:
                    # Conversion couleur
                    pil_mode = 'L' if color_mode == 'grayscale' else 'RGB'
                    img_converted = img.convert(pil_mode)

                    # Redimensionnement
                    img_resized = img_converted.resize((img_width, img_height))

                    # Conversion en NumPy array
                    img_array = np.array(img_resized)

                    images.append(img_array)
                    labels_original.append(subject_id)
                    processed_files += 1

            except FileNotFoundError:
                print(f"Erreur: Fichier non trouvé (peut-être supprimé pendant le scan?) : {filename}")
                skipped_files += 1
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {filename}: {e}")
                skipped_files += 1

        print(f"Chargement terminé. {processed_files} images traitées, {skipped_files} fichiers ignorés.")

        if not images:
            print("Erreur: Aucune image n'a pu être chargée correctement.")
            return None, None, None

        # Conversion en NumPy arrays
        X = np.array(images)
        y_original = np.array(labels_original)

        # Normalisation des pixels [0, 1]
        X = X.astype('float32') / 255.0

        # Reshape pour Keras/TensorFlow: (samples, height, width, channels)
        channels = 1 if color_mode == 'grayscale' else 3
        X = X.reshape(-1, img_height, img_width, channels)
        print(f"Shape final des données (X): {X.shape}")

        # Encodage des labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_original)
        num_classes = len(label_encoder.classes_)
        print(f"{len(y_original)} labels originaux encodés en {num_classes} classes numériques.")
        # Afficher le mapping (optionnel, peut être long)
        # print("Mapping LabelEncoder (numérique -> original):")
        # for i, class_name in enumerate(label_encoder.classes_):
        #    print(f"{i} -> {class_name}")

        return X, y, label_encoder

    except Exception as e:
        print(f"Erreur majeure lors du chargement des données : {e}")
        return None, None, None


# --- Fonctions de Division des Données ---

def split_data_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    validation_size: float = 0.0, # Proportion du set d'entraînement à utiliser pour la validation
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Divise les données en ensembles d'entraînement, de validation et de test
    en utilisant une stratification standard.

    Args:
        X: Données image.
        y: Labels entiers.
        test_size: Proportion pour l'ensemble de test (ex: 0.2 pour 20%).
        validation_size: Proportion de l'ensemble d'entraînement *initial* à
                         utiliser comme ensemble de validation (ex: 0.1 pour 10%).
                         Si 0.0, aucun ensemble de validation n'est retourné séparément.
        random_state: Seed pour la reproductibilité.

    Returns:
        Un dictionnaire contenant les ensembles:
        {'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ..., 'X_test': ..., 'y_test': ...}
        Les clés de validation peuvent être absentes si validation_size est 0.
    """
    print(f"Division des données : test_size={test_size}, validation_size={validation_size}")
    data_split: Dict[str, np.ndarray] = {}

    if test_size > 0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        data_split['X_test'] = X_test
        data_split['y_test'] = y_test
        print(f"Taille Test: {len(X_test)} échantillons")
    else:
        X_train_val, y_train_val = X.copy(), y.copy() # Pas de test set

    if validation_size > 0 and len(X_train_val) > 0 :
         # Calcule la taille de validation par rapport à l'ensemble restant (train+val)
         val_split_ratio = validation_size / (1.0 - test_size) if (1.0 - test_size) > 0 else validation_size
         if val_split_ratio >= 1.0:
              print("Attention: validation_size trop grand par rapport à test_size, pas assez de données pour l'entraînement.")
              data_split['X_train'] = np.array([])
              data_split['y_train'] = np.array([])
              data_split['X_val'] = X_train_val
              data_split['y_val'] = y_train_val
         else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_split_ratio, # Taille relative à X_train_val
                random_state=random_state,
                stratify=y_train_val
            )
            data_split['X_train'] = X_train
            data_split['y_train'] = y_train
            data_split['X_val'] = X_val
            data_split['y_val'] = y_val
            print(f"Taille Entraînement: {len(X_train)} échantillons")
            print(f"Taille Validation: {len(X_val)} échantillons")

    else: # Pas de validation set séparé
         data_split['X_train'] = X_train_val
         data_split['y_train'] = y_train_val
         print(f"Taille Entraînement: {len(X_train_val)} échantillons")
         # Pas de X_val, y_val dans le dictionnaire

    return data_split


def split_data_fixed_per_subject(
    X: np.ndarray,
    y: np.ndarray,
    n_train_per_class: int,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divise les données en s'assurant qu'il y a exactement n_train_per_class
    échantillons par classe dans l'ensemble d'entraînement.

    Args:
        X: Données image.
        y: Labels entiers.
        n_train_per_class: Nombre d'échantillons à mettre dans l'entraînement pour chaque classe.
        random_state: Seed pour la reproductibilité du mélange.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Division personnalisée : {n_train_per_class} échantillons d'entraînement par classe.")
    if random_state is not None:
        np.random.seed(random_state)

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    unique_labels, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if n_train_per_class > min_count:
        print(f"Erreur: n_train_per_class ({n_train_per_class}) est supérieur au nombre minimum "
              f"d'échantillons pour une classe ({min_count}). Ajustez n_train_per_class.")
        # Retourne des tableaux vides pour indiquer l'échec
        return np.array([]), np.array([]), np.array([]), np.array([])
    elif n_train_per_class == min_count:
        print(f"Attention: n_train_per_class ({n_train_per_class}) est égal au nombre minimum "
              f"d'échantillons. Certaines classes n'auront AUCUN échantillon de test.")


    for label_encoded in unique_labels:
        indices = np.where(y == label_encoded)[0]
        current_count = len(indices)

        if current_count < n_train_per_class:
             # Ne devrait pas arriver si la vérification ci-dessus est passée, mais sécurité
             print(f"Erreur interne: La classe {label_encoded} a moins d'échantillons ({current_count}) que n_train_per_class ({n_train_per_class}).")
             continue

        # Mélanger les indices pour cette classe
        np.random.shuffle(indices)

        # Sélectionner pour entraînement et test
        train_indices = indices[:n_train_per_class]
        test_indices = indices[n_train_per_class:] # Peut être vide si current_count == n_train_per_class

        # Ajouter aux listes
        X_train_list.append(X[train_indices])
        y_train_list.append(y[train_indices])
        if len(test_indices) > 0:
            X_test_list.append(X[test_indices])
            y_test_list.append(y[test_indices])

    # Concaténer les listes
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    if not X_test_list: # Si aucune classe n'avait d'échantillon de test
         X_test, y_test = np.array([]), np.array([])
         print("Attention: Aucun échantillon de test n'a été généré.")
    else:
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)


    # Mélanger les ensembles finaux (important car la concaténation regroupe par classe)
    train_perm = np.random.permutation(len(X_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    if len(X_test) > 0:
        test_perm = np.random.permutation(len(X_test))
        X_test = X_test[test_perm]
        y_test = y_test[test_perm]

    print(f"Division terminée. Entraînement: {len(X_train)} échantillons, Test: {len(X_test)} échantillons.")
    return X_train, X_test, y_train, y_test
