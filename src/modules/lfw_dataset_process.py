import os
import io
import base64
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people

import anony_process_pipeline

RECONSTRUCTED_DIR = "../../data/reconstructed"
os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)

def new_save_reconstructed_images(subject_id, images):
    """
    Enregistre le lot d'images reconstruites pour un sujet donné dans le dossier RECONSTRUCTED_DIR.
    Chaque image est enregistrée au format PNG et nommée sous la forme "reconstructed_<id_subject>_<num_img>.png".
    """
    for i, b64img in enumerate(images):
        img_bytes = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(img_bytes))
        file_name = f"reconstructed_{subject_id}_{i}.png"
        file_path = os.path.join(RECONSTRUCTED_DIR, file_name)
        img.save(file_path)
    print(f"Images reconstruites enregistrées pour le sujet {subject_id} dans {RECONSTRUCTED_DIR}")

anony_process_pipeline.save_reconstructed_images = new_save_reconstructed_images

def load_lfw_dataframe(min_faces_per_person: int = 20, n_samples_per_person: int = 20) -> pd.DataFrame:
    """
    Charge et équilibre le dataset LFW People pour obtenir exactement n_samples_per_person images par personne.
    Le DataFrame résultant contient les colonnes suivantes :
      - userFaces: Image PIL en niveaux de gris
      - imageId: Identifiant unique (index du DataFrame)
      - subject_number: Identifiant numérique du sujet
    Returns:
        pd.DataFrame: DataFrame prêt à être utilisé par la pipeline.
    """
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    height, width = lfw_people.images.shape[1], lfw_people.images.shape[2]

    data = lfw_people.data
    if data.max() <= 1.0:
        data = (data * 255).astype(np.uint8)
    else:
        data = data.astype(np.uint8)

    df = pd.DataFrame(data)
    df['subject_number'] = lfw_people.target
    df['target_names'] = df['subject_number'].apply(lambda x: lfw_people.target_names[x])

    grouped = df.groupby('target_names')
    balanced_dfs = []
    for name, group in tqdm(grouped, desc="Sampling each person"):
        if len(group) >= n_samples_per_person:
            sampled_group = group.sample(n=n_samples_per_person, random_state=42)
            balanced_dfs.append(sampled_group)
    df_balanced = pd.concat(balanced_dfs)

    def row_to_image(row):
        pixel_values = row.iloc[:height * width].values.astype(np.uint8)
        img_array = pixel_values.reshape((height, width))
        return Image.fromarray(img_array, mode='L')

    df_balanced['userFaces'] = df_balanced.apply(row_to_image, axis=1)
    df_balanced['imageId'] = df_balanced.index

    columns_to_drop = list(range(height * width)) + ['target_names']
    df_balanced.drop(columns=columns_to_drop, inplace=True)

    return df_balanced

def main():
    df = load_lfw_dataframe(min_faces_per_person=20, n_samples_per_person=20)
    print(f"DataFrame chargé avec {df.shape[0]} images réparties sur {df['subject_number'].nunique()} sujets.")

    pipeline_result = anony_process_pipeline.run_pipeline(df_images=df, epsilon=0.68, n_components_ratio=0.85)
    print("Traitement terminé. Les images reconstruites ont été enregistrées dans le dossier 'reconstructed'.")

if __name__ == '__main__':
    main()
