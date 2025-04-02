from PIL import Image
import os
import numpy as np
from collections import defaultdict
import random

def load_images_by_subject(folder, image_size=(100, 100)):
    subject_dict = defaultdict(list)
    print(f"Lecture des images dans : {folder}")
    for filename in os.listdir(folder):
        if filename.endswith(('.gif', '.pgm', '.png')):
            img_path = os.path.join(folder, filename)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("L")
                    img = img.resize(image_size)
                    img_np = np.array(img)
                    subject_id = filename.split('_')[0]
                    subject_dict[subject_id].append((img_np, filename))
            except Exception as e:
                print(f"⚠ Erreur lecture {filename} : {e}")
    return subject_dict

def k_same_pixel_individual(images, k=3):
    anonymized = []
    img_list = [img for img, _ in images]
    name_list = [name for _, name in images]

    for i in range(len(images)):
        base_img = img_list[i]
        indices = list(range(len(images)))
        indices.remove(i)
        if len(indices) >= k - 1:
            chosen_idx = random.sample(indices, k - 1)
        else:
            chosen_idx = (indices * ((k - 1) // len(indices) + 1))[:k - 1]

        group = [base_img] + [img_list[j] for j in chosen_idx]
        mean_face = np.mean(group, axis=0).astype(np.uint8)
        anonymized.append((mean_face, name_list[i]))

    return anonymized

