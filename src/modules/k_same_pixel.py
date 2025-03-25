from PIL import Image
import os
import numpy as np
import cv2
from collections import defaultdict

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
                    # Extraire le nom du sujet, ex: subject01
                    subject_id = filename.split('_')[0]
                    subject_dict[subject_id].append((img_np, filename))
            except Exception as e:
                print(f"⚠️ Erreur lecture {filename} : {e}")
    return subject_dict

def k_same_pixel(images, k=3):
    anonymized = []
    for i in range(0, len(images), k):
        group = images[i:i+k]
        if len(group) < k:
            group = group + group[:k - len(group)]  # compléter si groupe incomplet
        group_imgs = [img for img, _ in group]
        mean_face = np.mean(group_imgs, axis=0).astype(np.uint8)
        for _, name in group:
            anonymized.append((mean_face, name))
    return anonymized

def save_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img, name in images:
        save_path = os.path.join(output_folder, name)
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    input_folder = "/Users/elodiechen/PycharmProjects/Privacy_Preserving_Face_Recognition_Project/data/yalefaces"
    output_folder = "dataset/k_same_faces"
    k = 3

    subject_images = load_images_by_subject(input_folder)
    total_anonymized = []

    for subject_id, images in subject_images.items():
        if len(images) == 0:
            continue
        anonymized = k_same_pixel(images, k)
        total_anonymized.extend(anonymized)

    save_images(total_anonymized, output_folder)
    print(f"{len(total_anonymized)} images anonymisées (groupées par sujet) sauvegardées dans {output_folder}")

