import cv2
from PIL import Image
import matplotlib.pyplot as plt
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
                print(f"⚠️ Erreur lecture {filename} : {e}")
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

def save_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for img, name in images:
        save_path = os.path.join(output_folder, name)
        cv2.imwrite(save_path, img)

def show_comparison(original_images, individual_anonymized, nb=5):
    nb = min(nb, len(original_images))
    fig, axes = plt.subplots(nb, 2, figsize=(8, 2.5 * nb))
    fig.suptitle("Comparaison Original vs K-same-pixel", fontsize=14)

    for i in range(nb):
        orig_img, filename = original_images[i]
        indiv_img, _ = individual_anonymized[i]

        axes[i][0].imshow(orig_img, cmap='gray')
        axes[i][0].axis('off')
        axes[i][0].set_title(f"Original\n{filename}", fontsize=10)

        axes[i][1].imshow(indiv_img, cmap='gray')
        axes[i][1].axis('off')
        axes[i][1].set_title(f"K-Pixel\n{filename}", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    output_path = os.path.join(output_folder, 'visualization.png')
    plt.savefig(output_path, format='png')
    print(f"📸 Visualisation sauvegardée dans {output_path}")

if __name__ == "__main__":
    input_folder = "/Users/elodiechen/PycharmProjects/Privacy_Preserving_Face_Recognition_Project/data/dataset-yalefaces"
    output_folder = "/Users/elodiechen/PycharmProjects/Privacy_Preserving_Face_Recognition_Project/data/k_same_pixel_faces"
    k = 3  # nombre de voisins (ou groupe de k images)

    subject_images = load_images_by_subject(input_folder)
    total_anonymized_individual = []

    for subject_id, images in subject_images.items():
        if len(images) == 0:
            continue

        anonymized_individual = k_same_pixel_individual(images, k)
        total_anonymized_individual.extend(anonymized_individual)

    print("📁 Sauvegarde des images anonymisées individuelles...")
    save_images(total_anonymized_individual, output_folder)
    print(f"✅ {len(total_anonymized_individual)} images anonymisées individuellement enregistrées dans : {output_folder}")

    # Visualisation uniquement pour le premier sujet
    first_subject = next(iter(subject_images.values()))
    anonymized_first = k_same_pixel_individual(first_subject, k)
    show_comparison(
        original_images=first_subject,
        individual_anonymized=anonymized_first,
        nb=min(11, len(first_subject))
    )
