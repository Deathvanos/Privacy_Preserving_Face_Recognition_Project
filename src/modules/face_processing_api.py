from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import base64
import os
from collections import defaultdict
from tqdm import tqdm
import logging
from sklearn.decomposition import PCA
from src.config import IMAGE_SIZE

# Configuration Flask
app = Flask(__name__)

# Configuration logging
log_path = os.path.join("logs", "face_pipeline_debug.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
SHOW_DIR = "show_test_subject"
os.makedirs(SHOW_DIR, exist_ok=True)

# === MODULES ===

# ✅ Les images 'resized_image' et 'grayscale_image' peuvent être facilement affichées sur une GUI
# en les encodant en base64 ou affichées directement via matplotlib ou PIL

def preprocess_image(img: Image.Image, resize_size=(100, 100)) -> dict:
    img_resized = img.resize(resize_size)
    img_gray = img_resized.convert("L")
    img_array = np.array(img_gray) / 255.0
    flattened = img_array.flatten()
    return {
        "resized_image": img_resized,
        "grayscale_image": img_gray,
        "normalized_image": img_array,
        "flattened_image": flattened
    }

# ✅ Les 'eigenfaces' et 'mean_face' sont des images 2D normalisées et peuvent être affichées directement dans une GUI
# après transformation inverse ou encodage base64

def compute_eigenfaces(images_flattened, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(images_flattened)
    eigenfaces = pca.components_.reshape((n_components, *IMAGE_SIZE))
    mean_face = pca.mean_.reshape(IMAGE_SIZE)
    return pca, eigenfaces, mean_face

# ℹ️ Cette fonction retourne une matrice de données projetées avec bruit, utile pour visualisation indirecte (ex : valeurs projetées)
# mais pas directement affichable comme une image dans une GUI

def add_differential_noise(data, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

# ✅ Les images reconstruites peuvent être directement affichées dans une interface utilisateur graphique
# après éventuelle normalisation et encodage base64

def reconstruct_images(pca, projections):
    reconstructions = pca.inverse_transform(projections)
    return [img.reshape(IMAGE_SIZE) for img in reconstructions]

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def array_to_base64(img_array: np.ndarray) -> str:
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    return pil_to_base64(img)

def save_show_images(subject_id, images):
    for i, b64img in enumerate(images):
        img_bytes = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(SHOW_DIR, f"subject_{subject_id}_image_{i}.jpg"))

# === PIPELINE ===
pipeline_result = {}

@app.route("/pipeline", methods=["POST"])
def run_pipeline():
    global pipeline_result

    if "folder" not in request.form:
        return jsonify({"error": "Le champ 'folder' est requis pour traiter les sujets."}), 400

    folder_path = os.path.abspath(request.form.get("folder"))
    if not os.path.isdir(folder_path):
        return jsonify({"error": f"Le dossier '{folder_path}' est introuvable."}), 400

    epsilon = float(request.form.get("epsilon", 9.0))
    n_components_ratio = float(request.form.get("n_components_ratio", 0.9))

    logger.debug(f"Chargement des images depuis : {folder_path}")
    logger.debug(f"Paramètres - epsilon: {epsilon}, n_components_ratio: {n_components_ratio}")

    image_groups = defaultdict(list)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for filename in tqdm(image_files, desc="Prétraitement des images"):
        try:
            parts = filename.split("_")
            if len(parts) < 3:
                continue
            subject_id = parts[1]
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert("RGB")
            preprocessed = preprocess_image(img, resize_size=IMAGE_SIZE)
            if preprocessed and preprocessed['flattened_image'] is not None:
                image_groups[subject_id].append(preprocessed)
        except Exception as e:
            logger.warning(f"Erreur lors du traitement de {filename}: {e}")
            continue

    pipeline_result.clear()
    saved = False
    sorted_subjects = sorted(image_groups.keys())
    selected_subject = sorted_subjects[0] if sorted_subjects else None

    for subject_id in tqdm(image_groups, desc="Traitement par sujet"):
        images = image_groups[subject_id]
        if len(images) < 2:
            logger.info(f"Sujet {subject_id} ignoré (moins de 2 images)")
            continue

        logger.debug(f"Traitement du sujet {subject_id} avec {len(images)} images")
        flattened_stack = np.array([img['flattened_image'] for img in images])
        n_components = min(int(n_components_ratio * flattened_stack.shape[0]), flattened_stack.shape[1])

        pca, eigenfaces, mean_face = compute_eigenfaces(flattened_stack, n_components)
        projection = pca.transform(flattened_stack)
        projection_noisy = add_differential_noise(projection, epsilon)
        reconstructed = reconstruct_images(pca, projection_noisy)

        encoded_recon = [array_to_base64(img) for img in reconstructed]

        pipeline_result[subject_id] = {
            "resized": [pil_to_base64(img['resized_image']) for img in images],
            "grayscale": [pil_to_base64(img['grayscale_image']) for img in images],
            "normalized": [img['normalized_image'].tolist() for img in images],
            "flattened": flattened_stack.tolist(),
            "eigenfaces": [array_to_base64(face) for face in eigenfaces],
            "mean_face": array_to_base64(mean_face),
            "projection": projection.tolist(),
            "noised_projection": projection_noisy.tolist(),
            "reconstructed": encoded_recon
        }

        if not saved and subject_id == selected_subject:
            logger.info(f"Enregistrement du sujet exemple : {subject_id}")
            save_show_images(subject_id, encoded_recon)
            saved = True

        logger.debug(f"Sujet {subject_id} traité avec succès.")

    return jsonify({"message": f"Pipeline exécuté pour {len(pipeline_result)} sujets."})

@app.route("/result/<subject>/<step>", methods=["GET"])
def get_step(subject, step):
    if subject not in pipeline_result:
        return jsonify({"error": f"Sujet '{subject}' introuvable."}), 404
    if step not in pipeline_result[subject]:
        return jsonify({"error": f"'{step}' n'est pas une sortie valide pour le sujet '{subject}'."}), 404
    return jsonify({step: pipeline_result[subject][step]})

if __name__ == "__main__":
    app.run(debug=False)
