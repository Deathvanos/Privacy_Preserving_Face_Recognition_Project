
import os
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Import des modules du projet ---
try:
    from ..modules import data_loader
    try:
        from .. import config
    except ImportError:
        print("Erreur: Impossible d'importer src/config.py.")
        exit()
except ImportError as e:
    print(f"Erreur d'importation des modules du projet : {e}")
    exit()

# --- Fonction Principale d'Évaluation ---
def evaluate_model():
    """
    Charge un modèle entraîné et l'évalue sur le jeu de test.
    """
    print("--- Démarrage du Script d'Évaluation ---")
    start_time = time.time()

    # --- 1. Charger Configuration et Chemins ---
    print("Chargement de la configuration...")
    model_filename = f"{config.MODEL_NAME}.h5" # ou .keras
    model_filepath = os.path.join(config.MODEL_SAVE_DIR, model_filename)
    encoder_filename = f"{config.MODEL_NAME}_label_encoder.joblib"
    encoder_filepath = os.path.join(config.MODEL_SAVE_DIR, encoder_filename)

    print(f"  - Modèle à charger: {model_filepath}")
    print(f"  - Encodeur à charger: {encoder_filepath}")
    print(f"  - Données d'évaluation depuis: {config.LFW_ANONY_PATH}")

    # --- 2. Charger Modèle et Encodeur ---
    print("\n--- Chargement du modèle et de l'encodeur ---")
    if not os.path.exists(model_filepath):
        print(f"Erreur: Fichier modèle non trouvé: {model_filepath}")
        return
    try:
        model = keras.models.load_model(model_filepath)
        print("Modèle chargé avec succès.")
        model.summary()
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Keras: {e}")
        return

    label_encoder = data_loader.load_label_encoder(encoder_filepath)
    if label_encoder is None:
        print("Erreur critique : Impossible de charger l'encodeur de labels.")
        return
    num_classes = len(label_encoder.classes_)
    print(f"{num_classes} classes trouvées dans l'encodeur.")

    # --- 3. Charger et Préparer les Données de Test ---
    print("\n--- Chargement et préparation des données de test ---")
    X_full, y_full, _ = data_loader.load_lfw_anonymized_flat(
        data_dir=config.LFW_ANONY_PATH,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        color_mode=config.COLOR_MODE
    )

    if X_full is None or y_full is None:
        print("Erreur lors du chargement des données complètes.")
        return

    X_test, y_test = None, None
    if config.SPLIT_STRATEGY == 'stratified':
        temp_splits = data_loader.split_data_stratified(
            X_full, y_full,
            test_size=config.TEST_SPLIT_RATIO,
            validation_size=config.VALIDATION_SPLIT_RATIO,
            random_state=config.RANDOM_STATE
        )
        X_test = temp_splits.get('X_test')
        y_test = temp_splits.get('y_test')

    elif config.SPLIT_STRATEGY == 'fixed_per_subject':
        _, X_test, _, y_test = data_loader.split_data_fixed_per_subject(
            X_full, y_full,
            n_train_per_class=config.N_TRAIN_PER_SUBJECT,
            random_state=config.RANDOM_STATE
        )
    else:
        print(f"Erreur: Stratégie de split '{config.SPLIT_STRATEGY}' non reconnue.")
        return

    if X_test is None or y_test is None or len(X_test) == 0:
        print("Erreur: Impossible de générer ou d'isoler le jeu de données de test.")
        return

    print(f"Jeu de test isolé : {len(X_test)} échantillons.")

    # --- 4. Évaluation sur le Jeu de Test ---
    print("\n--- Évaluation du modèle sur le jeu de test ---")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nRésultats de model.evaluate():")
    print(f"  - Perte (Loss) : {loss:.4f}")
    print(f"  - Précision (Accuracy) : {accuracy:.4f} ({accuracy*100:.2f}%)")

    # --- 5. Prédictions et Métriques Détaillées ---
    print("\n--- Génération des prédictions et métriques détaillées ---")
    y_pred_proba = model.predict(X_test)
    y_pred_indices = np.argmax(y_pred_proba, axis=1)

    y_test_original_labels = label_encoder.inverse_transform(y_test)
    y_pred_original_labels = label_encoder.inverse_transform(y_pred_indices)
    class_names = label_encoder.classes_

    # Rapport de Classification (Précision, Rappel, F1 par classe)
    print("\n--- Rapport de Classification ---")
    try:
        present_labels = sorted(list(np.unique(y_test_original_labels)))
        print(classification_report(y_test_original_labels, y_pred_original_labels, labels=present_labels, target_names=present_labels, digits=3))
    except ValueError as e:
        print(f"Erreur lors de la génération du rapport de classification : {e}")
        print("Cela peut arriver si certaines classes présentes dans l'encodeur ne sont pas dans le jeu de test.")
        try:
             print(classification_report(y_test_original_labels, y_pred_original_labels, digits=3))
        except Exception as e2:
             print(f"Échec de la deuxième tentative de rapport: {e2}")


    # Matrice de Confusion
    print("\n--- Matrice de Confusion ---")
    try:
        conf_matrix = confusion_matrix(y_test_original_labels, y_pred_original_labels, labels=present_labels)
        print(conf_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues',
                    xticklabels=present_labels, yticklabels=present_labels)
        plt.title(f'Matrice de Confusion - {config.MODEL_NAME}')
        plt.ylabel('Vrai Label (Subject ID)')
        plt.xlabel('Label Prédit (Subject ID)')
        if len(present_labels) > 20:
             plt.xticks(ticks=np.arange(len(present_labels)) + 0.5, labels=present_labels, rotation=90, fontsize=8)
             plt.yticks(ticks=np.arange(len(present_labels)) + 0.5, labels=present_labels, rotation=0, fontsize=8)
        else:
             plt.xticks(rotation=45, ha='right')
             plt.yticks(rotation=0)

        plt.tight_layout()
        plot_save_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME}_confusion_matrix.png")
        plt.savefig(plot_save_path)
        print(f"\nMatrice de confusion sauvegardée dans : {plot_save_path}")

    except Exception as e:
        print(f"Erreur lors de la génération/affichage de la matrice de confusion: {e}")


    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Script d'Évaluation Terminé en {duration:.2f} secondes ---")

# --- Point d'Entrée ---
if __name__ == "__main__":
    evaluate_model()