import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

# --- Import des modules du projet ---
try:
    from ..modules import data_loader
    from . import ml_models
    try:
        from .. import config
    except ImportError:
        print("Erreur: Impossible d'importer src/config.py.")
        print("Veuillez vous assurer que le fichier existe et ne contient pas d'erreurs de syntaxe.")
        exit()

except ImportError as e:
    print(f"Erreur d'importation des modules du projet : {e}")
    print("Assurez-vous d'exécuter ce script depuis le répertoire racine du projet ou que la structure est correcte.")
    exit()

# --- Fonction Principale d'Entraînement ---
def train_model():
    """
    Orchestre le processus complet d'entraînement du modèle de reconnaissance faciale.
    """
    print("--- Démarrage du Script d'Entraînement ---")
    start_time = time.time()

    # --- 1. Chargement de la Configuration ---
    print("Configuration chargée depuis config.py:")
    print(f"  - Dossier Données: {config.LFW_ANONY_PATH}")
    print(f"  - Dossier Sauvegarde Modèles: {config.MODEL_SAVE_DIR}")
    print(f"  - Architecture Modèle: {config.MODEL_ARCHITECTURE}")
    print(f"  - Nom Modèle: {config.MODEL_NAME}")
    print(f"  - Dimensions Image: {config.IMG_HEIGHT}x{config.IMG_WIDTH}x{config.CHANNELS}")
    print(f"  - Stratégie Split: {config.SPLIT_STRATEGY}")
    print(f"  - Époques: {config.EPOCHS}, Batch Size: {config.BATCH_SIZE}")

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    if hasattr(config, 'LOG_DIR') and config.LOG_DIR:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        print(f"  - Dossier Logs TensorBoard: {config.LOG_DIR}")


    # --- 2. Chargement et Préparation des Données ---
    print("\n--- Chargement des données ---")
    X, y, label_encoder = data_loader.load_lfw_anonymized_flat(
        data_dir=config.LFW_ANONY_PATH,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        color_mode=config.COLOR_MODE
    )

    if X is None or y is None or label_encoder is None:
        print("Erreur critique lors du chargement des données. Arrêt du script.")
        return

    num_classes = len(label_encoder.classes_)
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS)
    print(f"Nombre de classes détectées : {num_classes}")

    # --- 3. Division des Données ---
    print("\n--- Division des données ---")
    X_train, y_train = None, None
    X_val, y_val = None, None
    X_test, y_test = None, None

    if config.SPLIT_STRATEGY == 'stratified':
        data_splits = data_loader.split_data_stratified(
            X, y,
            test_size=config.TEST_SPLIT_RATIO,
            validation_size=config.VALIDATION_SPLIT_RATIO,
            random_state=config.RANDOM_STATE
        )
        X_train = data_splits.get('X_train')
        y_train = data_splits.get('y_train')
        X_val = data_splits.get('X_val')
        y_val = data_splits.get('y_val')
        X_test = data_splits.get('X_test')
        y_test = data_splits.get('y_test')

        if X_val is None and config.VALIDATION_SPLIT_RATIO > 0 and X_train is not None and len(X_train) > 0:
             val_ratio_from_train = config.VALIDATION_SPLIT_RATIO / (1.0 - config.TEST_SPLIT_RATIO)
             if val_ratio_from_train < 1.0:
                 print(f"Création du set de validation depuis l'entraînement (ratio: {val_ratio_from_train:.2f})")
                 X_train, X_val, y_train, y_val = train_test_split(
                     X_train, y_train,
                     test_size=val_ratio_from_train,
                     random_state=config.RANDOM_STATE,
                     stratify=y_train
                 )
             else:
                 print("Attention: Ratios de split incohérents, pas de données d'entraînement restantes après validation.")

    elif config.SPLIT_STRATEGY == 'fixed_per_subject':
        X_train_full, X_test, y_train_full, y_test = data_loader.split_data_fixed_per_subject(
            X, y,
            n_train_per_class=config.N_TRAIN_PER_SUBJECT,
            random_state=config.RANDOM_STATE
        )
        if config.VALIDATION_SPLIT_RATIO > 0 and X_train_full is not None and len(X_train_full) > 0:
            print(f"Création du set de validation depuis l'entraînement (ratio: {config.VALIDATION_SPLIT_RATIO})")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=config.VALIDATION_SPLIT_RATIO,
                random_state=config.RANDOM_STATE,
                stratify=y_train_full
            )
        else:
            X_train, y_train = X_train_full, y_train_full
            X_val, y_val = None, None

    else:
        print(f"Erreur: Stratégie de split '{config.SPLIT_STRATEGY}' non reconnue.")
        return

    if X_train is None or len(X_train) == 0:
        print("Erreur: Aucune donnée d'entraînement disponible après la division.")
        return
    if X_val is None or len(X_val) == 0:
        print("Attention: Aucune donnée de validation disponible. L'entraînement se fera sans suivi de validation.")
        validation_data = None # `fit` utilisera pas de validation
    else:
        validation_data = (X_val, y_val)
        print(f"Taille finale - Entraînement: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test) if X_test is not None else 0}")


    # --- 4. Construction du Modèle ---
    print("\n--- Construction du modèle ---")
    model = None
    if config.MODEL_ARCHITECTURE == 'simple_cnn':
        model = ml_models.build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    elif config.MODEL_ARCHITECTURE.startswith('transfer_'):
        base_name = getattr(config, 'TRANSFER_BASE_MODEL_NAME', 'MobileNetV2')
        freeze = getattr(config, 'TRANSFER_FREEZE_BASE', True)
        print(f"Utilisation du modèle de base: {base_name}, Freeze: {freeze}")
        model = ml_models.build_transfer_model(input_shape=input_shape,
                                               num_classes=num_classes,
                                               base_model_name=base_name,
                                               freeze_base=freeze)
    else:
        print(f"Erreur: Architecture de modèle non reconnue dans config: {config.MODEL_ARCHITECTURE}")
        return

    if model is None:
        print("Erreur critique lors de la construction du modèle. Arrêt.")
        return

    # --- 5. Compilation du Modèle ---
    print("\n--- Compilation du modèle ---")
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print("Modèle compilé avec Adam optimizer.")
    model.summary()

    # --- 6. Configuration des Callbacks ---
    print("\n--- Configuration des Callbacks ---")
    callbacks = []

    model_filename = f"{config.MODEL_NAME}.h5"
    model_filepath = os.path.join(config.MODEL_SAVE_DIR, model_filename)
    print(f"  - ModelCheckpoint: Sauvegarde du meilleur modèle dans {model_filepath}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint_callback)

    if hasattr(config, 'EARLY_STOPPING_PATIENCE') and config.EARLY_STOPPING_PATIENCE > 0:
        print(f"  - EarlyStopping: Activé avec patience={config.EARLY_STOPPING_PATIENCE}")
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
    else:
        print("  - EarlyStopping: Désactivé.")


    if hasattr(config, 'LOG_DIR') and config.LOG_DIR:
        tensorboard_log_dir = os.path.join(config.LOG_DIR, config.MODEL_NAME + "_" + time.strftime("%Y%m%d-%H%M%S"))
        print(f"  - TensorBoard: Logs dans {tensorboard_log_dir}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1
        )
        callbacks.append(tensorboard_callback)
    else:
        print("  - TensorBoard: Désactivé.")

    csv_log_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME}_training_log.csv")
    print(f"  - CSVLogger: Logs dans {csv_log_path}")
    csv_logger_callback = tf.keras.callbacks.CSVLogger(csv_log_path, append=False)
    callbacks.append(csv_logger_callback)


    # --- 7. Entraînement du Modèle ---
    print("\n--- Démarrage de l'entraînement ---")
    history = None
    try:
        history = model.fit(
            X_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        print("--- Entraînement terminé ---")

    except Exception as e:
        print(f"\nErreur pendant l'entraînement : {e}")
        encoder_save_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME}_label_encoder.joblib")
        print("\nSauvegarde de l'encodeur de labels (même si l'entraînement a échoué)...")
        data_loader.save_label_encoder(label_encoder, encoder_save_path)
        return # Arrêter le script ici


    # --- 8. Post-Entraînement ---
    encoder_save_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME}_label_encoder.joblib")
    print("\n--- Sauvegarde de l'encodeur de labels ---")
    data_loader.save_label_encoder(label_encoder, encoder_save_path)

    if history is not None:
        print("\n--- Affichage des courbes d'apprentissage ---")
        try:
            acc = history.history['accuracy']
            loss = history.history['loss']
            epochs_range = range(len(acc))

            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            if validation_data: # Seulement si validation existe
                 val_acc = history.history['val_accuracy']
                 plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            if validation_data: # Seulement si validation existe
                val_loss = history.history['val_loss']
                plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')

            plot_save_path = os.path.join(config.MODEL_SAVE_DIR, f"{config.MODEL_NAME}_training_curves.png")
            plt.savefig(plot_save_path)
            print(f"Courbes sauvegardées dans : {plot_save_path}")

        except Exception as plot_e:
            print(f"Erreur lors de la génération/sauvegarde des courbes: {plot_e}")


    end_time = time.time()
    duration = end_time - start_time
    print(f"\n--- Script d'Entraînement Terminé en {duration:.2f} secondes ---")
    print(f"Le meilleur modèle devrait être sauvegardé dans : {model_filepath}")
    print(f"L'encodeur de labels est sauvegardé dans : {encoder_save_path}")


# --- Point d'Entrée du Script ---
if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.py')
    if not os.path.exists(config_path):
         print(f"Erreur: Le fichier de configuration '{config_path}' est introuvable.")
         print("Veuillez créer ce fichier et y définir les paramètres nécessaires.")
    else:
        train_model()