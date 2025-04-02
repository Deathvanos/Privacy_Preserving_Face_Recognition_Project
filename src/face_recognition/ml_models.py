from tensorflow.keras import layers, models, applications
from typing import Tuple, Optional

def build_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> models.Model:
    """
    Construit un modèle CNN simple pour la classification d'images.

    Args:
        input_shape: Tuple indiquant la forme des images d'entrée (height, width, channels).
                     Exemple: (64, 64, 1) pour grayscale, (64, 64, 3) pour RGB.
        num_classes: Nombre de classes de sortie (nombre de sujets uniques).

    Returns:
        Un modèle Keras compilable.
    """
    print(f"Construction du modèle CNN simple avec input_shape={input_shape} et num_classes={num_classes}")

    model_input = layers.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1_1")(model_input)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.25, name="drop1")(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.25, name="drop2")(x)

    # Partie Dense (Classification)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation='relu', name="dense1")(x)
    x = layers.BatchNormalization(name="bn_dense1")(x)
    x = layers.Dropout(0.5, name="drop_dense1")(x)

    # Couche de Sortie
    model_output = layers.Dense(num_classes, activation='softmax', name="output_softmax")(x)

    # Création du modèle final
    model = models.Model(inputs=model_input, outputs=model_output, name="simple_cnn")

    print("Modèle CNN simple construit.")
    return model


def build_transfer_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    base_model_name: str = 'MobileNetV2',
    freeze_base: bool = True
) -> Optional[models.Model]:
    """
    Construit un modèle basé sur le transfert d'apprentissage.

    Charge un modèle pré-entraîné sur ImageNet, retire sa tête de classification,
    gèle (optionnellement) ses poids, et ajoute une nouvelle tête de classification.

    Args:
        input_shape: Tuple indiquant la forme des images d'entrée (height, width, channels).
                     DOIT être compatible avec le modèle de base choisi.
        num_classes: Nombre de classes de sortie.
        base_model_name: Nom du modèle de base à charger depuis tf.keras.applications.
        freeze_base: Si True, gèle les poids du modèle de base pendant l'entraînement initial.

    Returns:
        Un modèle Keras compilable, ou None si le nom du modèle de base est invalide.
    """
    print(f"Construction du modèle de transfert avec base={base_model_name}, input_shape={input_shape}, num_classes={num_classes}")

    if input_shape[-1] != 3:
        print(f"Attention: Le modèle de base '{base_model_name}' attend typiquement 3 canaux d'entrée (RGB), "
              f"mais input_shape est {input_shape}. Assurez-vous que vos données sont chargées en RGB.")

    try:
        if base_model_name == 'MobileNetV2':
            base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        elif base_model_name == 'ResNet50':
            base_model = applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
        elif base_model_name == 'EfficientNetB0':
             base_model = applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
        else:
            print(f"Erreur: Modèle de base '{base_model_name}' non reconnu ou non implémenté ici.")
            return None
    except ValueError as e:
         print(f"Erreur lors du chargement du modèle de base '{base_model_name}' avec input_shape {input_shape}: {e}")
         print("Vérifiez que la taille d'image est compatible avec le modèle choisi (ex: MobileNetV2 >= 32x32, ResNet50 >= 32x32, EfficientNet >= 32x32).")
         return None
    except Exception as e:
         print(f"Erreur inattendue lors du chargement du modèle de base: {e}")
         return None


    print(f"Modèle de base '{base_model_name}' chargé.")

    if freeze_base:
        base_model.trainable = False
        print("Poids du modèle de base gelés.")
    else:
        base_model.trainable = True
        print("Poids du modèle de base NON gelés (mode fine-tuning).")

    model_input = base_model.input

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation='relu', name="dense_head_1")(x)
    x = layers.Dropout(0.5, name="dropout_head")(x)
    model_output = layers.Dense(num_classes, activation='softmax', name="output_softmax")(x)

    model = models.Model(inputs=model_input, outputs=model_output, name=f"transfer_{base_model_name}")

    print(f"Modèle de transfert '{model.name}' construit.")
    return model
