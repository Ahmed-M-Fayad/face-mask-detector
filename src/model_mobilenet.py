# =============================================================================
# model_mobilenet.py
# Transfer learning using MobileNetV2 as the base model.
#
# The idea: MobileNetV2 was already trained on 1.2 million images (ImageNet).
# It already "knows" how to detect edges, textures, shapes, and objects.
# We freeze that knowledge and just train a small classifier on top of it
# to answer our specific question: mask or no mask?
#
# MobileNetV2 is designed for mobile/real-time use — lightweight and fast.
# =============================================================================

from keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.config import IMG_SIZE, IMG_CHANNELS


def build_mobilenet():
    """
    Builds a MobileNetV2-based transfer learning model.

    Strategy — Feature Extraction:
      1. Load MobileNetV2 pre-trained on ImageNet, WITHOUT its top classifier
      2. Freeze all its layers (we don't retrain them)
      3. Add our own small classifier head on top
      4. Only the head gets trained on our mask dataset

    Returns:
        model: compiled Keras Model
    """

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)  # (150, 150, 3)

    # Load MobileNetV2 without the top classification layers.
    # weights='imagenet': load the pre-trained weights from ImageNet training.
    # include_top=False: exclude the final Dense layers (we add our own).
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze the base model — we don't want to change its learned weights.
    # It will act purely as a feature extractor.
    base_model.trainable = False

    # --- Add our classifier head ---
    x = base_model.output

    # GlobalAveragePooling2D: averages each feature map into a single number.
    # More elegant than Flatten — reduces overfitting, fewer parameters.
    x = GlobalAveragePooling2D()(x)

    # Dense layer for decision making
    x = Dense(128, activation='relu')(x)

    # Dropout for regularization
    x = Dropout(0.5)(x)

    # Output: same as custom CNN — sigmoid for binary classification
    output = Dense(1, activation='sigmoid')(x)

    # Build the full model: inputs from base, outputs from our head
    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # Lower LR for transfer learning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def unfreeze_and_finetune(model, fine_tune_from=100):
    """
    Optional second phase: unfreeze the top layers of MobileNetV2
    and fine-tune them at a very low learning rate.

    This adapts the deep features slightly toward our specific domain
    (face mask images) rather than generic ImageNet patterns.

    Args:
        model: the already-trained transfer learning model
        fine_tune_from: layer index from which to unfreeze (default: layer 100 onwards)

    Returns:
        model: model with some base layers unfrozen, recompiled
    """

    # Unfreeze the base model
    model.layers[0].trainable = True  # layers[0] is the base_model

    # But freeze all layers before `fine_tune_from`
    for layer in model.layers[0].layers[:fine_tune_from]:
        layer.trainable = False

    # Recompile with a very low learning rate to avoid destroying learned weights
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # 10x lower than initial training
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
