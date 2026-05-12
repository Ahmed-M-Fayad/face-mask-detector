# =============================================================================
# model_vgg16.py
# Transfer learning using VGG16 as the base model.
#
# VGG16 is an older, heavier architecture (2014) compared to MobileNetV2 (2018).
# It's larger and slower but very well-known academically — good for comparison.
# The point is to contrast: does a bigger, older model do better or worse
# than a lightweight modern one on this specific task?
# =============================================================================

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.config import IMG_SIZE, IMG_CHANNELS


def build_vgg16():
    """
    Builds a VGG16-based transfer learning model.

    Same strategy as MobileNetV2:
      1. Load VGG16 pre-trained on ImageNet, without top classifier
      2. Freeze all base layers
      3. Add our classifier head
      4. Train only the head

    The architecture difference from MobileNetV2:
    - VGG16 uses simple stacked Conv layers (no depthwise separable convolutions)
    - Much more parameters (~138M vs ~3.4M for MobileNetV2)
    - Slower inference but historically strong accuracy

    Returns:
        model: compiled Keras Model
    """

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)

    # Load VGG16 without top layers, with ImageNet weights
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all VGG16 layers
    base_model.trainable = False

    # --- Classifier head ---
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
