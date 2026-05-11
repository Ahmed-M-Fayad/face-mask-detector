# =============================================================================
# model.py
# Defines the custom CNN architecture (enhanced version of the original).
# Only responsible for building and compiling the model — nothing else.
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from config import IMG_SIZE, IMG_CHANNELS


def build_custom_cnn():
    """
    Builds and returns a compiled custom CNN model.

    Architecture improvements over the original:
    - BatchNormalization after each Conv block: stabilizes training,
      allows higher learning rates, reduces sensitivity to weight init.
    - Dropout(0.5) before final Dense: randomly zeros 50% of neurons
      during training to prevent overfitting on the small dataset.
    - The original imported SpatialDropout2D and Dropout but never used them.
      We now use Dropout properly.

    Returns:
        model: compiled Keras Sequential model
    """

    input_shape = (IMG_SIZE[0], IMG_SIZE[1], IMG_CHANNELS)  # (150, 150, 3)

    model = Sequential([

        # --- Block 1 ---
        # Conv2D: 32 filters, each 3x3 pixels, scans the image for basic features
        # like edges and color transitions. ReLU activation kills negative values
        # (keeps only "positive signals").
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # BatchNormalization: normalizes the outputs of the layer above so they
        # have mean~0 and std~1. Makes training more stable.
        BatchNormalization(),
        # MaxPooling2D: takes the maximum value in each 2x2 region, shrinking
        # the feature map by half. Reduces computation and adds position invariance.
        MaxPooling2D(),

        # --- Block 2 ---
        # Same structure — but now learning more complex combinations of features
        # from Block 1's output (e.g. shapes, texture patterns).
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        # --- Block 3 ---
        # Final conv block — learning high-level features like "covered mouth area"
        # vs "visible mouth/nose".
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        # --- Classifier Head ---
        # Flatten: converts the 3D feature maps into a 1D vector so Dense layers can use it.
        Flatten(),

        # Dense(100): fully connected layer — combines all features into a decision.
        Dense(100, activation='relu'),

        # Dropout(0.5): during training only, randomly turns off 50% of neurons.
        # Forces the network not to rely on any single neuron — improves generalization.
        Dropout(0.5),

        # Output layer: single neuron with sigmoid.
        # Output < 0.5  → MASK (class 0)
        # Output >= 0.5 → NO MASK (class 1)
        Dense(1, activation='sigmoid')
    ])

    # binary_crossentropy: standard loss for binary classification (0 or 1 output).
    # adam: adaptive learning rate optimizer — good default for most problems.
    # accuracy: the metric we track during training.
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
