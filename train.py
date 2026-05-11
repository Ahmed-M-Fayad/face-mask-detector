# =============================================================================
# train.py
# Handles all training logic — data loading, augmentation, training, saving.
# Run this once to produce the .h5 model file, then use detector.py for live use.
# =============================================================================

import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from config import IMG_SIZE, BATCH_SIZE, EPOCHS, TRAIN_DIR, TEST_DIR


def get_data_generators():
    """
    Creates and returns the training and test data generators.

    ImageDataGenerator does two things:
      1. Augmentation (training only): artificially varies the images so the
         model sees more diversity without needing more data.
      2. Normalization: rescales pixel values from [0,255] to [0.0,1.0].

    Augmentation techniques applied:
      - rescale: normalize pixel values
      - shear_range: slight shearing/slanting distortion
      - zoom_range: random zoom in/out up to 20%
      - horizontal_flip: randomly mirror the image left-right
        (a masked face flipped is still a masked face)

    Test generator only normalizes — no augmentation on test data,
    because we want to evaluate on "real" images.

    Returns:
        training_set, test_set: Keras DirectoryIterator objects
    """

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # flow_from_directory: reads images from disk, auto-labels them based on
    # subfolder names (with_mask → 0, without_mask → 1, alphabetical order).
    # target_size: resizes all images to (150, 150) on the fly.
    # class_mode='binary': produces labels as 0 or 1 (not one-hot vectors).
    training_set = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_set = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return training_set, test_set


def run_training(model, save_path):
    """
    Trains the given model on the mask dataset and saves weights to save_path.

    Args:
        model:     compiled Keras model (from model.py, model_mobilenet.py, or model_vgg16.py)
        save_path: string path where the trained model will be saved (e.g. 'mymodel.h5')
    """

    # Validate that train/test directories exist
    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(
            f"Training directory '{TRAIN_DIR}' not found. "
            "Make sure your train/ folder is in the project root."
        )
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(
            f"Test directory '{TEST_DIR}' not found. "
            "Make sure your test/ folder is in the project root."
        )

    training_set, test_set = get_data_generators()

    print(f"\n--- Training started ---")
    print(f"Classes: {training_set.class_indices}")  # shows which folder maps to 0/1
    print(f"Training samples: {training_set.samples}")
    print(f"Test samples:     {test_set.samples}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}\n")

    # model.fit: the main training loop.
    # validation_data: evaluates on test set at the end of each epoch.
    history = model.fit(
        training_set,
        epochs=EPOCHS,
        validation_data=test_set,
        verbose=1
    )

    # Save the trained model (architecture + weights) to disk
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    # Print final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final Training Accuracy:   {final_train_acc:.2%}")
    print(f"Final Validation Accuracy: {final_val_acc:.2%}")

    return history


# =============================================================================
# Entry point — run directly: python train.py --model custom
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a face mask detection model.')
    parser.add_argument(
        '--model',
        choices=['custom', 'mobilenet', 'vgg16'],
        default='custom',
        help='Which model architecture to train (default: custom)'
    )
    args = parser.parse_args()

    # Import and build the chosen model
    if args.model == 'custom':
        from model import build_custom_cnn
        model = build_custom_cnn()
        save_path = 'mymodel_custom.h5'

    elif args.model == 'mobilenet':
        from model_mobilenet import build_mobilenet
        model = build_mobilenet()
        save_path = 'mymodel_mobilenet.h5'

    elif args.model == 'vgg16':
        from model_vgg16 import build_vgg16
        model = build_vgg16()
        save_path = 'mymodel_vgg16.h5'

    model.summary()
    run_training(model, save_path)
