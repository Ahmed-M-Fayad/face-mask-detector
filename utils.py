# =============================================================================
# utils.py
# Shared helper functions used by both train.py and detector.py.
# Keeps reusable logic in one place — if you need to change preprocessing,
# you change it here and both training and detection stay consistent.
# =============================================================================

import cv2
import numpy as np
import datetime
from config import IMG_SIZE, CONFIDENCE_THRESHOLD


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_face(face_img):
    """
    Takes a raw face crop from the webcam/image and prepares it for the model.

    This replaces the original temp.jpg trick — no disk I/O, all in memory.

    Steps:
      1. Resize to model's expected input size (150x150)
      2. Convert BGR (OpenCV default) to RGB (what the model was trained on)
      3. Apply histogram equalization for better lighting robustness
      4. Normalize pixel values from [0, 255] to [0.0, 1.0]
      5. Add batch dimension (model expects shape: (1, 150, 150, 3))

    Args:
        face_img: numpy array, the cropped face region from OpenCV (BGR format)

    Returns:
        numpy array of shape (1, 150, 150, 3), ready to feed into the model
    """

    # Step 1: Resize
    face = cv2.resize(face_img, IMG_SIZE)

    # Step 2: BGR → RGB
    # OpenCV reads images in Blue-Green-Red order.
    # The model was trained on RGB images (via Keras ImageDataGenerator).
    # Mismatch here would hurt prediction quality.
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Step 3: Histogram equalization on the luminance channel
    # Convert to YCrCb color space — Y is the luminance (brightness) channel.
    # Equalizing Y redistributes brightness values so the image uses the full
    # 0-255 range. This helps in low light or overexposed conditions.
    face_ycrcb = cv2.cvtColor(face, cv2.COLOR_RGB2YCrCb)
    face_ycrcb[:, :, 0] = cv2.equalizeHist(face_ycrcb[:, :, 0])
    face = cv2.cvtColor(face_ycrcb, cv2.COLOR_YCrCb2RGB)

    # Step 4: Normalize to [0.0, 1.0]
    # Neural networks work better with small input values.
    # Dividing by 255 maps the [0, 255] integer range to [0.0, 1.0] floats.
    face = face / 255.0

    # Step 5: Add batch dimension
    # Model.predict() expects input shape (batch_size, H, W, C).
    # np.expand_dims adds a dimension at position 0: (150,150,3) → (1,150,150,3)
    face = np.expand_dims(face, axis=0)

    return face


def preprocess_for_training(image_array):
    """
    Lighter preprocessing for training-time augmentation pipeline.
    The ImageDataGenerator handles most of this — this is just normalization.
    """
    return image_array / 255.0


# =============================================================================
# DRAWING / DISPLAY
# =============================================================================

def draw_prediction(img, x, y, w, h, pred):
    """
    Draws a bounding box and label on the frame for one detected face.

    Args:
        img:  the full video frame (numpy array, will be modified in place)
        x, y: top-left corner of the detected face rectangle
        w, h: width and height of the detected face rectangle
        pred: model output — float between 0.0 and 1.0
              < CONFIDENCE_THRESHOLD → MASK
              >= CONFIDENCE_THRESHOLD → NO MASK
    """

    if pred >= CONFIDENCE_THRESHOLD:
        # NO MASK — red box
        color = (0, 0, 255)
        confidence = pred                          # confidence of NO MASK
        label = f'NO MASK ({confidence:.0%})'
    else:
        # MASK — green box
        color = (0, 255, 0)
        confidence = 1 - pred                     # confidence of MASK
        label = f'MASK ({confidence:.0%})'

    # Draw rectangle around the face
    # Arguments: image, top-left corner, bottom-right corner, color (BGR), thickness
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

    # Put label text just above the rectangle (y - 10 moves it up)
    cv2.putText(
        img,
        label,
        (x, y - 10),                             # position: above box
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,                                      # font scale
        color,
        2                                         # thickness
    )


def draw_timestamp(img):
    """
    Draws the current date and time in the bottom-right corner of the frame.
    Matches the behavior of the original code.
    """
    timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    cv2.putText(
        img,
        timestamp,
        (10, img.shape[0] - 10),                 # bottom-left area
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


def draw_stats(img, total, masked, unmasked):
    """
    Draws a small summary counter in the top-left corner.
    Shows: total faces, masked count, unmasked count.

    Args:
        img:      the video frame
        total:    total number of faces detected this frame
        masked:   number of masked faces
        unmasked: number of unmasked faces
    """
    stats = f'Faces: {total}  |  Masked: {masked}  |  No Mask: {unmasked}'
    cv2.putText(
        img,
        stats,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),   # yellow
        2
    )
