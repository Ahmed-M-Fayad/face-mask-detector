# =============================================================================
# detector.py
# Live webcam face mask detection.
# Loads a trained model, opens the webcam, detects faces each frame,
# classifies each as MASK or NO MASK, and draws results on screen.
# Press 'q' to quit.
# =============================================================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_face, draw_prediction, draw_timestamp, draw_stats
from config import CASCADE_PATH, CONFIDENCE_THRESHOLD


def run_detection(model_path):
    """
    Runs the live webcam detection loop.

    Args:
        model_path: path to the trained .h5 model file to load
    """

    # --- Load model ---
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully.")

    # --- Load face detector ---
    # Haar Cascade is a classical (non-deep-learning) face detector.
    # It's fast and works well for frontal faces in decent lighting.
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise FileNotFoundError(
            f"Haar Cascade file not found at: {CASCADE_PATH}\n"
            "Make sure 'hear_cascade_frontal_face_default.xml' is in the project folder."
        )

    # --- Open webcam ---
    # cv2.VideoCapture(0): opens the default webcam (index 0).
    # Use index 1 or 2 if you have multiple cameras and 0 doesn't work.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. "
            "Make sure your camera is connected and not used by another app."
        )

    print("\nWebcam opened. Press 'q' to quit.\n")

    while cap.isOpened():
        # cap.read() returns:
        #   ret: bool — True if frame was read successfully
        #   frame: the actual image as a numpy array (H x W x 3, BGR)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Convert frame to grayscale for the face detector.
        # Haar Cascade works on grayscale — it's faster and the face
        # structure it looks for doesn't depend on color.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectMultiScale: scans the image at multiple scales to find faces.
        # scaleFactor=1.1: reduce image size by 10% at each scale step
        # minNeighbors=4: a face region must be confirmed by 4 neighboring
        #                 rectangles to be counted — filters out false positives
        # Returns: list of (x, y, w, h) rectangles for each detected face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4
        )

        masked_count = 0
        unmasked_count = 0

        for (x, y, w, h) in faces:
            # Crop the face region from the original color frame
            face_crop = frame[y:y + h, x:x + w]

            # Preprocess: resize, BGR→RGB, histogram eq, normalize, expand dims
            # All in memory — no temp.jpg disk write
            processed = preprocess_face(face_crop)

            # Run the model — returns float in [0.0, 1.0]
            pred = model.predict(processed, verbose=0)[0][0]

            # Draw bounding box + label on the frame
            draw_prediction(frame, x, y, w, h, pred)

            # Count for stats display
            if pred >= CONFIDENCE_THRESHOLD:
                unmasked_count += 1
            else:
                masked_count += 1

        # Draw overlays
        draw_stats(frame, len(faces), masked_count, unmasked_count)
        draw_timestamp(frame)

        # Display the frame in a window named 'Face Mask Detector'
        cv2.imshow('Face Mask Detector', frame)

        # waitKey(1): wait 1ms for a key press.
        # ord('q') returns the ASCII code of 'q'.
        # If 'q' is pressed, break the loop.
        if cv2.waitKey(1) == ord('q'):
            print("Quitting...")
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


# =============================================================================
# Entry point — run directly: python detector.py --model mymodel_custom.h5
# =============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run live face mask detection.')
    parser.add_argument(
        '--model',
        type=str,
        default='mymodel_custom.h5',
        help='Path to the trained .h5 model file (default: mymodel_custom.h5)'
    )
    args = parser.parse_args()

    run_detection(args.model)
