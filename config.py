# =============================================================================
# config.py
# Central configuration file — all constants and paths live here.
# If you ever need to change a value, change it once here, not hunting
# through multiple files.
# =============================================================================

# --- Image settings ---
IMG_SIZE = (150, 150)       # Width x Height the model expects
IMG_CHANNELS = 3            # RGB = 3 channels

# --- Training settings ---
BATCH_SIZE = 16             # How many images to process per training step
EPOCHS = 10                 # How many full passes over the training data

# --- Paths ---
MODEL_PATH = "model_v0.h5"                              # Where the trained model is saved/loaded
CASCADE_PATH = "hear_cascade_frontal_face_default.xml"   # OpenCV face detector file
TRAIN_DIR = "train"                                    # Training images directory
TEST_DIR = "test"                                      # Test images directory

# --- Detection settings ---
SCALE_FACTOR = 1.1          # How much the face detector shrinks the image each step (Haar Cascade param)
MIN_NEIGHBORS = 4           # How many neighboring rectangles needed before declaring a face found
CONFIDENCE_THRESHOLD = 0.5  # Model output above this = NO MASK, below = MASK
