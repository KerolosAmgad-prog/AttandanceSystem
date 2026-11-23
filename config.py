from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.pkl"

# Camera Configuration
CAMERA_DEVICE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Model & Recognition Configuration
INSIGHTFACE_MODEL = "buffalo_sc"
DETECTION_SIZE = (320, 320)
SIMILARITY_THRESHOLD = 0.45
