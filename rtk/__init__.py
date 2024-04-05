import os

__version__ = "0.0.2.dev0"

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_DIR = os.path.join(DEFAULT_CACHE_DIR, "tmp")

DEFAULT_DATA_PATH = os.path.join(DEFAULT_CACHE_DIR, "datasets")
FEATURE_KEYNAME = "text_prompts"
IMAGE_KEYNAME = "image_files"
LABEL_KEYNAME = "labels"
COLUMN_NAMES = [IMAGE_KEYNAME, LABEL_KEYNAME]

DEFAULT_MODEL_PATH = os.path.join(DEFAULT_CACHE_DIR, "models")
