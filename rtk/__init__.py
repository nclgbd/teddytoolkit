import os

__version__ = "0.0.1.dev0"

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
DEFAULT_DATA_PATH = os.path.join(DEFAULT_CACHE_DIR, "datasets")
