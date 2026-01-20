# src/config.py
"""
Project configuration module.

This file centralizes all configuration related to:
- Dataset locations
- Training hyperparameters
- Logging behavior
- Node communication (ZMQ ports)
- Fault-tolerance settings

Design principles:
- No hard-coded user-specific paths.
- No side effects at import time (no automatic file loading or I/O).
- All paths are relative or environment-based.
- Configuration is explicit and readable for external users.
"""

import os
import pandas as pd

# ============================
# BASE PATHS
# ============================

# Root of the repository (assumes config.py is inside src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default locations (can be overridden via environment variables)
DATA_PATH = os.getenv(
    "DCCL_DATA_PATH",
    os.path.join(BASE_DIR, "..", "data", "pid_dataset.xlsx")
)

LOG_DIR = os.getenv(
    "DCCL_LOG_DIR",
    os.path.join(BASE_DIR, "..", "logs")
)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# ============================
# DATASET UTILITIES
# ============================

def verify_dataset(file_path: str) -> pd.DataFrame:
    """
    Verify that the dataset exists and is readable.

    This function does NOT run automatically on import.
    It must be called explicitly from main or a setup script.

    Args:
        file_path: Path to the Excel dataset.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path}\n"
            "Set the environment variable DCCL_DATA_PATH to a valid file."
        )

    try:
        data = pd.read_excel(file_path)
        print(f"[INFO] Dataset loaded successfully. Shape: {data.shape}")
        print(f"[INFO] Columns: {list(data.columns)}")
        return data
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {e}")


def ensure_directories():
    """
    Create required directories (logs) if they do not exist.

    This function should be called explicitly from main.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

# ============================
# TRAINING HYPERPARAMETERS
# ============================

TEST_SIZE = 0.2           # Fraction of data used for validation
RANDOM_STATE = 42         # Reproducibility seed

BATCH_SIZE = 128
EPOCHS = 200

LEARNING_RATE = 5e-5
L2_REGULARIZATION = 5e-4
DROPOUT_RATE = 0.0005

# Number of distributed nodes/controllers
NUM_NODES = 4

# ============================
# LOGGING
# ============================

LOG_LEVEL = os.getenv("DCCL_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# ============================
# ZMQ COMMUNICATION PORTS
# ============================

NODE_PORTS = {
    0: {"subscribe": [5555, 5556], "publish": 5557},
    1: {"subscribe": [5557, 5556], "publish": 5558},
    2: {"subscribe": [5558, 5555], "publish": 5559},
    3: {"subscribe": [5559, 5555], "publish": 5560},
}

# ============================
# CONSENSUS / FAULT TOLERANCE
# ============================

FAULT_TOLERANT_NODES = 1

# Communication reliability settings
ZMQ_RETRY_LIMIT = 5
CONSENSUS_TIMEOUT = 10
