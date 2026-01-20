# src/transfer_learning.py
"""
Pretraining and model reuse utilities.

This module provides:
- Optional centralized pretraining on the full dataset.
- Deterministic saving of a base model and its weights.
- A safe loader that can:
    - reuse the pretrained model when input shapes match, or
    - rebuild a compatible model and transfer weights when shapes differ.

Design principles:
- No side effects at import time.
- Explicit file names for artifacts.
- Clear separation between model creation and weight transfer.
"""

import os
from typing import Optional, Tuple

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import EPOCHS, BATCH_SIZE, LEARNING_RATE
from model import create_model

# ------------------------------------------------------------------
# Artifact locations (relative to project root)
# ------------------------------------------------------------------

PRETRAINED_MODEL_PATH = "artifacts/pretrained_model.keras"
PRETRAINED_WEIGHTS_PATH = "artifacts/base_model_weights.h5"


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _ensure_artifact_dir() -> None:
    """Create the artifacts directory if it does not exist."""
    os.makedirs(os.path.dirname(PRETRAINED_MODEL_PATH), exist_ok=True)


def _select_device() -> str:
    """Return a suitable TensorFlow device string."""
    return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"


# ------------------------------------------------------------------
# Pretraining
# ------------------------------------------------------------------

def pretrain_on_full_data(
    X,
    y,
    input_shape: Tuple[int, ...],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    callbacks: Optional[list] = None,
) -> tf.keras.Model:
    """
    Pretrain a base model on the full dataset.

    This is optional and is intended to provide a better initialization
    for decentralized training.

    Args:
        X: Feature matrix (NumPy array or Tensor).
        y: Targets (NumPy array or Tensor).
        input_shape: Shape of one input sample (excluding batch dimension).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        callbacks: Optional list of Keras callbacks.

    Returns:
        Trained Keras model.
    """
    _ensure_artifact_dir()
    device = _select_device()

    with tf.device(device):
        # Fixed hyperparameters for reproducible pretraining
        hp = kt.HyperParameters()
        hp.Fixed("activation", "relu")
        hp.Fixed("num_layers", 6)
        hp.Fixed("units_0", 256)
        hp.Fixed("units_1", 256)
        hp.Fixed("units_2", 256)
        hp.Fixed("units_3", 128)
        hp.Fixed("units_4", 128)
        hp.Fixed("units_5", 64)
        hp.Fixed("loss_function", "huber")
        hp.Fixed("learning_rate", LEARNING_RATE)

        model = create_model(hp, input_shape=input_shape)

        # Sensible defaults if none are provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                ),
            ]

        model.fit(
            X,
            y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Persist both model and weights for later reuse
        model.save(PRETRAINED_MODEL_PATH)
        model.save_weights(PRETRAINED_WEIGHTS_PATH)

    return model


# ------------------------------------------------------------------
# Loading and weight transfer
# ------------------------------------------------------------------

def load_pretrained_model(
    input_shape: Optional[Tuple[int, ...]] = None,
    freeze_layers: bool = False,
) -> tf.keras.Model:
    """
    Load a pretrained model from disk.

    Behavior:
    1) If no pretrained model exists, build a fresh model.
    2) If a model exists and input_shape matches, load it directly.
    3) If input_shape differs, rebuild a compatible model and
       transfer weights where possible.

    Args:
        input_shape: Desired input shape for the current node.
        freeze_layers: If True, freeze all layers after loading.

    Returns:
        Compiled Keras model.
    """
    device = _select_device()

    with tf.device(device):
        # Case 1: No pretrained model available
        if not os.path.exists(PRETRAINED_MODEL_PATH):
            hp = kt.HyperParameters()
            hp.Fixed("activation", "relu")
            hp.Fixed("num_layers", 4)
            hp.Fixed("units_0", 128)
            hp.Fixed("units_1", 128)
            hp.Fixed("units_2", 128)
            hp.Fixed("units_3", 128)
            hp.Fixed("loss_function", "huber")
            hp.Fixed("learning_rate", LEARNING_RATE)

            model = create_model(hp, input_shape=input_shape)

        else:
            # Load existing model without compiling (we recompile below)
            base_model = tf.keras.models.load_model(
                PRETRAINED_MODEL_PATH, compile=False
            )

            # Case 2: Shapes match → reuse model directly
            if input_shape is None or input_shape == base_model.input_shape[1:]:
                model = base_model

            # Case 3: Shapes differ → rebuild and transfer weights
            else:
                hp = kt.HyperParameters()
                hp.Fixed("activation", "relu")
                hp.Fixed("num_layers", 4)
                hp.Fixed("units_0", 128)
                hp.Fixed("units_1", 128)
                hp.Fixed("units_2", 128)
                hp.Fixed("units_3", 128)
                hp.Fixed("loss_function", "huber")
                hp.Fixed("learning_rate", LEARNING_RATE)

                model = create_model(hp, input_shape=input_shape)

                try:
                    model.set_weights(base_model.get_weights())
                except Exception:
                    # If shapes are incompatible, continue with random initialization
                    pass

        # Optional freezing
        if freeze_layers:
            for layer in model.layers:
                layer.trainable = False

        # Final compilation (consistent across all cases)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=["mae", "mse"],
        )

    return model
