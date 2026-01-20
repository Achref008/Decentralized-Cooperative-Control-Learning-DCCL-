# src/model.py
"""
Neural network model definitions and hyperparameter tuning utilities.

This module provides:
- A flexible feedforward (or optional sequence) model for PID prediction,
  with optional auxiliary classification output.
- A Hyperband-based tuning routine using Keras Tuner.

Design principles:
- Explicit interfaces and minimal hidden behavior.
- Deterministic layer naming.
- Clear separation between model construction and tuning logic.
"""

import logging
from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt

logger = logging.getLogger(__name__)


# ================================================================
# MODEL BUILDER
# ================================================================

def create_model(
    hp: kt.HyperParameters,
    input_shape: Tuple[int, ...],
    use_sequence: bool = False,
    multi_output: bool = False,
) -> tf.keras.Model:
    """
    Construct a configurable neural network for PID prediction.

    Args:
        hp: Keras Tuner HyperParameters object.
        input_shape: Feature shape, either (features,) or (timesteps, features).
        use_sequence: If True, use an LSTM front-end for sequential input.
        multi_output: If True, add an auxiliary classification head.

    Returns:
        Compiled Keras model.

    Raises:
        AssertionError: If input shape is incompatible with multi-output mode.
    """
    if multi_output:
        assert input_shape[-1] >= 3, (
            "multi_output=True requires at least 3 features in the last dimension."
        )

    # ---------------- Input block ----------------

    if use_sequence:
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.LSTM(128, return_sequences=False, name="lstm_encoder")(inputs)
    else:
        inputs = layers.Input(shape=input_shape, name="flat_input")
        x = inputs

    # ---------------- Hidden backbone ----------------

    activation = hp.Choice(
        "activation",
        values=["relu", "swish", "tanh", "elu"],
        default="relu",
    )

    num_layers = hp.Int("num_layers", min_value=4, max_value=8, step=1)

    skip_tensors = []

    for i in range(num_layers):
        units = hp.Int(f"units_{i}", min_value=128, max_value=512, step=32)

        x = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            name=f"dense_{i}",
        )(x)

        if hp.Boolean(f"batch_norm_{i}"):
            x = layers.BatchNormalization(name=f"bn_{i}")(x)

        if hp.Boolean(f"dropout_{i}"):
            rate = hp.Float(f"dropout_rate_{i}", 0.0, 0.3, step=0.1)
            x = layers.Dropout(rate, name=f"dropout_{i}")(x)

        if hp.Boolean(f"use_skip_{i}"):
            skip_tensors.append(x)

    # ---------------- Skip / residual connections ----------------

    if skip_tensors:
        try:
            # Check if all skip tensors share the same last dimension
            dims = [tf.keras.backend.int_shape(t)[-1] for t in skip_tensors + [x]]

            if all(d == dims[0] for d in dims):
                x = layers.Add(name="skip_add")(skip_tensors + [x])
            else:
                x = layers.Concatenate(name="skip_concat")(skip_tensors + [x])

        except Exception as e:
            logger.warning(
                "Skip connections disabled due to shape mismatch: %s", str(e)
            )

    # ---------------- Output heads ----------------

    # Primary PID regression head (3 outputs)
    pid_output = layers.Dense(3, activation="relu", name="pid_output")(x)

    if multi_output:
        # Auxiliary classification head (4 classes)
        class_output = layers.Dense(
            4, activation="softmax", name="system_class"
        )(x)

        model = models.Model(
            inputs=inputs, outputs=[pid_output, class_output]
        )

        learning_rate = hp.Float(
            "learning_rate", 1e-4, 1e-2, sampling="log"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss={
                "pid_output": "mse",
                "system_class": "categorical_crossentropy",
            },
            metrics={
                "pid_output": ["mae", "mse"],
                "system_class": ["accuracy"],
            },
        )

    else:
        # Single-output regression model
        loss_choice = hp.Choice("loss_function", ["huber", "mse"])
        loss_fn = (
            tf.keras.losses.Huber(delta=1.0)
            if loss_choice == "huber"
            else "mse"
        )

        model = models.Model(inputs=inputs, outputs=pid_output)

        learning_rate = hp.Float(
            "learning_rate", 1e-4, 1e-2, sampling="log"
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=["mae", "mse"],
        )

    return model


# ================================================================
# HYPERPARAMETER TUNING
# ================================================================

def run_tuning(
    X_raw: Union[tf.Tensor, tf.data.Dataset],
    y_raw: Union[tf.Tensor, tuple],
    use_sequence: bool = False,
    multi_output: bool = False,
    max_epochs: int = 30,
    project_name: str = "pid_flexible_model",
) -> tf.keras.Model:
    """
    Run Hyperband tuning and return the best model.

    Args:
        X_raw: Training features.
        y_raw: Targets (single array or tuple for multi-output).
        use_sequence: Whether inputs are sequential.
        multi_output: Whether to tune a multi-output model.
        max_epochs: Upper bound on training epochs for Hyperband.
        project_name: Directory name for tuner artifacts.

    Returns:
        Best compiled Keras model according to validation loss.
    """
    input_shape = tuple(X_raw.shape[1:])

    def model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
        return create_model(
            hp,
            input_shape=input_shape,
            use_sequence=use_sequence,
            multi_output=multi_output,
        )

    tuner = kt.Hyperband(
        model_builder,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        directory="tuner_logs",
        project_name=project_name,
        overwrite=True,
    )

    tuner.search(
        X_raw,
        y_raw,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            ),
        ],
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    return create_model(
        best_hp,
        input_shape=input_shape,
        use_sequence=use_sequence,
        multi_output=multi_output,
    )
