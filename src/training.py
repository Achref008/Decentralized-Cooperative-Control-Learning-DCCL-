# src/training.py
"""
Training utilities for decentralized PID parameter learning.

Responsibilities of this module:
- Local training of a single node.
- A lightweight baseline decentralized training loop
  (train → aggregate → synchronize).
- Per-node evaluation helpers.
- Optional hyperparameter tuning via Keras Tuner.

This module focuses strictly on training logic.
System orchestration lives in `main.py`.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List

import numpy as np
from tqdm import tqdm
import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import EPOCHS, PATIENCE
from consensus import ByzantineFaultTolerance
from model import create_model

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# Local node training
# ------------------------------------------------------------------

def train_node(node, epochs: int = EPOCHS) -> Optional[Dict[str, Any]]:
    """
    Train a single node on its local dataset.

    Effects:
    - Updates `node.history["loss"]` and `node.history["val_loss"]`.
    - Saves the best local checkpoint for this node.

    Args:
        node: Node instance exposing `train_dataset`, `test_dataset`, and `model`.
        epochs: Number of local training epochs.

    Returns:
        Updated history dictionary, or None if training failed.
    """
    logger.info(
        "Node %s: starting local training (%d epochs)", node.node_id, epochs
    )

    _ensure_dir("checkpoints")
    ckpt_path = os.path.join(
        "checkpoints", f"best_model_node_{node.node_id}.keras"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, PATIENCE // 3),
            min_lr=1e-6,
            verbose=0,
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=0,
        ),
    ]

    try:
        history_obj = node.model.fit(
            node.train_dataset,
            validation_data=node.test_dataset,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
        )

        # Persist history in a consistent format
        node.history["loss"] = history_obj.history.get("loss", [])
        node.history["val_loss"] = history_obj.history.get("val_loss", [])

        if node.history["val_loss"]:
            best = float(np.min(node.history["val_loss"]))
            logger.info(
                "Node %s: training complete (best val_loss=%.6f)",
                node.node_id,
                best,
            )
        else:
            logger.info(
                "Node %s: training complete (no val_loss recorded)",
                node.node_id,
            )

        return node.history

    except Exception as exc:
        logger.exception(
            "Node %s: training failed: %s", node.node_id, exc
        )
        return None


# ------------------------------------------------------------------
# Baseline decentralized training loop
# ------------------------------------------------------------------

def decentralized_training(nodes: List, num_rounds: int = 5) -> None:
    """
    Simple round-based decentralized training baseline.

    For each round:
    1) Train each node locally.
    2) Collect weights and histories.
    3) Aggregate with Byzantine-tolerant consensus.
    4) Synchronize aggregated weights to all nodes.

    This is a lightweight reference implementation.
    More advanced coordination is handled in `main.py`.
    """
    logger.info(
        "Starting decentralized training: %d nodes, %d rounds",
        len(nodes),
        num_rounds,
    )

    consensus = ByzantineFaultTolerance(
        total_nodes=len(nodes),
        fault_tolerant_nodes=1,
    )

    previous_weights = None

    for round_idx in range(num_rounds):
        logger.info("Round %d/%d", round_idx + 1, num_rounds)

        # Local training
        for node in tqdm(nodes, desc="Local training", leave=False):
            train_node(node)

        # Collect updates
        weights_list = [node.model.get_weights() for node in nodes]
        node_histories = [node.history for node in nodes]

        # Robust aggregation with optional EMA smoothing
        aggregated_weights = consensus.aggregate_weights(
            weights_list=weights_list,
            node_histories=node_histories,
            previous_weights=previous_weights,
        )

        # Keep a safe copy for next round
        previous_weights = [w.copy() for w in aggregated_weights]

        # Broadcast aggregated weights (baseline: identical for all nodes)
        for node in nodes:
            node.model.set_weights(aggregated_weights)

        # Snapshot per-node validation loss
        for node in nodes:
            val = node.history.get("val_loss", [])
            if val:
                logger.info(
                    "Node %s: latest val_loss=%.6f",
                    node.node_id,
                    float(val[-1]),
                )

    logger.info("Decentralized training finished.")


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_nodes(nodes: List) -> None:
    """
    Evaluate each node on its local validation dataset.

    Assumes models are already compiled with MAE/MSE metrics.
    """
    logger.info("Evaluating %d nodes", len(nodes))

    for node in nodes:
        if node.test_dataset is None:
            logger.warning(
                "Node %s: no test dataset; skipping evaluation",
                node.node_id,
            )
            continue

        try:
            results = node.model.evaluate(
                node.test_dataset, verbose=0
            )

            # Keras returns [loss] + metrics in order
            if isinstance(results, (list, tuple)):
                results = [float(x) for x in results]

            logger.info(
                "Node %s: evaluation results = %s",
                node.node_id,
                results,
            )

        except Exception as exc:
            logger.exception(
                "Node %s: evaluation failed: %s", node.node_id, exc
            )


# ------------------------------------------------------------------
# Hyperparameter tuning
# ------------------------------------------------------------------

def run_hyperparameter_tuning(
    train_dataset,
    val_dataset,
    input_shape,
    max_trials: int = 30,
    executions_per_trial: int = 1,
    strategy: str = "auto",
    use_sequence: bool = False,
    multi_output: bool = False,
):
    """
    Hyperparameter tuning wrapper.

    Strategy options:
    - "random": RandomSearch
    - "hyperband": Hyperband
    - "auto": choose based on dataset size heuristic

    Returns:
        Keras Tuner object (not the model).
    """

    def model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
        return create_model(
            hp=hp,
            input_shape=input_shape,
            use_sequence=use_sequence,
            multi_output=multi_output,
        )

    if strategy == "auto":
        # Rough heuristic based on data size
        train_batches = sum(1 for _ in train_dataset)
        feature_dim = (
            input_shape[-1]
            if isinstance(input_shape, (tuple, list))
            else 0
        )

        score = train_batches * max(1, feature_dim)
        strategy = "hyperband" if score > 10_000 else "random"

        logger.info(
            "Tuning strategy(auto): batches=%d, feature_dim=%d → %s",
            train_batches,
            feature_dim,
            strategy,
        )

    if strategy == "random":
        tuner = kt.RandomSearch(
            model_builder,
            objective="val_mae",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="tuner_results",
            project_name="dccl_random",
            overwrite=True,
        )

    elif strategy == "hyperband":
        tuner = kt.Hyperband(
            model_builder,
            objective="val_mae",
            max_epochs=50,
            factor=3,
            directory="tuner_results",
            project_name="dccl_hyperband",
            overwrite=True,
        )

    else:
        raise ValueError(
            "Unsupported strategy. Use 'random', 'hyperband', or 'auto'."
        )

    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(
                monitor="val_mae",
                patience=PATIENCE,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_mae",
                factor=0.5,
                patience=max(2, PATIENCE // 3),
                min_lr=1e-7,
            ),
        ],
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Best hyperparameters: %s", best_hp.values)
    return tuner
