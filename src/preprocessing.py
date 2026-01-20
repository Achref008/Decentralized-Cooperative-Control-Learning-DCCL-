# src/preprocessing.py
"""
Feature engineering, scaling, and train/test partitioning for
decentralized control learning nodes.

Responsibilities of this module:
- Derive additional features from raw columns.
- Assign stability classes based on control metrics.
- Normalize features and (optionally) targets.
- Produce per-node train/test splits with associated scalers.

This module does NOT load the dataset from disk; that is handled by
`data_loader.py`.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ------------------------------------------------------------------
# Global defaults
# ------------------------------------------------------------------

TEST_SIZE = 0.2
RANDOM_STATE = 42

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Scaling utilities
# ------------------------------------------------------------------

def get_scaler(method: str = "standard"):
    """
    Instantiate a feature/target scaler.

    Args:
        method: "standard" (default) or "minmax".

    Returns:
        Fitted scaler class (not yet fit).

    Raises:
        ValueError: for unsupported method.
    """
    method = method.lower()

    if method == "minmax":
        return MinMaxScaler()
    if method == "standard":
        return StandardScaler()

    raise ValueError("Invalid scaler method. Choose 'minmax' or 'standard'.")


# ------------------------------------------------------------------
# Labeling logic
# ------------------------------------------------------------------

def classify_region(settling_time: float, iae: float) -> int:
    """
    Map control performance metrics to a discrete region label.

    Regions (0–3):
    0: Excellent
    1: Fair
    2: Poor
    3: Unstable

    Args:
        settling_time: St_i value for a node.
        iae: IAE_i value for a node.

    Returns:
        Integer class label in [0, 3].
    """
    if settling_time >= 0.5 and iae > 0:
        return 3
    if 0.2 <= settling_time < 0.5 and iae <= 0.01:
        return 2
    if 0.1 <= settling_time < 0.2 and iae <= 0.01:
        return 1
    if settling_time < 0.1 and iae <= 0.01:
        return 0
    return 3  # Default to unstable


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, node_index: int) -> pd.DataFrame:
    """
    Create derived features for a single node.

    Required base columns:
        St{i}, IAE{i}

    Derived features:
        - first differences,
        - ratio IAE/St,
        - short moving averages (window=3).

    Args:
        df: Input DataFrame.
        node_index: Node identifier i (1-based in the dataset).

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()

    st = f"St{node_index}"
    iae = f"IAE{node_index}"

    df[f"{st}_diff"] = df[st].diff().fillna(0.0)
    df[f"{iae}_diff"] = df[iae].diff().fillna(0.0)

    df[f"{iae}_ratio"] = (df[iae] / df[st]) \
        .replace([np.inf, -np.inf], 0.0) \
        .fillna(0.0)

    df[f"{st}_sma"] = df[st].rolling(window=3, min_periods=1).mean()
    df[f"{iae}_sma"] = df[iae].rolling(window=3, min_periods=1).mean()

    return df


# ------------------------------------------------------------------
# Main preprocessing pipeline
# ------------------------------------------------------------------

def preprocess_and_split_data(
    data_path: str,
    scaler_type: str = "standard",
    normalize_targets: bool = True,
    multi_output: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Optional[object]]]]:
    """
    Load, engineer features, scale, and split data for each detected node.

    The number of nodes is inferred from columns named `St{i}`.

    Args:
        data_path: Path to the Excel file.
        scaler_type: "standard" or "minmax".
        normalize_targets: Whether to scale PID targets.
        multi_output: Whether to append one-hot class labels.

    Returns:
        nodes_data:
            List of dicts with keys:
            - X_train, X_test, y_train, y_test (NumPy arrays)

        scalers:
            List of dicts with keys:
            - feature_scaler, target_scaler (or None)
    """
    df = pd.read_excel(data_path)
    df.columns = df.columns.str.strip()

    # Detect nodes from column names
    num_nodes = sum(1 for c in df.columns if c.startswith("St"))
    logger.info("Detected %d nodes/controllers in dataset.", num_nodes)

    nodes_data: List[Dict[str, np.ndarray]] = []
    scalers: List[Dict[str, Optional[object]]] = []

    for i in range(1, num_nodes + 1):
        required_cols = [f"St{i}", f"IAE{i}", f"Kp{i}", f"Ki{i}", f"Kd{i}"]

        if not all(col in df.columns for col in required_cols):
            logger.warning(
                "Node %d missing required columns %s; skipping.",
                i,
                required_cols,
            )
            continue

        # Feature engineering
        df_fe = engineer_features(df, i)

        feature_cols = [
            f"St{i}",
            f"IAE{i}",
            f"St{i}_diff",
            f"IAE{i}_diff",
            f"IAE{i}_ratio",
            f"St{i}_sma",
            f"IAE{i}_sma",
        ]

        X = df_fe[feature_cols].to_numpy(dtype=np.float32)

        # Primary PID targets
        y_pid = df_fe[[f"Kp{i}", f"Ki{i}", f"Kd{i}"]].to_numpy(dtype=np.float32)

        # Optional classification output
        if multi_output:
            class_labels = df_fe.apply(
                lambda r: classify_region(r[f"St{i}"], r[f"IAE{i}"]),
                axis=1,
            ).to_numpy()

            y_class = tf.keras.utils.to_categorical(class_labels, num_classes=4)
            y = np.concatenate([y_pid, y_class], axis=1)

            counts = pd.Series(class_labels).value_counts().sort_index()
            for cls, cnt in counts.items():
                logger.info("Node %d: region %d has %d samples.", i, cls, cnt)
        else:
            y = y_pid

        # Sanity check
        if np.isnan(X).any() or np.isnan(y).any():
            logger.warning("NaNs detected for node %d; skipping.", i)
            continue

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # Feature scaling
        feature_scaler = get_scaler(scaler_type)
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_test_scaled = feature_scaler.transform(X_test)

        # Target scaling (PID outputs only)
        target_scaler = None

        if normalize_targets:
            target_scaler = get_scaler(scaler_type)
            y_train_pid = target_scaler.fit_transform(y_train[:, :3])
            y_test_pid = target_scaler.transform(y_test[:, :3])

            if multi_output:
                y_train_scaled = np.concatenate(
                    [y_train_pid, y_train[:, 3:]], axis=1
                )
                y_test_scaled = np.concatenate(
                    [y_test_pid, y_test[:, 3:]], axis=1
                )
            else:
                y_train_scaled = y_train_pid
                y_test_scaled = y_test_pid
        else:
            y_train_scaled = y_train
            y_test_scaled = y_test

        nodes_data.append(
            {
                "X_train": X_train_scaled,
                "X_test": X_test_scaled,
                "y_train": y_train_scaled,
                "y_test": y_test_scaled,
            }
        )

        scalers.append(
            {
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
            }
        )

        logger.info(
            "Node %d: X shape %s, y shape %s, features=%s",
            i,
            X.shape,
            y.shape,
            feature_cols,
        )

    return nodes_data, scalers
