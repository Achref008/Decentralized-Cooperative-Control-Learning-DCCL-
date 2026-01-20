# src/visualization.py
"""
Plotting utilities for decentralized PID learning.

Design principles:
- Portable output paths (no hard-coded machine folders).
- Safe for headless environments (servers, CI).
- All functions save figures by default and show only when requested.
- No side effects at import time.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix

logger = logging.getLogger(__name__)

# Default output location (portable)
DEFAULT_OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR", os.path.join("outputs", "figures")
)


def _ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def get_node_output_dir(
    node_id: int, base_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    """
    Return (and create if needed) the output folder for a specific node.
    """
    node_dir = os.path.join(base_dir, f"node_{node_id}")
    _ensure_dir(node_dir)
    return node_dir


# ------------------------------------------------------------------
# Smoothing utility
# ------------------------------------------------------------------

def smooth_curve(values: Sequence[float], smoothing: float = 0.97) -> list:
    """
    Exponential smoothing for noisy curves (useful for loss plots).
    """
    values = list(values)
    if not values:
        return []

    smoothed = []
    last = values[0]

    for v in values:
        last = last * smoothing + v * (1 - smoothing)
        smoothed.append(last)

    return smoothed


# ------------------------------------------------------------------
# Training history plots
# ------------------------------------------------------------------

def plot_all_nodes_training_history(
    nodes, base_dir: str = DEFAULT_OUTPUT_DIR, show: bool = False
) -> str:
    """
    Plot loss/val_loss curves for all nodes on one figure.
    """
    _ensure_dir(base_dir)
    save_path = os.path.join(
        base_dir, "all_nodes_training_history.png"
    )

    plt.figure(figsize=(12, 6))

    for node in nodes:
        hist = getattr(node, "history", None)
        if not hist:
            continue

        loss = hist.get("loss", [])
        val_loss = hist.get("val_loss", [])

        if loss:
            plt.plot(loss, label=f"Node {node.node_id} train")
        if val_loss:
            plt.plot(
                val_loss,
                linestyle="dashed",
                label=f"Node {node.node_id} val",
            )

    plt.title("Training and Validation Loss (All Nodes)")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Loss values")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    logger.info("Saved training history plot: %s", save_path)
    return save_path


# ------------------------------------------------------------------
# Prediction diagnostics
# ------------------------------------------------------------------

def plot_actual_vs_predicted(
    y_true,
    y_pred,
    scaler=None,
    node_id: Optional[int] = None,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> Optional[str]:
    """
    Plot a flattened 'actual vs predicted' curve and report MSE.
    """
    try:
        _ensure_dir(base_dir)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if scaler is not None:
            y_true = scaler.inverse_transform(y_true.reshape(-1, 3))
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 3))

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        mse = mean_squared_error(y_true_f, y_pred_f)

        plt.figure(figsize=(12, 4))
        plt.plot(y_true_f, label="Actual")
        plt.plot(y_pred_f, label="Predicted")

        title = (
            f"Actual vs Predicted (MSE={mse:.4f})"
            if node_id is None
            else f"Node {node_id}: Actual vs Predicted (MSE={mse:.4f})"
        )

        plt.title(title)
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if node_id is not None:
            node_dir = get_node_output_dir(node_id, base_dir)
            save_path = os.path.join(node_dir, "pred_vs_actual.png")
        else:
            save_path = os.path.join(base_dir, "pred_vs_actual.png")

        plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

        logger.info("Saved prediction plot: %s", save_path)
        return save_path

    except Exception as exc:
        logger.exception(
            "Failed to plot actual vs predicted: %s", exc
        )
        return None


def display_node_mse(
    nodes, base_dir: str = DEFAULT_OUTPUT_DIR, show: bool = False
) -> Optional[str]:
    """
    Bar plot of mean validation loss per node.
    """
    _ensure_dir(base_dir)
    save_path = os.path.join(
        base_dir, "node_mse_comparison.png"
    )

    node_ids = []
    mse_values = []

    for node in nodes:
        hist = getattr(node, "history", None)
        if not hist or not hist.get("val_loss"):
            continue

        mse = float(np.mean(hist["val_loss"]))
        node_ids.append(node.node_id)
        mse_values.append(mse)

    if not node_ids:
        logger.warning(
            "No node histories with val_loss found; skipping MSE plot."
        )
        return None

    plt.figure(figsize=(10, 4))
    plt.bar(node_ids, mse_values)
    plt.title("Mean Validation Loss per Node")
    plt.xlabel("Node ID")
    plt.ylabel("Mean val_loss")
    plt.xticks(node_ids)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    logger.info("Saved MSE comparison plot: %s", save_path)
    return save_path


# ------------------------------------------------------------------
# Fit status visualization
# ------------------------------------------------------------------

def plot_node_fit_status(
    fit_status_dict: Dict[int, str],
    save_path: Optional[str] = None,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> str:
    """
    Visualize a simple per-node status label:
    Good fit / Overfitting / Underfitting / Insufficient data.
    """
    _ensure_dir(base_dir)

    if save_path is None:
        save_path = os.path.join(
            base_dir, "node_fit_status.png"
        )

    status_colors = {
        "Good fit": "tab:green",
        "Overfitting": "tab:red",
        "Underfitting": "tab:blue",
        "Insufficient data": "tab:gray",
    }

    node_ids = list(fit_status_dict.keys())
    statuses = [fit_status_dict[n] for n in node_ids]
    colors = [
        status_colors.get(s, "tab:gray") for s in statuses
    ]

    plt.figure(figsize=(10, 3))
    plt.bar(
        node_ids,
        [1] * len(node_ids),
        color=colors,
        edgecolor="black",
    )
    plt.xticks(node_ids, [f"Node {n}" for n in node_ids])
    plt.yticks([])
    plt.title("Node Fit Status")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, label=l)
        for l, c in status_colors.items()
    ]

    plt.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(status_colors),
    )
    plt.tight_layout()
    plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close()

    logger.info("Saved fit status plot: %s", save_path)
    return save_path


# ------------------------------------------------------------------
# PID component plots
# ------------------------------------------------------------------

def plot_pid_components(
    y_true,
    y_pred,
    node_id: int,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> None:
    """
    Plot Kp, Ki, Kd curves separately.
    Expects y_true and y_pred shape (n_samples, 3).
    """
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if (
            y_true.ndim != 2
            or y_pred.ndim != 2
            or y_true.shape[1] != 3
            or y_pred.shape[1] != 3
        ):
            logger.warning(
                "Node %s: expected (n_samples, 3). Got %s and %s",
                node_id,
                y_true.shape,
                y_pred.shape,
            )
            return

        node_dir = get_node_output_dir(node_id, base_dir)
        names = ["Kp", "Ki", "Kd"]

        for i, name in enumerate(names):
            plt.figure(figsize=(12, 3))
            plt.plot(y_true[:, i], label="Actual", alpha=0.8)
            plt.plot(y_pred[:, i], label="Predicted", alpha=0.8)
            plt.title(f"Node {node_id}: {name} Prediction")
            plt.xlabel("Sample")
            plt.ylabel(name)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(
                node_dir, f"{name.lower()}_pred_vs_actual.png"
            )
            plt.savefig(save_path)

            if show:
                plt.show()
            else:
                plt.close()

            logger.info(
                "Saved %s plot: %s", name, save_path
            )

    except Exception as exc:
        logger.exception(
            "Failed to plot PID components for node %s: %s",
            node_id,
            exc,
        )


def plot_node_predictions_summary(
    y_true,
    y_pred,
    node_id: int,
    base_dir: str = DEFAULT_OUTPUT_DIR,
    show: bool = False,
) -> None:
    """
    Summary plot: Kp/Ki/Kd panels + flattened output with MSE.
    """
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if (
            y_true.shape != y_pred.shape
            or y_true.ndim != 2
            or y_true.shape[1] != 3
        ):
            logger.warning(
                "Node %s: invalid shapes for summary plot: %s vs %s",
                node_id,
                y_true.shape,
                y_pred.shape,
            )
            return

        node_dir = get_node_output_dir(node_id, base_dir)
        save_path = os.path.join(
            node_dir, "prediction_summary.png"
        )

        mse = mean_squared_error(
            y_true.flatten(), y_pred.flatten()
        )

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        names = ["Kp", "Ki", "Kd"]

        for i, name in enumerate(names):
            ax = axs[i // 2, i % 2]
            ax.plot(y_true[:, i], label="Actual", alpha=0.8)
            ax.plot(y_pred[:, i], label="Predicted", alpha=0.8)
            ax.set_title(f"{name} Prediction")
            ax.set_xlabel("Sample")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.legend()

        ax = axs[1, 1]
        ax.plot(y_true.flatten(), label="Actual")
        ax.plot(y_pred.flatten(), label="Predicted")
        ax.set_title(
            f"Flattened Output (MSE={mse:.4f})"
        )
        ax.set_xlabel("Sample")
        ax.set_ylabel("PID (flattened)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.suptitle(
            f"Node {node_id}: Prediction Summary", fontsize=12
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)

        if show:
            plt.show()
        else:
            plt.close()

        logger.info(
            "Saved prediction summary plot: %s", save_path
        )

    except Exception as exc:
        logger.exception(
            "Failed to create prediction summary for node %s: %s",
            node_id,
            exc,
        )


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------

def evaluate_all_nodes(nodes, show: bool = False) -> None:
    """
    Kept for backward compatibility with `main.py` imports.
    """
    for node in nodes:
        logger.info(
            "Node %s: evaluation is handled in main.py",
            node.node_id,
        )


def plot_classification_results(
    y_true_onehot,
    y_pred_proba,
    node_id: Optional[int] = None,
    show: bool = True,
) -> None:
    """
    Optional classification diagnostics: report + confusion matrix.
    """
    y_true = np.argmax(
        np.asarray(y_true_onehot), axis=1
    )
    y_pred = np.argmax(
        np.asarray(y_pred_proba), axis=1
    )

    print(f"\nNode {node_id} classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Excellent",
                "Fair",
                "Poor",
                "Unstable",
            ],
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    # Optional seaborn; fallback to plain matplotlib
    try:
        import seaborn as sns

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=[
                "Excellent",
                "Fair",
                "Poor",
                "Unstable",
            ],
            yticklabels=[
                "Excellent",
                "Fair",
                "Poor",
                "Unstable",
            ],
        )
    except Exception:
        plt.figure(figsize=(6, 4))
        plt.imshow(cm)
        plt.colorbar()
        plt.xticks(
            range(4),
            ["Excellent", "Fair", "Poor", "Unstable"],
            rotation=45,
        )
        plt.yticks(
            range(4),
            ["Excellent", "Fair", "Poor", "Unstable"],
        )
        for (i, j), val in np.ndenumerate(cm):
            plt.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
            )

    plt.title(f"Node {node_id} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.close()
