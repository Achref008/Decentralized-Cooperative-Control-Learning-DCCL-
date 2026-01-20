# src/consensus.py
"""
Robust parameter reconciliation for decentralized learning nodes.

This module implements a Byzantine-tolerant aggregation mechanism that:
- filters out anomalous updates,
- weights contributions using validation performance,
- optionally applies exponential moving average (EMA) smoothing.

The goal is not to “average everything”, but to produce a stable
global parameter estimate in the presence of noisy or faulty nodes.
"""

import logging
from typing import List, Sequence, Optional, Dict

import numpy as np


class ByzantineFaultTolerance:
    """
    Robust aggregation operator for distributed nodes.

    Each aggregation step:
    1) Validates input structure.
    2) Identifies unreliable nodes using validation loss and simple overfitting checks.
    3) Performs a deviation-based filtering step.
    4) Computes a loss-weighted average over the remaining nodes.
    5) Optionally smooths the result using EMA with the previous model.
    """

    def __init__(self, total_nodes: int, fault_tolerant_nodes: int):
        """
        Args:
            total_nodes: Total number of participating nodes.
            fault_tolerant_nodes: Maximum number of nodes that may be faulty.

        Raises:
            ValueError: If the fault tolerance limit violates classical BFT bounds.
        """
        self.total_nodes = int(total_nodes)
        self.fault_tolerant_nodes = int(fault_tolerant_nodes)
        self.recent_valid_ratios: List[float] = []

        # Classical BFT limit: f < (n - 1) / 3
        if self.fault_tolerant_nodes > (self.total_nodes - 1) // 3:
            raise ValueError(
                f"fault_tolerant_nodes={fault_tolerant_nodes} exceeds BFT limit for "
                f"total_nodes={total_nodes}."
            )

    # ------------------------------------------------------------------
    # Outlier filtering
    # ------------------------------------------------------------------

    def adaptive_trimmed_mean(
        self,
        weights_list: Sequence[np.ndarray],
        base_multiplier: float = 3.5,
        fallback_trim: float = 0.1,
    ) -> np.ndarray:
        """
        Remove extreme outliers before averaging.

        This uses a median-based deviation filter with an adaptive threshold.
        If too many updates are removed, it falls back to a conservative
        trimmed mean.

        Args:
            weights_list: List of layer weight arrays (same shape).
            base_multiplier: Base scaling factor for deviation threshold.
            fallback_trim: Fraction to trim on fallback.

        Returns:
            Filtered mean weight tensor.
        """
        weights = np.stack(weights_list, axis=0)
        median = np.median(weights, axis=0)

        # Deviation magnitude per node
        deviations = np.linalg.norm(weights - median, axis=tuple(range(1, weights.ndim)))

        # Adapt threshold based on recent history
        if self.recent_valid_ratios:
            avg_valid = float(np.mean(self.recent_valid_ratios[-5:]))
            multiplier = float(
                np.clip(base_multiplier + (0.5 - avg_valid) * 4.0, 2.0, 5.0)
            )
        else:
            multiplier = base_multiplier

        threshold = np.median(deviations) * multiplier

        filtered = [
            w for w, d in zip(weights_list, deviations) if d < threshold
        ]

        # Fallback if too many are removed
        if len(filtered) < max(1, len(weights_list) // 2):
            logging.warning(
                "Too many updates rejected by deviation filter. "
                "Falling back to trimmed mean."
            )

            # Sort by distance to median and trim extremes
            sorted_pairs = sorted(
                zip(weights_list, deviations),
                key=lambda x: x[1],
            )
            trim_k = max(1, int(len(sorted_pairs) * fallback_trim))
            trimmed = [w for w, _ in sorted_pairs[trim_k:-trim_k]]

            return np.mean(np.stack(trimmed, axis=0), axis=0)

        return np.mean(np.stack(filtered, axis=0), axis=0)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def detect_overfitting(
        val_loss_history: Sequence[float],
        patience: int = 3,
        threshold: float = 0.03,
    ) -> bool:
        """
        Simple heuristic to flag clear overfitting.

        Returns True if validation loss increases consistently over
        the last `patience` steps by more than `threshold`.
        """
        if len(val_loss_history) < patience + 1:
            return False

        recent = val_loss_history[-(patience + 1):]
        return recent[-1] > recent[0] * (1.0 + threshold)

    # ------------------------------------------------------------------
    # Main aggregation routine
    # ------------------------------------------------------------------

    def aggregate_weights(
        self,
        weights_list: Sequence[Sequence[np.ndarray]],
        node_histories: Sequence[Dict[str, Sequence[float]]],
        previous_weights: Optional[Sequence[np.ndarray]] = None,
        ema_beta: float = 0.65,
    ) -> List[np.ndarray]:
        """
        Aggregate model parameters from all nodes.

        Args:
            weights_list: List of nodes, each containing a list of layer arrays.
            node_histories: Per-node training histories (must contain 'val_loss').
            previous_weights: Optional previous global model for EMA smoothing.
            ema_beta: EMA coefficient (0=no memory, 1=full memory).

        Returns:
            List of aggregated layer arrays.

        Raises:
            ValueError: If inputs are empty or structurally invalid.
            RuntimeError: If any layer cannot be aggregated safely.
        """
        if not weights_list or not node_histories:
            raise ValueError("weights_list and node_histories must be non-empty.")

        logging.info(
            f"Aggregating parameters from {len(weights_list)} nodes."
        )

        # Normalize to numpy arrays
        weights_list = [
            [np.asarray(layer, dtype=np.float32) for layer in node]
            for node in weights_list
        ]

        num_layers = len(weights_list[0])

        # ------------------------------------------------------------------
        # Node quality assessment
        # ------------------------------------------------------------------

        val_losses: List[float] = []
        overfitting_flags: List[bool] = []

        for i, hist in enumerate(node_histories):
            try:
                losses = hist.get("val_loss", [])
                loss = float(losses[-1]) if losses else float("inf")
                if not np.isfinite(loss):
                    raise ValueError("Non-finite validation loss")
            except Exception as e:
                logging.warning(f"Node {i}: invalid val_loss ({e}). Treating as worst case.")
                loss = float("inf")

            val_losses.append(loss)
            overfitting_flags.append(
                self.detect_overfitting(hist.get("val_loss", []))
            )

        # Select candidate nodes based on performance
        threshold = float(np.percentile(val_losses, 60))

        valid_nodes = [
            i for i, (l, of) in enumerate(zip(val_losses, overfitting_flags))
            if l <= threshold and not of
        ]

        # Relax selection if too few nodes remain
        min_required = max(1, int(self.total_nodes * 0.3))

        if len(valid_nodes) < min_required:
            logging.warning("Too few valid nodes; relaxing threshold to 75th percentile.")
            threshold = float(np.percentile(val_losses, 75))
            valid_nodes = [i for i, l in enumerate(val_losses) if l <= threshold]

        if len(valid_nodes) < min_required:
            logging.warning("Still too few valid nodes; using all nodes.")
            valid_nodes = list(range(len(weights_list)))

        logging.info(f"Selected nodes for aggregation: {valid_nodes}")

        # Track quality ratio
        valid_ratio = len(valid_nodes) / float(self.total_nodes)
        self.recent_valid_ratios.append(valid_ratio)
        if len(self.recent_valid_ratios) > 10:
            self.recent_valid_ratios.pop(0)

        # ------------------------------------------------------------------
        # Layer-wise aggregation
        # ------------------------------------------------------------------

        aggregated: List[np.ndarray] = []

        for layer_idx in range(num_layers):
            try:
                layer_updates = [weights_list[i][layer_idx] for i in valid_nodes]
                layer_losses = [val_losses[i] for i in valid_nodes]

                if not layer_updates:
                    raise ValueError(f"No valid updates for layer {layer_idx}.")

                # Loss-weighted averaging (better nodes contribute more)
                inv_losses = np.array([1.0 / (l + 1e-8) for l in layer_losses])
                inv_losses /= inv_losses.sum()

                stacked = np.stack(layer_updates, axis=0)
                layer_mean = np.tensordot(inv_losses, stacked, axes=(0, 0))

                # Optional additional outlier filtering
                layer_mean = self.adaptive_trimmed_mean(
                    [layer_mean] + layer_updates
                )

                aggregated.append(layer_mean.astype(np.float32))

            except Exception as e:
                raise RuntimeError(
                    f"Aggregation failed at layer {layer_idx}: {e}"
                )

        # ------------------------------------------------------------------
        # Optional EMA smoothing with previous global model
        # ------------------------------------------------------------------

        if previous_weights is not None:
            logging.info("Applying EMA smoothing to aggregated parameters.")
            aggregated = [
                ema_beta * prev + (1.0 - ema_beta) * new
                for prev, new in zip(previous_weights, aggregated)
            ]

        # Final integrity checks
        for idx, w in enumerate(aggregated):
            if not isinstance(w, np.ndarray):
                raise RuntimeError(
                    f"Layer {idx} aggregation did not produce a NumPy array."
                )
            if np.isnan(w).any():
                logging.warning(f"Layer {idx} contains NaNs after aggregation.")
            if w.shape != weights_list[0][idx].shape:
                raise RuntimeError(
                    f"Shape mismatch at layer {idx}: "
                    f"got {w.shape}, expected {weights_list[0][idx].shape}."
                )

        logging.info(
            f"Aggregation complete. First layer mean = {float(np.mean(aggregated[0])):.6f}"
        )
        return aggregated
