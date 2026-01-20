# src/main.py
"""
Entry point for decentralized cooperative control learning (DCCL).

High-level execution flow:

1. Environment setup (logging, GPU configuration, output directories).
2. Data preprocessing and optional centralized pretraining.
3. Node initialization (one model per node).
4. Optional per-node hyperparameter tuning.
5. Decentralized training rounds with:
   - local training,
   - parameter exchange,
   - robust aggregation,
   - global synchronization.
6. Post-training evaluation and visualization.

This script is intended to be run from the project root:

    python -m src.main --tuning auto --max_trials 20
"""

import argparse
import logging
import os
import threading
import time
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import zmq
from tqdm import tqdm

from communication import setup_publisher, setup_receiver, sync_subscribers
from consensus import ByzantineFaultTolerance
from config import (
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_FILE,
    DATA_PATH,
    FAULT_TOLERANT_NODES,
)
from node import Node
from preprocessing import preprocess_and_split_data
from transfer_learning import pretrain_on_full_data
from training import train_node, evaluate_nodes, run_hyperparameter_tuning
from visualization import (
    evaluate_all_nodes,
    plot_all_nodes_training_history,
    plot_actual_vs_predicted,
    display_node_mse,
    plot_pid_components,
    plot_node_predictions_summary,
    plot_node_fit_status,
)


# ------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------

def setup_environment(output_dir: str = "figures") -> None:
    """
    Configure logging, create output directories, and configure GPUs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Configure TensorFlow GPU behavior
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Detected {len(gpus)} GPU(s); enabling memory growth.")
        except RuntimeError as e:
            print(f"Could not configure GPU memory growth: {e}")
    else:
        print("No GPU detected; running on CPU.")

    # Centralized logging configuration
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    logging.info("Environment initialized.")


# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

def prepare_data(multi_output: bool = False) -> Tuple[list, list, np.ndarray, np.ndarray]:
    """
    Load, preprocess, and merge training data across nodes.

    Returns:
        node_data: per-node train/test splits,
        scalers: per-node scalers,
        X_all: stacked training features (all nodes),
        y_all: stacked training targets (all nodes).
    """
    node_data, scalers = preprocess_and_split_data(
        DATA_PATH, multi_output=multi_output
    )

    X_all = np.vstack([n["X_train"] for n in node_data])
    y_all = np.vstack([n["y_train"] for n in node_data])

    logging.info(
        "Prepared data: total X shape %s, total y shape %s.",
        X_all.shape,
        y_all.shape,
    )
    return node_data, scalers, X_all, y_all


# ------------------------------------------------------------------
# Node initialization
# ------------------------------------------------------------------

def initialize_nodes(
    node_data: list,
    context: zmq.Context,
    use_sequence: bool = False,
    multi_output: bool = False,
) -> List[Node]:
    """
    Instantiate one Node object per data partition.
    """
    nodes = []
    for i, data in enumerate(node_data):
        node = Node(
            node_id=i,
            context=context,
            data_chunk=(data["X_train"], data["y_train"]),
            use_sequence=use_sequence,
            multi_output=multi_output,
        )
        nodes.append(node)
    return nodes


# ------------------------------------------------------------------
# Hyperparameter tuning (optional)
# ------------------------------------------------------------------

def tune_models(
    nodes: List[Node],
    strategy: str = "auto",
    max_trials: int = 20,
    use_sequence: bool = False,
    multi_output: bool = False,
) -> None:
    """
    Run per-node hyperparameter tuning and replace each node's model
    with the tuned version.
    """
    for node in nodes:
        logging.info("Tuning model for node %d.", node.node_id)

        tuner = run_hyperparameter_tuning(
            train_dataset=node.train_dataset,
            val_dataset=node.test_dataset,
            input_shape=node.input_shape,
            strategy=strategy,
            max_trials=max_trials,
            use_sequence=use_sequence,
            multi_output=multi_output,
        )

        best_hp = tuner.get_best_hyperparameters(1)[0]
        node.set_model(
            create_model(
                best_hp,
                input_shape=node.input_shape,
                use_sequence=use_sequence,
                multi_output=multi_output,
            )
        )


# ------------------------------------------------------------------
# Synchronization barrier
# ------------------------------------------------------------------

def start_synchronization(nodes: List[Node], context: zmq.Context) -> threading.Thread:
    """
    Start a background thread that waits for all subscribers to connect.
    """
    thread = threading.Thread(
        target=sync_subscribers,
        args=(context, len(nodes), 5560),
        daemon=True,
    )
    thread.start()

    for node in nodes:
        node.subscribe_to_updates(port=5556, sync_port=5560)

    return thread


# ------------------------------------------------------------------
# Decentralized training loop
# ------------------------------------------------------------------

def decentralized_training(
    nodes: List[Node],
    context: zmq.Context,
    num_rounds: int = 5,
) -> None:
    """
    Run decentralized training rounds with robust aggregation.
    """
    publisher = setup_publisher(context, port=5556)
    receiver = setup_receiver(context, port=5557)

    consensus = ByzantineFaultTolerance(
        total_nodes=len(nodes),
        fault_tolerant_nodes=FAULT_TOLERANT_NODES,
    )

    for round_idx in range(num_rounds):
        logging.info(
            "Starting training round %d/%d.", round_idx + 1, num_rounds
        )

        # Local training and weight submission
        for node in tqdm(nodes, desc=f"Round {round_idx + 1}"):
            train_node(node)
            node.send_weights(port=5557)

        # Collect updates
        received_weights = []
        poller = zmq.Poller()
        poller.register(receiver, zmq.POLLIN)

        deadline = time.time() + 5.0
        while time.time() < deadline and len(received_weights) < len(nodes):
            events = dict(poller.poll(500))
            if receiver in events:
                try:
                    received_weights.append(receiver.recv_pyobj(flags=zmq.NOBLOCK))
                except zmq.Again:
                    continue

        if not received_weights:
            logging.warning("No updates received this round; skipping aggregation.")
            continue

        logging.info(
            "Received %d/%d updates.", len(received_weights), len(nodes)
        )

        # Robust aggregation
        node_histories = [n.history for n in nodes]
        aggregated = consensus.aggregate_weights(
            received_weights, node_histories
        )

        # Broadcast global model
        publisher.send_pyobj(aggregated)

        # Apply update on each node
        time.sleep(1.0)
        for node in nodes:
            node.listen_for_update(timeout_ms=3000)


# ------------------------------------------------------------------
# Post-training evaluation
# ------------------------------------------------------------------

def post_training_evaluation(nodes: List[Node], scalers: list) -> None:
    """
    Run evaluation and generate visual diagnostics.
    """
    evaluate_nodes(nodes)
    evaluate_all_nodes(nodes, show=False)
    plot_all_nodes_training_history(nodes, show=True)
    display_node_mse(nodes, show=True)

    fit_status = {}

    for i, node in enumerate(nodes):
        try:
            y_pred = node.predict()
            batch = next(iter(node.test_dataset))
            _, y_true = batch

            # Handle multi-output vs single-output
            if isinstance(y_true, tuple):
                y_pid_true, y_class_true = y_true
                y_pid_pred, y_class_pred = y_pred

                scaler = scalers[i].get("target_scaler")
                if scaler:
                    y_pid_pred = scaler.inverse_transform(y_pid_pred)
                    y_pid_true = scaler.inverse_transform(y_pid_true)

                plot_node_predictions_summary(
                    y_pid_true,
                    y_pid_pred,
                    node_id=node.node_id,
                    show=True,
                )
                plot_classification_results(
                    y_class_true, y_class_pred, node_id=node.node_id, show=True
                )

            else:
                scaler = scalers[i].get("target_scaler")
                if scaler:
                    y_pred = scaler.inverse_transform(y_pred)
                    y_true = scaler.inverse_transform(y_true)

                plot_node_predictions_summary(
                    y_true,
                    y_pred,
                    node_id=node.node_id,
                    show=True,
                )

            # Simple fit diagnostics
            val_loss = node.history.get("val_loss", [])
            if len(val_loss) < 2:
                status = "Insufficient data"
            elif val_loss[-1] > val_loss[0] * 1.1:
                status = "Overfitting"
            elif val_loss[-1] < 0.01:
                status = "Good fit"
            else:
                status = "Underfitting"

            fit_status[node.node_id] = status

        except Exception as e:
            logging.error(
                "Failed to process predictions for node %d: %s", i, str(e)
            )

    plot_node_fit_status(fit_status, show=True)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decentralized PID tuning and training"
    )

    parser.add_argument(
        "--tuning",
        type=str,
        default="auto",
        choices=["auto", "random", "hyperband"],
        help="Hyperparameter tuning strategy.",
    )

    parser.add_argument(
        "--max_trials",
        type=int,
        default=20,
        help="Maximum number of tuning trials per node.",
    )

    parser.add_argument(
        "--use_sequence",
        action="store_true",
        help="Use LSTM-based sequential model.",
    )

    parser.add_argument(
        "--multi_output",
        action="store_true",
        help="Enable multi-output model (PID + classification).",
    )

    args = parser.parse_args()

    start_time = time.time()

    try:
        logging.info(
            "Starting decentralized training "
            "(tuning=%s, use_sequence=%s, multi_output=%s).",
            args.tuning,
            args.use_sequence,
            args.multi_output,
        )

        setup_environment()

        node_data, scalers, X_all, y_all = prepare_data(
            multi_output=args.multi_output
        )

        # Optional centralized pretraining
        input_shape = X_all.shape[1:]
        pretrain_on_full_data(
            X_all, y_all, input_shape=input_shape
        )

        context = zmq.Context()
        nodes = initialize_nodes(
            node_data,
            context,
            use_sequence=args.use_sequence,
            multi_output=args.multi_output,
        )

        logging.info("Running hyperparameter tuning (if enabled).")
        tune_models(
            nodes,
            strategy=args.tuning,
            max_trials=args.max_trials,
            use_sequence=args.use_sequence,
            multi_output=args.multi_output,
        )

        sync_thread = start_synchronization(nodes, context)
        decentralized_training(nodes, context, num_rounds=5)
        sync_thread.join()

        post_training_evaluation(nodes, scalers)

    except Exception as e:
        logging.error("Fatal error in main execution: %s", str(e))

    logging.info(
        "Total runtime: %.2f seconds.", time.time() - start_time
    )


if __name__ == "__main__":
    main()
