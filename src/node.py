# src/node.py
"""
Distributed learning node abstraction.

A Node encapsulates:
- Local data (train/validation split),
- A TensorFlow/Keras model,
- ZMQ communication endpoints for:
    - sending local parameters to an aggregator,
    - receiving aggregated parameters from peers.

Design principles:
- Explicit contracts for inputs and outputs.
- Minimal side effects in the constructor.
- Deterministic dataset creation.
- Clear separation between training, inference, and communication.
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import zmq
from sklearn.model_selection import train_test_split

from config import EPOCHS, TEST_SIZE, RANDOM_STATE, BATCH_SIZE
from transfer_learning import load_pretrained_model
from communication import deserialize_weights

logger = logging.getLogger(__name__)


class Node:
    """
    A single decentralized learning node.

    Each node:
    - owns a local dataset,
    - maintains a local model,
    - can send its parameters to an aggregator,
    - can receive and apply aggregated parameters.
    """

    def __init__(
        self,
        node_id: int,
        context: zmq.Context,
        data_chunk: Tuple[np.ndarray, np.ndarray],
        initial_weights: Optional[list] = None,
        use_sequence: bool = False,
        multi_output: bool = False,
    ):
        """
        Args:
            node_id: Unique integer identifier for this node.
            context: Active ZeroMQ context.
            data_chunk: Tuple (X, y) as NumPy arrays.
            initial_weights: Optional list of model weights to initialize from.
            use_sequence: Whether inputs are sequential (LSTM front-end).
            multi_output: Whether the model has two outputs (PID + classification).

        Raises:
            ValueError: If the data format is invalid or too small.
        """
        self.node_id = int(node_id)
        self.context = context
        self.use_sequence = bool(use_sequence)
        self.multi_output = bool(multi_output)
        self.subscriber_socket: Optional[zmq.Socket] = None

        logger.info("Node %d: validating local data format.", self.node_id)

        if not (
            isinstance(data_chunk, tuple)
            and len(data_chunk) == 2
            and isinstance(data_chunk[0], np.ndarray)
            and isinstance(data_chunk[1], np.ndarray)
        ):
            raise ValueError(
                f"Node {self.node_id}: expected data_chunk=(X, y) as NumPy arrays."
            )

        X, y = data_chunk

        if X.shape[0] < 2:
            raise ValueError(
                f"Node {self.node_id}: insufficient samples for train/validation split."
            )

        # Deterministic train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        self.input_shape = tuple(X_train.shape[1:])

        # Build TensorFlow datasets
        self.train_dataset = self._create_dataset(X_train, y_train)
        self.test_dataset = self._create_dataset(X_val, y_val)

        # Load or initialize model
        self.model = load_pretrained_model(input_shape=self.input_shape)

        if initial_weights is not None:
            try:
                self.set_weights(initial_weights)
            except Exception as e:
                logger.error(
                    "Node %d: failed to apply initial weights: %s",
                    self.node_id,
                    str(e),
                )

        # Minimal training history tracking
        self.history = {"loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Dataset utilities
    # ------------------------------------------------------------------

    def _create_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = BATCH_SIZE,
        buffer_size: int = 1024,
    ) -> tf.data.Dataset:
        """
        Create a shuffled, batched tf.data.Dataset.

        For multi-output, the last columns of y are assumed to be
        one-hot class labels; the first three are PID targets.
        """
        if self.multi_output:
            y_pid = y[:, :3]
            y_class = y[:, 3:]
            dataset = tf.data.Dataset.from_tensor_slices((X, (y_pid, y_class)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((X, y))

        return dataset.shuffle(buffer_size).batch(batch_size)

    # ------------------------------------------------------------------
    # Model accessors
    # ------------------------------------------------------------------

    def get_weights(self) -> list:
        """Return the current model weights as a list of NumPy arrays."""
        return self.model.get_weights()

    def set_weights(self, weights: list) -> None:
        """Set model weights. Raises if shapes are incompatible."""
        self.model.set_weights(weights)

    def set_model(self, model: tf.keras.Model) -> None:
        """Replace the current model with a new one."""
        self.model = model

    # ------------------------------------------------------------------
    # Evaluation and inference
    # ------------------------------------------------------------------

    def evaluate(self):
        """
        Evaluate the model on the local validation set.

        Returns:
            Keras evaluation metrics.
        """
        if self.test_dataset is None:
            raise RuntimeError(f"Node {self.node_id}: no test dataset available.")

        return self.model.evaluate(self.test_dataset, verbose=1)

    def predict(self) -> np.ndarray:
        """
        Run inference on the local validation set.

        Returns:
            Model predictions as a NumPy array.
        """
        if self.test_dataset is None:
            raise RuntimeError(f"Node {self.node_id}: no test dataset available.")

        return self.model.predict(self.test_dataset, verbose=0)

    # ------------------------------------------------------------------
    # Communication primitives (ZMQ)
    # ------------------------------------------------------------------

    def send_weights(self, port: int) -> None:
        """
        Send local model parameters to an aggregator via PUSH.

        Args:
            port: TCP port of the aggregator PULL socket.
        """
        socket = self.context.socket(zmq.PUSH)
        socket.setsockopt(zmq.LINGER, 0)

        try:
            socket.connect(f"tcp://localhost:{port}")
            socket.send_pyobj(self.get_weights())
            logger.info("Node %d: sent local weights to port %d.", self.node_id, port)
        finally:
            socket.close()

    def subscribe_to_updates(
        self,
        port: int,
        sync_port: int = 5560,
    ) -> None:
        """
        Subscribe to aggregated model updates (PUB/SUB) and perform a
        simple synchronization handshake before training starts.

        Args:
            port: Publisher port to subscribe to.
            sync_port: REQ/REP synchronization port.
        """
        # Subscriber socket for model updates
        self.subscriber_socket = self.context.socket(zmq.SUB)
        self.subscriber_socket.setsockopt(zmq.LINGER, 0)
        self.subscriber_socket.connect(f"tcp://localhost:{port}")
        self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Synchronization handshake
        sync_socket = self.context.socket(zmq.REQ)
        sync_socket.setsockopt(zmq.LINGER, 0)

        try:
            sync_socket.connect(f"tcp://localhost:{sync_port}")
            sync_socket.send(b"ready")
            sync_socket.recv()
            logger.info(
                "Node %d: synchronization handshake completed.", self.node_id
            )
        finally:
            sync_socket.close()

    def listen_for_update(self, timeout_ms: int = 3000) -> None:
        """
        Block until an aggregated model update is received or timeout occurs.

        Args:
            timeout_ms: Maximum wait time in milliseconds.

        Raises:
            TimeoutError: If no update is received in time.
        """
        if self.subscriber_socket is None:
            raise RuntimeError(
                f"Node {self.node_id}: not subscribed to updates."
            )

        poller = zmq.Poller()
        poller.register(self.subscriber_socket, zmq.POLLIN)

        deadline = time.time() + timeout_ms / 1000.0
        received = False

        while time.time() < deadline:
            events = dict(poller.poll(timeout=100))

            if self.subscriber_socket in events:
                try:
                    # Prefer native PyObj if available, otherwise fall back to string
                    try:
                        new_weights = self.subscriber_socket.recv_pyobj(
                            flags=zmq.NOBLOCK
                        )
                    except Exception:
                        serialized = self.subscriber_socket.recv_string(
                            flags=zmq.NOBLOCK
                        )
                        new_weights = deserialize_weights(serialized)

                    self.set_weights(new_weights)
                    logger.info(
                        "Node %d: applied aggregated weights.", self.node_id
                    )
                    received = True
                    break

                except Exception as e:
                    raise RuntimeError(
                        f"Node {self.node_id}: failed to apply update: {e}"
                    ) from e

        if not received:
            raise TimeoutError(
                f"Node {self.node_id}: timed out waiting for model update."
            )

    # ------------------------------------------------------------------
    # Compatibility helper
    # ------------------------------------------------------------------

    def set_socket_for_receiving(self, socket: zmq.Socket) -> None:
        """
        Backward-compatibility helper used in some tests.

        Prefer `subscribe_to_updates` in new code.
        """
        self.subscriber_socket = socket
