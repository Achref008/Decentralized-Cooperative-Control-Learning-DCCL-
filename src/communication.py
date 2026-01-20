# src/communication.py
"""
ZeroMQ-based communication utilities for decentralized node coordination.

This module provides:
- Publisher / subscriber setup for broadcasting model parameters.
- Point-to-point receiver setup for collecting parameters.
- Deterministic (de)serialization of weight objects.
- Basic retry logic with exponential backoff.
- A synchronization barrier so all nodes start publishing at the same time.

Design principles:
- No hidden side effects.
- Sockets are created explicitly by the caller.
- Minimal assumptions about the caller’s execution context.
- Clear separation between transport (ZMQ) and application logic.
"""

import base64
import logging
import pickle
import threading
import time
from typing import List, Callable, Any

import zmq
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ================================================================
# ZMQ SOCKET FACTORIES
# ================================================================

def setup_publisher(context: zmq.Context, port: int) -> zmq.Socket:
    """
    Create and bind a PUB socket for broadcasting model parameters.

    The caller is responsible for closing the socket.

    Args:
        context: Active ZMQ context.
        port: TCP port to bind on (e.g., 5556).

    Returns:
        Bound ZMQ PUB socket.
    """
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    socket.setsockopt(zmq.SNDHWM, 1000)
    logging.info(f"Publisher bound to tcp://*:{port}")
    return socket


def setup_receiver(context: zmq.Context, port: int) -> zmq.Socket:
    """
    Create and bind a PULL socket for collecting weights from nodes.

    Args:
        context: Active ZMQ context.
        port: TCP port to bind on.

    Returns:
        Bound ZMQ PULL socket.
    """
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{port}")
    logging.info(f"Receiver (PULL) bound to tcp://*:{port}")
    return socket


def setup_subscribers(context: zmq.Context, ports: List[int]) -> List[zmq.Socket]:
    """
    Create SUB sockets and connect them to a list of publisher ports.

    Each subscriber listens to all topics (empty subscription filter).

    Args:
        context: Active ZMQ context.
        ports: List of TCP ports to connect to.

    Returns:
        List of connected SUB sockets.
    """
    sockets: List[zmq.Socket] = []

    for port in ports:
        sock = context.socket(zmq.SUB)
        sock.connect(f"tcp://localhost:{port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVHWM, 1000)
        sock.setsockopt(zmq.RCVBUF, 10 * 1024 * 1024)  # 10 MB receive buffer
        sockets.append(sock)
        logging.info(f"Subscriber connected to tcp://localhost:{port}")

    return sockets


# ================================================================
# WEIGHT SERIALIZATION
# ================================================================

def serialize_weights(weights: List[np.ndarray]) -> str:
    """
    Serialize a list of NumPy arrays for transmission over ZMQ.

    We use pickle + base64 to guarantee byte-safe transport.

    Args:
        weights: List of model weight arrays.

    Returns:
        Base64-encoded string.
    """
    payload = pickle.dumps(weights)
    return base64.b64encode(payload).decode("utf-8")


def deserialize_weights(serialized_weights: str) -> List[np.ndarray]:
    """
    Inverse of `serialize_weights`.

    Args:
        serialized_weights: Base64-encoded string.

    Returns:
        List of NumPy arrays.
    """
    raw = base64.b64decode(serialized_weights.encode("utf-8"))
    return pickle.loads(raw)


# ================================================================
# RETRY UTILITY
# ================================================================

def exponential_backoff_retry(
    func: Callable[..., Any],
    retries: int = 5,
    max_delay: int = 10,
    *args,
    **kwargs,
) -> Any:
    """
    Execute a function with exponential backoff on failure.

    Args:
        func: Callable to execute.
        retries: Maximum number of attempts.
        max_delay: Upper bound on sleep time (seconds).
        *args, **kwargs: Passed to `func`.

    Returns:
        Function result on success, or None after final failure.
    """
    delay = 1.0

    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(
                f"Retry {attempt}/{retries} failed: {e}"
            )
            if attempt < retries:
                time.sleep(delay)
                delay = min(delay * 2.0, max_delay)

    logging.error("Operation failed after all retry attempts.")
    return None


# ================================================================
# NODE SYNCHRONIZATION
# ================================================================

def receive_weights_for_node(
    node,
    retries: int = 5,
    timeout_ms: int = 1000,
) -> threading.Thread:
    """
    Spawn a background thread that waits for a weight update
    and applies it to `node.model` when received.

    This is used only during global synchronization.

    Args:
        node: Object exposing `socket` and `model.set_weights(...)`.
        retries: Number of polling attempts.
        timeout_ms: Poll timeout per attempt (milliseconds).

    Returns:
        Thread handle.
    """

    def worker():
        poller = zmq.Poller()
        poller.register(node.socket, zmq.POLLIN)

        for _ in range(retries):
            events = dict(poller.poll(timeout_ms))

            if node.socket in events and events[node.socket] == zmq.POLLIN:
                try:
                    message = node.socket.recv_string(flags=zmq.NOBLOCK)
                    weights = deserialize_weights(message)
                    node.model.set_weights(weights)
                    logging.info(
                        f"Node {node.node_id}: weights updated from broadcast."
                    )
                    return
                except Exception as e:
                    logging.error(
                        f"Node {node.node_id}: failed to apply weights: {e}"
                    )

        logging.warning(
            f"Node {node.node_id}: no update received after {retries} attempts."
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def synchronize_nodes(
    publisher_socket: zmq.Socket,
    nodes: list,
    aggregated_weights: List[np.ndarray],
    retries: int = 3,
    timeout_ms: int = 1000,
) -> None:
    """
    Broadcast aggregated weights and wait for all nodes to apply them.

    Args:
        publisher_socket: Bound PUB socket.
        nodes: List of node objects (each must expose `.socket` and `.model`).
        aggregated_weights: List of aggregated NumPy arrays.
        retries: Polling attempts per node.
        timeout_ms: Poll timeout per attempt.
    """
    payload = serialize_weights(aggregated_weights)

    try:
        publisher_socket.send_string(payload)
        logging.info("Broadcasted aggregated weights to all subscribers.")
    except Exception as e:
        logging.error(f"Failed to broadcast weights: {e}")
        return

    threads = [
        receive_weights_for_node(node, retries=retries, timeout_ms=timeout_ms)
        for node in nodes
    ]

    for t in threads:
        t.join()


def sync_subscribers(
    context: zmq.Context,
    expected_count: int,
    sync_port: int = 5560,
) -> None:
    """
    Barrier synchronization: wait until all expected subscribers
    signal readiness before training begins.

    This uses a simple REQ/REP handshake.

    Args:
        context: Active ZMQ context.
        expected_count: Number of subscribers to wait for.
        sync_port: TCP port for the synchronization channel.
    """
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{sync_port}")

    logging.info(
        f"Waiting for {expected_count} subscriber(s) to synchronize..."
    )

    for i in range(expected_count):
        socket.recv()
        logging.info(
            f"Received sync signal {i + 1}/{expected_count}"
        )
        socket.send(b"")

    socket.close()
    logging.info("All subscribers synchronized. Ready to proceed.")


# ================================================================
# MANUAL TEST ENTRY POINT (OPTIONAL)
# ================================================================

def main() -> None:
    """
    Minimal manual smoke test for socket setup.

    This is NOT part of the training pipeline.
    """
    ctx = zmq.Context()

    pub = setup_publisher(ctx, port=5555)
    subs = setup_subscribers(ctx, ports=[5556, 5557])

    logging.info("Communication module initialized (manual test).")

    for s in subs:
        s.close()
    pub.close()
    ctx.term()


if __name__ == "__main__":
    main()
