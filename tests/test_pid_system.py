# tests/test_pid_system.py
"""
Integration and unit tests for the DCCL (Decentralized Cooperative Control Learning) system.

Coverage:
- Model construction and basic fitting behavior.
- Byzantine-tolerant consensus aggregation.
- Node prediction and dataset handling.
- ZMQ-based communication and synchronization.
- Partial failure handling in distributed communication.

These tests are intended to be run with:
    pytest -q
"""

import os
import threading
import time

import numpy as np
import pytest
import tensorflow as tf
import zmq

from communication import (
    setup_publisher,
    setup_subscribers,
    serialize_weights,
    deserialize_weights,
    sync_subscribers,
)
from consensus import ByzantineFaultTolerance
from model import create_model
from node import Node
from visualization import plot_node_fit_status


# ================================================================
# FIXTURES
# ================================================================

@pytest.fixture(scope="session")
def zmq_context():
    context = zmq.Context()
    yield context
    context.term()


# ================================================================
# MODEL TESTS
# ================================================================

def test_model_creation():
    from keras_tuner.engine.hyperparameters import HyperParameters

    hp = HyperParameters()
    model = create_model(hp, input_shape=(2,))

    assert isinstance(model, tf.keras.Model)
    assert model.input_shape[1:] == (2,)
    assert model.output_shape[1] == 3


def test_model_fit_shape_match():
    from keras_tuner.engine.hyperparameters import HyperParameters

    hp = HyperParameters()
    X = np.random.rand(20, 2).astype(np.float32)
    y = np.random.rand(20, 3).astype(np.float32)

    model = create_model(hp, input_shape=(2,))
    history = model.fit(X, y, epochs=1, verbose=0)

    assert history is not None


def test_model_fit_shape_mismatch_should_fail():
    from keras_tuner.engine.hyperparameters import HyperParameters

    hp = HyperParameters()
    model = create_model(hp, input_shape=(2,))

    X = np.random.rand(10, 3).astype(np.float32)
    y = np.random.rand(10, 3).astype(np.float32)

    with pytest.raises(ValueError):
        model.fit(X, y, epochs=1, verbose=0)


# ================================================================
# CONSENSUS TESTS
# ================================================================

def test_consensus_aggregation():
    consensus = ByzantineFaultTolerance(total_nodes=4, fault_tolerant_nodes=1)

    weights_list = [
        [np.ones((2, 3)), np.ones((3,)) * 2],
        [np.ones((2, 3)) * 2, np.ones((3,)) * 3],
        [np.ones((2, 3)) * 3, np.ones((3,)) * 4],
        [np.ones((2, 3)) * 4, np.ones((3,)) * 5],
    ]

    histories = [
        {"val_loss": [0.5]},
        {"val_loss": [0.3]},
        {"val_loss": [0.4]},
        {"val_loss": [0.6]},
    ]

    aggregated = consensus.aggregate_weights(weights_list, histories)

    assert isinstance(aggregated, list)
    assert aggregated[0].shape == (2, 3)


# ================================================================
# NODE TESTS
# ================================================================

def test_node_prediction():
    X = np.random.rand(20, 2).astype(np.float32)
    y = np.random.rand(20, 3).astype(np.float32)

    ctx = zmq.Context()
    try:
        node = Node(0, ctx, (X, y))
        preds = node.predict()

        assert preds is not None
        assert preds.shape[1] == 3
    finally:
        ctx.term()


# ================================================================
# PLOTTING TEST
# ================================================================

def test_plot_node_fit_status(tmp_path):
    status = {
        0: "Good fit",
        1: "Overfitting",
        2: "Underfitting",
        3: "Insufficient data",
    }

    save_path = tmp_path / "fit_status.png"
    plot_node_fit_status(status, save_path=str(save_path), show=False)

    assert os.path.exists(save_path)


# ================================================================
# COMMUNICATION TESTS
# ================================================================

def test_communication_between_nodes(zmq_context):
    pub_port = 6000

    publisher = setup_publisher(zmq_context, pub_port)
    publisher.setsockopt(zmq.LINGER, 0)

    subscribers = setup_subscribers(zmq_context, [pub_port])
    for s in subscribers:
        s.setsockopt(zmq.LINGER, 0)

    # Allow connections to stabilize
    time.sleep(0.5)

    test_weights = [np.ones((2, 3)), np.ones((3,))]
    message = serialize_weights(test_weights)

    received = []

    def recv_thread(socket):
        msg = socket.recv_string()
        received.append(deserialize_weights(msg))

    threads = [
        threading.Thread(target=recv_thread, args=(s,), daemon=True)
        for s in subscribers
    ]

    for t in threads:
        t.start()

    time.sleep(0.5)
    publisher.send_string(message)
    time.sleep(0.5)

    for t in threads:
        t.join(timeout=2.0)

    assert len(received) == len(subscribers)

    publisher.close()
    for s in subscribers:
        s.close()


def test_multi_node_training_communication_sync(zmq_context):
    pub_port, pull_port, sync_port = 6010, 6011, 5560

    publisher = setup_publisher(zmq_context, pub_port)
    aggregator = zmq_context.socket(zmq.PULL)
    aggregator.bind(f"tcp://*:{pull_port}")

    nodes = []

    # Start synchronization barrier
    sync_thread = threading.Thread(
        target=sync_subscribers,
        args=(zmq_context, 3, sync_port),
        daemon=True,
    )
    sync_thread.start()

    # Create and train nodes
    for i in range(3):
        X = np.random.rand(20, 2).astype(np.float32)
        y = np.random.rand(20, 3).astype(np.float32)

        node = Node(i, zmq_context, (X, y))
        node.subscribe_to_updates(port=pub_port, sync_port=sync_port)
        node.model.fit(node.train_dataset, epochs=1, verbose=0)

        nodes.append(node)

    sync_thread.join(timeout=5.0)

    # Send local weights to aggregator
    for node in nodes:
        sock = zmq_context.socket(zmq.PUSH)
        sock.connect(f"tcp://localhost:{pull_port}")
        sock.send_pyobj(node.get_weights())
        sock.close()

    # Collect weights
    received_weights = []
    poller = zmq.Poller()
    poller.register(aggregator, zmq.POLLIN)

    deadline = time.time() + 5.0
    while time.time() < deadline and len(received_weights) < 3:
        events = dict(poller.poll(500))
        if aggregator in events:
            received_weights.append(aggregator.recv_pyobj())

    assert len(received_weights) == 3

    # Aggregate
    consensus = ByzantineFaultTolerance(3, fault_tolerant_nodes=1)
    dummy_histories = [{"val_loss": [0.3 + 0.01 * i]} for i in range(3)]

    aggregated = consensus.aggregate_weights(
        received_weights, dummy_histories
    )

    assert isinstance(aggregated, list)
    assert all(isinstance(w, np.ndarray) for w in aggregated)

    # Broadcast aggregated weights
    publisher.send_string(serialize_weights(aggregated))
    time.sleep(1.0)

    # Verify each node receives and applies update
    for node in nodes:
        node.set_socket_for_receiving(node.subscriber_socket)
        node.listen_for_update(timeout_ms=5000)

        for nw, aw in zip(node.get_weights(), aggregated):
            layer_scale = np.mean(np.abs(aw))
            tolerance = max(1e-3, layer_scale * 0.1)

            assert np.allclose(
                nw, aw, rtol=1e-2, atol=tolerance
            )

    aggregator.close()
    publisher.close()


def test_partial_communication_failure(zmq_context):
    pub_port, pull_port, sync_port = 6020, 6021, 5560

    publisher = setup_publisher(zmq_context, pub_port)
    aggregator = zmq_context.socket(zmq.PULL)
    aggregator.bind(f"tcp://*:{pull_port}")

    nodes = []

    for i in range(3):
        X = np.random.rand(20, 2).astype(np.float32)
        y = np.random.rand(20, 3).astype(np.float32)

        node = Node(i, zmq_context, (X, y))
        node.subscribe_to_updates(port=pub_port, sync_port=sync_port)
        node.model.fit(node.train_dataset, epochs=1, verbose=0)
        nodes.append(node)

    sync_subscribers(
        zmq_context, expected_count=len(nodes), sync_port=sync_port
    )

    # Send only two updates (simulate failure of node 2)
    sender = zmq_context.socket(zmq.PUSH)
    sender.connect(f"tcp://localhost:{pull_port}")
    sender.send_pyobj(nodes[0].get_weights())
    sender.send_pyobj(nodes[1].get_weights())
    sender.close()

    # Collect received updates
    received_weights = []
    poller = zmq.Poller()
    poller.register(aggregator, zmq.POLLIN)

    deadline = time.time() + 5.0
    while time.time() < deadline and len(received_weights) < 2:
        events = dict(poller.poll(500))
        if aggregator in events:
            received_weights.append(aggregator.recv_pyobj())

    assert len(received_weights) == 2

    consensus = ByzantineFaultTolerance(2, fault_tolerant_nodes=1)
    aggregated = consensus.aggregate_weights(
        received_weights, [{"val_loss": [0.3]}] * 2
    )

    publisher.send_string(serialize_weights(aggregated))
    time.sleep(2.0)

    # Nodes apply updates; skipped node may time out
    for node in nodes:
        node.set_socket_for_receiving(node.subscriber_socket)

        try:
            node.listen_for_update(timeout_ms=5000)
        except TimeoutError:
            # Expected for the skipped node
            continue

        for nw, aw in zip(node.get_weights(), aggregated):
            assert np.allclose(nw, aw, atol=1e-1)

    aggregator.close()
    publisher.close()
