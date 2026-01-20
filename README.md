# DCCL: Decentralized Cooperative Control Learning

In industrial automation, setting the right PID gains ($K_p, K_i, K_d$) is often a manual, time-consuming process. **DCCL** is a distributed machine learning framework designed to automate the tuning of industrial PID controllers. It uses **Decentralized Federated Learning** to train neural networks that can predict these optimal gains automatically. 
 
**The Goal:** 
Nodes (controllers) work together to build a high-performance "global brain" for PID tuning without ever sharing their local, sensitive telemetry data.

*   **Input:** Performance metrics of a system (e.g., Settling Time and Integrated Absolute Error).
*   **Output:** The optimal PID gains ($K_p, K_i, K_d$) required to stabilize that system.
*   **Environment:** This is a **simulated distributed system**. While the code runs on a single machine for demonstration, it uses **ZeroMQ (ZMQ)** to mimic real-world network communication between independent nodes, making it ready for deployment across actual edge hardware.

---


## Dataset


The project is designed to work directly with:
data/pid_dataset.xlsx

This dataset must contain at least these columns:

```text
Iteration | Kp[1-4] | Ki[1-4] | Kd[1-4] | St[1-4] | IAE[1-4]

```
*   **Iteration:** The chronological step of the live experiment(500 in pid_dataset.xlsx).
*   **Kp, Ki, Kd (Targets):** The Proportional, Integral, and Derivative gains used by the controllers. The model learns to predict these three values.
*   **St (Settling Time):** A performance metric measuring how long the system takes to reach a stable state.
*   **IAE (Integral Absolute Error):** A metric representing the cumulative error over time; lower values indicate higher precision.

### the data structure
In a real-world deployment, the **St** and **IAE** columns serve as the **features (Inputs)**. When a controller experiences a high Settling Time or high Error, the trained DCCL model analyzes these inputs to suggest the optimized **Kp, Ki, and Kd (Outputs)** needed to correct the system's behavior.

---

Your code will do:
1.  **Cleaning:** Handles missing values via mean/mode imputation and partitions data into local node-specific chunks.
2.  **Engineering:** Derives "Deltas" (change over time), performance ratios ($IAE/St$), and applies Moving Averages to filter sensor noise.
3.  **Stability Labeling:** Automatically classifies system health (Excellent to Unstable) for multi-task learning.
4.  **Decentralized Training:** Local data stays on the node; only model weights are serialized and exchanged via ZMQ to create a robust "Global Brain."

---
## Project Structure
The repository is organized into a modular pipeline following production standards:

```text
outputs/
│   └── Training Loss.png
│
src/
├── main.py             # System orchestrator: manages training rounds and node lifecycles.
├── node.py             # Node abstraction: handles local datasets and model states.
├── consensus.py        # Robust Aggregator: filters outliers and merges weights (BFT).
├── communication.py    # Transport layer: ZMQ socket logic for PUB/SUB and PUSH/PULL.
├── model.py            # Architecture: defines MLP/LSTM models and Keras Tuner logic.
├── training.py         # Logic for local gradient descent and evaluation.
├── preprocessing.py    # Feature engineering: transforms raw PID logs into ML-ready tensors.
├── transfer_learning.py# Weight initialization: handles pretraining and parameter reuse.
├── config.py           # Configuration: centralized hyperparameters and port mapping.
└── visualization.py    # Analytics: generates prediction curves and fit-status reports.
│
tests/
└── test_pid_system.py  # Integration tests: validates ZMQ comms and aggregation logic.
```

---

##  Key Engineering Challenges Addressed
*   **Byzantine Fault Tolerance:** In a real factory, one sensor might be broken or "noisy." The `consensus.py` module implements a robust aggregator that statistically identifies and ignores updates from faulty nodes.
*   **Network Resilience:** Using ZeroMQ, the system handles asynchronous communication. It includes a synchronization barrier to ensure that all nodes are aligned before a new global tuning round begins.
*   **Model Heterogeneity:** Not every controller has the same data. The system includes **Automated Hyperparameter Tuning** (via Keras Tuner) so each node can independently determine if it needs a simple MLP or a complex LSTM architecture.
*   **Feature Engineering:** Raw control logs are noisy. `preprocessing.py` derives specialized metrics like IAE-ratios and moving averages to help the neural network converge faster.

---

## Technical Stack
*   **Frameworks:** TensorFlow / Keras
*   **Distributed Comms:** ZeroMQ (ZMQ)
*   **Optimization:** Keras Tuner (Hyperband)
*   **Data Science:** NumPy, Pandas, Scikit-learn
*   **DevOps/QA:** PyTest (Asynchronous networking tests)

---

## Getting Started

**1. Installation**
```bash
pip install tensorflow zmq pandas scikit-learn matplotlib keras-tuner openpyxl
```

**2. Run the Simulation**
This will launch 4 virtual nodes that communicate over local TCP ports, train on the included dataset, and synchronize their models.
```bash
python -m src.main --tuning auto --max_trials 20
```

**3. Run Tests**
Validate the serialization and peer-to-peer weight exchange logic:
```bash
pytest tests/test_pid_system.py
```

---

