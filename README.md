# Neural Network from Scratch

<p align="center">
  <img src="docs/visuals/architecture.svg" alt="Neural Network Architecture" width="900"/>
</p>

<h1 align="center">A Neural Network Built From First Principles</h1>

<p align="center">
  <strong>NumPy • Manual Backpropagation • BatchNorm • Dropout • Adam • MNIST</strong><br/>
  A framework-free implementation focused on understanding what happens inside a neural network.
</p>

<p align="center">
  <a href="https://github.com/chamanvashishth/neural-net-scratch">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/NumPy-only-013243?logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/MNIST-10%20Classes-6f42c1" alt="MNIST"/>
  <img src="https://img.shields.io/badge/Parameters-235%2C146-8250df" alt="Parameters"/>
  <img src="https://img.shields.io/badge/Reported%20Accuracy-97.4%25-2ea44f" alt="Reported Accuracy"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="MIT License"/>
</p>

<p align="center">
  <em>No PyTorch. No TensorFlow. No high-level neural-network API.</em>
</p>

---

## What is this?

This repository is a **fully connected neural network implemented from scratch with NumPy** for handwritten-digit classification on MNIST.

Instead of hiding the learning process behind a framework, the project implements the core pieces explicitly:

**data → tensors → layers → forward pass → loss → gradients → backpropagation → optimizer → checkpoint → evaluation**

The project is intentionally small enough to inspect, but complete enough to train, save, evaluate, and diagnose a real model.

> **Core idea:** don't just use a neural network — understand the machinery that makes it learn.

---

## At a Glance

| Area | Implementation |
|---|---|
| Dataset | MNIST |
| Task | 10-class image classification |
| Input | 28×28 grayscale → 784 features |
| Architecture | 784 → 256 → 128 → 10 |
| Activations | ReLU + Stable Softmax |
| Normalization | Batch Normalization |
| Regularization | Inverted Dropout |
| Loss | Cross-Entropy |
| Optimizer | Adam |
| Initialization | He / Kaiming |
| Trainable parameters | **235,146** |
| Training epochs | 50 |
| Batch size | 128 |
| Validation | Held-out validation split |
| Test set | 10,000 samples |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Frameworks | **None** |
| Numerical stack | NumPy |

---

## Project Philosophy

This project was built around a simple question:

> **What actually happens inside `model.fit()`?**

That question led to implementing the important pieces manually rather than relying on a deep-learning framework.

### The implementation exposes

- matrix multiplication and tensor shapes
- parameter initialization
- activation functions
- forward propagation
- loss computation
- gradient computation
- reverse-mode backpropagation
- BatchNorm training/inference behavior
- inverted Dropout
- numerical stabilization
- Adam moment estimation
- learning-rate decay
- checkpoint selection
- independent test evaluation
- error visualization

This makes the repository useful both as an **ML implementation project** and as a compact reference for the mechanics of neural-network training.

---

## System Overview

<p align="center">
  <img src="docs/visuals/training-loop.svg" alt="Training loop"/>
</p>

The complete lifecycle is:

```text
                  ┌─────────────────────┐
                  │       MNIST         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Preprocess / Batch  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Forward Pass    │
                  │ Dense → ReLU → BN   │
                  │ Dropout → Dense ... │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Cross-Entropy     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Manual Backprop     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Adam Update      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Validation Check    │
                  │ Save best checkpoint│
                  └─────────────────────┘
```

---

## Model Architecture

<p align="center">
  <img src="docs/visuals/architecture.svg" alt="Detailed neural network architecture"/>
</p>

### Forward path

```text
28×28 MNIST image
       │
       ▼
Flatten + Normalize
       │
       ▼
Dense: 784 → 256
       │
       ▼
ReLU
       │
       ▼
BatchNorm
       │
       ▼
Dropout (p=0.30)
       │
       ▼
Dense: 256 → 128
       │
       ▼
ReLU
       │
       ▼
BatchNorm
       │
       ▼
Dropout (p=0.20)
       │
       ▼
Dense: 128 → 10
       │
       ▼
Numerically Stable Softmax
       │
       ▼
Class probabilities
```

### Parameter scale

The network contains **235,146 trainable parameters** distributed across the three dense layers and normalization parameters.

The architecture is deliberately compact: large enough to demonstrate real optimization behavior, small enough to inspect mathematically.

---

## Training Configuration

```text
Epochs:              50
Batch size:          128
Initial learning rate: 0.001
Adam β1:              0.9
Adam β2:              0.999
Adam ε:               1e-8
Learning-rate decay:  0.95 every 10 epochs
Random seed:          42
```

### Data separation

The training process keeps the roles of the splits explicit:

```text
                 MNIST
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    TRAIN         VAL         TEST
       │           │           │
       │           │           └── Final measurement
       │           └────────────── Model selection
       └────────────────────────── Parameter updates
```

The best checkpoint is selected from validation performance. The test set is reserved for final evaluation.

---

## The Mathematics

### Dense layer

```text
Z = XW + b
```

### ReLU

```text
ReLU(z) = max(0, z)
```

### Batch Normalization

```text
x̂ = (x - μ) / √(σ² + ε)
y  = γx̂ + β
```

### Inverted Dropout

During training, activations are randomly masked and scaled by the inverse keep probability.

This keeps the expected activation scale consistent between training and inference.

### Stable Softmax

For logits `z`, the implementation stabilizes exponentiation using the maximum logit:

```text
z'ᵢ = zᵢ - max(z)

softmax(zᵢ) = exp(z'ᵢ) / Σⱼ exp(z'ⱼ)
```

This prevents large logits from causing exponential overflow.

### Cross-Entropy

```text
L = -Σ y log(ŷ)
```

For a mini-batch:

```text
L = -(1/m) Σₙ Σₖ yₙₖ log(ŷₙₖ)
```

---

## Manual Backpropagation

<p align="center">
  <img src="docs/visuals/gradient-flow.svg" alt="Manual backpropagation"/>
</p>

The gradients are propagated explicitly through the computational graph.

For a dense layer:

```text
dW = Aᵀδ
db = Σδ
dA = δWᵀ
```

For Softmax + Cross-Entropy with one-hot labels:

```text
δ = (ŷ - y) / m
```

The backward path covers:

```text
Loss
 ↓
Softmax + Cross-Entropy
 ↓
Dense
 ↓
Dropout
 ↓
BatchNorm
 ↓
ReLU
 ↓
Dense
 ↓
Dropout
 ↓
BatchNorm
 ↓
ReLU
 ↓
Dense
```

No automatic differentiation engine is responsible for these gradients.

---

## Adam Optimizer

Adam maintains first- and second-moment estimates of the gradients.

```text
mₜ = β₁mₜ₋₁ + (1 - β₁)gₜ
vₜ = β₂vₜ₋₁ + (1 - β₂)gₜ²

m̂ₜ = mₜ / (1 - β₁ᵗ)
v̂ₜ = vₜ / (1 - β₂ᵗ)

θ ← θ - α · m̂ₜ / (√v̂ₜ + ε)
```

The optimizer is implemented directly in the repository rather than imported from a framework.

---

## Evaluation Pipeline

<p align="center">
  <img src="docs/visuals/evaluation-pipeline.svg" alt="Independent test evaluation"/>
</p>

Run:

```bash
python evaluate.py
```

The evaluator:

1. loads the held-out MNIST test split
2. loads the saved checkpoint
3. performs inference with training behavior disabled
4. generates predictions for all test samples
5. computes classification metrics
6. computes per-class F1
7. builds the confusion matrix
8. writes `results/evaluation.json`

### Metrics

| Metric | What it answers |
|---|---|
| Accuracy | How often is the predicted class correct? |
| Precision | When the model predicts a class, how often is it correct? |
| Recall | How much of each class does the model recover? |
| Macro F1 | How balanced is performance across classes? |
| Confusion Matrix | Which digits does the model confuse? |

---

## Reported Benchmark

The repository currently reports:

| Metric | Reported |
|---|---:|
| **Test Accuracy** | **97.4%** |
| Macro Precision | 97.0% |
| Macro Recall | 97.0% |
| Macro F1 | 97.0% |
| Best reported class | Digit 1 — 99.2% F1 |
| Lowest reported class | Digit 5 — 95.5% F1 |
| Trainable parameters | 235,146 |

> **Important:** these are repository-reported benchmark values. Run the evaluation locally to verify the current checkpoint rather than treating README numbers as immutable results.

---

## Reproducibility

The project keeps training and evaluation deterministic where practical through a fixed NumPy seed:

```python
np.random.seed(42)
```

### Reproduce

```bash
git clone https://github.com/chamanvashishth/neural-net-scratch.git
cd neural-net-scratch

pip install -r requirements.txt

python train.py
python evaluate.py
python visualize.py
```

MNIST is downloaded automatically by the data loader when required.

Generated artifacts are written under:

```text
results/
├── history.json
├── evaluation.json
├── loss_curve.png
├── accuracy_curve.png
├── confusion_matrix.png
├── weight_distributions.png
├── sample_predictions.png
└── per_class_f1.png
```

---

## Diagnostics

The visualization pipeline exposes several different failure modes instead of relying on accuracy alone.

```bash
python visualize.py
```

### Training behavior

- training loss
- validation loss
- training accuracy
- validation accuracy
- best validation epoch

### Classification behavior

- normalized confusion matrix
- per-class F1
- sample-level predictions
- prediction confidence

### Parameter behavior

- initial weight distributions
- final weight distributions

This makes it possible to ask not only **"How accurate is the model?"**, but also:

> **"How is it learning, where does it fail, and what changed inside the parameters?"**

---

## Implementation Map

| File | Responsibility |
|---|---|
| `src/layers.py` | Dense, ReLU, BatchNorm, Dropout, Softmax |
| `src/losses.py` | Cross-Entropy and numerical stability |
| `src/optimizers.py` | Adam and parameter updates |
| `src/network.py` | Network composition, forward/backward/update |
| `src/data_loader.py` | MNIST download, preprocessing and batching |
| `src/metrics.py` | Accuracy, precision, recall, F1, confusion matrix |
| `train.py` | Training loop, validation and checkpoint selection |
| `evaluate.py` | Final held-out test evaluation |
| `visualize.py` | Training and evaluation diagnostics |
| `notebooks/math_derivations.ipynb` | Mathematical derivations |

---

## Repository Structure

```text
neural-net-scratch/
│
├── src/
│   ├── layers.py
│   ├── losses.py
│   ├── optimizers.py
│   ├── network.py
│   ├── data_loader.py
│   └── metrics.py
│
├── notebooks/
│   └── math_derivations.ipynb
│
├── docs/
│   └── visuals/
│       ├── architecture.svg
│       ├── training-loop.svg
│       ├── evaluation-pipeline.svg
│       └── gradient-flow.svg
│
├── checkpoints/
│   └── saved model parameters
│
├── results/
│   ├── history.json
│   ├── evaluation.json
│   └── generated diagnostics
│
├── train.py
├── evaluate.py
├── visualize.py
├── requirements.txt
└── README.md
```

---

## Engineering Decisions

### Framework-free core

The neural-network implementation uses NumPy for numerical computation rather than PyTorch or TensorFlow.

### Validation-based checkpointing

The training loop saves the model whenever validation accuracy improves.

### Separate inference path

Evaluation calls the model in inference mode so training-specific behavior such as Dropout is not applied.

### Numerical stability

Softmax is stabilized before exponentiation and Cross-Entropy uses the resulting probabilities safely.

### Inspectable state

Training history and evaluation results are persisted as JSON, while model parameters are saved as checkpoints.

---

## What This Project Demonstrates

This project demonstrates implementation-level understanding of:

- neural-network architecture
- matrix and tensor operations
- forward propagation
- reverse-mode gradient propagation
- numerical stability
- normalization
- regularization
- adaptive optimization
- checkpointing
- validation methodology
- held-out test evaluation
- classification metrics
- error analysis
- experiment visualization

More importantly, it demonstrates the ability to move below the abstraction layer of a high-level ML framework and implement the learning mechanics directly.

---

## Limitations

This is a **fully connected MNIST classifier**, not a production vision system.

It intentionally does not attempt to compete with modern convolutional or transformer-based vision architectures.

The project is designed to optimize for:

- transparency
- mathematical understanding
- inspectability
- reproducibility
- implementation depth

rather than benchmark chasing.

---

## Possible Extensions

Natural next experiments include:

- finite-difference gradient checking
- ablation of BatchNorm
- ablation of Dropout
- optimizer comparisons
- systematic hyperparameter sweeps
- confidence calibration
- harder datasets
- convolutional layers from scratch
- experiment tracking
- error clustering and failure analysis

These extensions would turn the repository from a single implementation into a broader experimental ML laboratory.

---

## Quick Start

### Install

```bash
git clone https://github.com/chamanvashishth/neural-net-scratch.git
cd neural-net-scratch
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

### Evaluate

```bash
python evaluate.py
```

### Visualize

```bash
python visualize.py
```

---

## Dependencies

```text
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
requests>=2.28
tqdm>=4.65
```

The model implementation itself does not use PyTorch, TensorFlow, or a high-level neural-network training API.

---

## License

This project is licensed under the **MIT License**.

---

## Author

<p align="center">
  <strong>Chaman Vashishth</strong><br/>
  AI/ML Engineering • Machine Learning Systems • Applied AI
</p>

<p align="center">
  <a href="https://github.com/chamanvashishth">GitHub</a> •
  <a href="https://www.linkedin.com/in/chamanvashishth">LinkedIn</a>
</p>

---

<p align="center">
  <strong>Built from first principles with NumPy.</strong><br/>
  <sub>Understand the gradient. Understand the model.</sub>
</p>
