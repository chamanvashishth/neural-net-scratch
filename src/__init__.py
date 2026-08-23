"""Neural Network From Scratch (NumPy Only)."""

from .layers import Dense, ReLU, BatchNorm, Dropout, Softmax
from .losses import CrossEntropyLoss
from .optimizers import Adam
from .network import NeuralNetwork
from .data_loader import load_mnist, one_hot, iterate_minibatches
from .metrics import accuracy, confusion_matrix, precision_recall_f1

__version__ = "1.0.0"

__all__ = [
    "Dense",
    "ReLU",
    "BatchNorm",
    "Dropout",
    "Softmax",
    "CrossEntropyLoss",
    "Adam",
    "NeuralNetwork",
    "load_mnist",
    "one_hot",
    "iterate_minibatches",
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
]
