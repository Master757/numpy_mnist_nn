#  MNIST Classifier from Scratch (NumPy + Custom Backpropagation)

This project implements a neural network **from scratch using only NumPy**, trained on the MNIST handwritten digits dataset. It **does not use any high-level deep learning frameworks** like Keras or PyTorch for the model logic — everything is built manually to learn the internals of how neural networks work.

---

## Key Features

- Built using **only NumPy** (except for data loading via TensorFlow/Keras)
- Implements:
  - Fully connected layers
  - ReLU activation
  - Softmax output
  - Cross-entropy loss (with optional label smoothing)
  - Manual **backpropagation**
- Trains and evaluates on the **MNIST** dataset
- Includes both **Jupyter Notebook** and **.py script** versions
- Clean and modular OOP structure

---

##  Files in This Repo

| File | Description |
|------|-------------|
| `mnist_numpy_nn.ipynb` | Interactive notebook version with code + output |
| `mnist_numpy_nn.py` | Standalone script version (for command-line execution) |
| `README.md` | This readme |
| `LICENSE` | MIT license (open-source) |

---

## How It Works

1. **Data Loading**:  
   Uses `tf.keras.datasets.mnist.load_data()` to fetch and normalize the dataset.

2. **Model Architecture**:
   - Input layer: 784 units (28×28 flattened)
   - Hidden layer: 128 units, ReLU activation
   - Output layer: 10 units (Softmax for classification)

3. **Training**:
   - Mini-batch gradient descent
   - Manual weight updates via gradients
   - Label smoothing is optionally applied for better generalization

4. **Evaluation**:
   - Accuracy is computed on both the validation and test datasets

---

Install using:

```bash
pip install numpy tensorflow
```

sample output : -
  Validation Accuracy: 92.34%
  Test Accuracy: 92.19%
