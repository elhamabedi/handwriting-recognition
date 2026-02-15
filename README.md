## Persian Handwritten Digit Recognition with Neural Networks

This project implements a fully connected feedforward neural network to recognize Persian handwritten digits (0-9) using a completely original dataset created by the author. Unlike typical implementations that rely on TensorFlow or PyTorch, this solution demonstrates deep understanding of neural network fundamentals by implementing:


+ Forward and backward propagation from scratch


+ Xavier/He weight initialization


+ Label smoothing for regularization


+ Comprehensive training/validation monitoring


+ Overfitting analysis through learning curves


The system achieves 95% validation accuracy on Persian digit recognition with robust preprocessing specifically designed for Persian script characteristics.

## Technical Implementation
### Core Components

```
class SimpleNeuralNet:
    def __init__(self, input_size, hidden_sizes, output_size, init_method='xavier')
    def relu(self, x)               # ReLU activation with proper derivative
    def softmax(self, x)            # Numerically stable softmax implementation
    def forward(self, X)            # Full forward propagation with activation storage
    def compute_loss(self, y_pred, y_true, label_smoothing=0.0)
    def backward(self, y_true, label_smoothing=0.0)  # Complete backpropagation
    def train(self, X, y, epochs, lr, label_smoothing, X_val, y_val)
    def predict(self, X)
```

### Critical Design Choices

+ Input dimension: 784 (28×28 normalized grayscale images)
+ Architecture: 784 → 64 → 32 → 10 neurons (configurable hidden layers)
+ Activation functions: ReLU in hidden layers, Softmax in output layer
+ Loss function: Categorical cross-entropy with label smoothing (α=0.1)
+ Optimization: Full-batch gradient descent with configurable learning rate
+ Regularization: Label smoothing to mitigate overfitting without dropout


### Dataset
This project features a completely original dataset of 100 Persian digit samples (10 samples per digit class 0-9) personally handwritten by the author specifically for this neural networks course project. 
#### Model Architecture
```
Input Layer (784)
       ↓
[Linear: 784→64] → ReLU → [Dropout equivalent via label smoothing]
       ↓
[Linear: 64→32]  → ReLU → [Dropout equivalent via label smoothing]
       ↓
[Linear: 32→10]  → Softmax
       ↓
Output (10-class probability distribution)
```

### Results

#### Performance Metrics:

###### Training Accuracy: 97.50%


###### Validation Accuracy: 95.00%


###### Performance Gap: 2.5% (minimal overfitting)


###### Convergence: Stable within 400 epochs


### Directory Structure
```
Handwriting Recognition/
├── data/                 # Original handwritten dataset (subfolders 0-9)
│   ├── 0/                # 10 personally handwritten samples of digit 0
│   ├── 1/                # 10 personally handwritten samples of digit 1
│   └── ...               # ...continuing through digit 9
├── test_input/           # Sample images for testing
├── handWriting.ipynb     # Main implementation notebook
├── README.md             # This file
```

### Project Structure
```
handWriting.ipynb
├── Data Loading & Preprocessing
│   ├── preprocess_image()       # Persian digit normalization pipeline
│   └── load_dataset()           # Dataset ingestion with stratification
├── Neural Network Core
│   ├── SimpleNeuralNet class    # Complete NN implementation
│   │   ├── __init__()           # Architecture configuration
│   │   ├── forward()            # Forward propagation
│   │   ├── backward()           # Backpropagation
│   │   └── train()              # Training loop with validation
├── Evaluation Tools
│   ├── analyze_overfitting()    # Performance gap analysis
│   ├── plot_overfitting_analysis()  # Learning curves visualization
│   └── test_model_on_samples()  # Real-world sample testing
└── Main Execution Block          # End-to-end pipeline demonstration
```

© 2025 Neural Networks Course Project | Persian Handwritten Digit Recognition
Implementation from scratch with original handwritten dataset created by Elham Abedi
