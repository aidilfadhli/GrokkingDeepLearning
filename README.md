# Grokking Deep Learning
## Code Reproduction & Theoretical Deep-Dive
**Enrichment Assignment 2 — Deep Learning Course**

---

### Student Identity
- **Name**  : Muhammad Aidil Fadhli Awalludin
- **NIM**   : 101032300087
- **Class** : TK4701

---

## 1. Overview & Objectives

This repository contains the complete code reproduction and structured theoretical deep-dive of the book **"Grokking Deep Learning"** by **Andrew W. Trask** (Manning Publications).

The primary objective of this assignment is to deepen theoretical intuition and practical implementation skills in artificial neural networks from **first principles**. Every neural network architecture, forward pass, error objective, gradient calculation, activation function, and backpropagation routine in this repository is implemented using **pure Python and NumPy**, without reliance on high-level black-box frameworks (such as PyTorch, TensorFlow, or Keras). This ensures complete transparency of data flow and matrix calculus at every layer of computation.

---

## 2. Chapter Roadmap & Status

| Chapter | Notebook File | Primary Topic | Core Theoretical & Practical Focus | Status |
|:---:|:---|:---|:---|:---:|
| **01** | [`01.introduction.ipynb`](./01.introduction.ipynb) | Introducing Deep Learning | AI vs. ML vs. DL hierarchies, representation learning, rule-based vs. data-driven paradigms. | **Completed** |
| **02** | [`02.Fundamental-Concepts.ipynb`](./02.Fundamental-Concepts.ipynb) | Fundamental Concepts | The *Predict, Compare, Learn* cycle, practical linear algebra, dot products, matrices from scratch. | **Completed** |
| **03** | [`03.making-predictions.ipynb`](./03.making-predictions.ipynb) | Forward Propagation | Single/multi input & output topologies, weights as volume knobs, deep hidden layer forward pass. | **Completed** |
| **04** | [`04.gradient-descent.ipynb`](./04.gradient-descent.ipynb) | Gradient Descent | Mean Squared Error (MSE), limitations of *Hot & Cold* learning, analytical derivatives, learning rate ($\alpha$). | **Completed** |
| **05** | [`05.generalizing-gradient-descent.ipynb`](./05.generalizing-gradient-descent.ipynb) | Generalizing Gradient Descent | Multi-weight optimization, error bowls, frozen weights, matrix updates via vector outer products. | **Completed** |
| **06** | [`06.backpropagation.ipynb`](./06.backpropagation.ipynb) | Backpropagation | Non-linear limits (Streetlights/XOR), ReLU activation, Chain Rule, 2-layer Deep Network from scratch. | **Completed** |
| **07** | [`07.picture-perfect-mnist.ipynb`](./07.picture-perfect-mnist.ipynb) | Computer Vision & MNIST | Flattening $28 \times 28$ image matrices, one-hot label encoding, 10-digit classification with SGD. | **Completed** |
| **08** | [`08.regularization-and-dropout.ipynb`](./08.regularization-and-dropout.ipynb) | Regularization & Dropout | Overfitting vs. underfitting, signal vs. noise, L2 weight decay, inverted Dropout mask from scratch. | **Completed** |
| **09** | [`09.activation-functions.ipynb`](./09.activation-functions.ipynb) | Activation Functions | Probability modeling, Sigmoid, vanishing gradients, numerically stable Softmax, Cross-Entropy Loss. | **Completed** |
| **10** | [`10.convolutional-neural-networks.ipynb`](./10.convolutional-neural-networks.ipynb) | Convolutional Neural Networks | Spatial locality, weight sharing, 2D convolution, Max Pooling, and a train-ready Mini-CNN from scratch. | **Completed** |

---

## 3. General Summary of Chapters

### Chapter 1: Introducing Deep Learning — Why You Should Learn It
- **Theoretical Core**: Unpacks the transition from classical software engineering ($\text{Rules} + \text{Data} \to \text{Answers}$) to machine learning ($\text{Data} + \text{Answers} \to \text{Rules/Parameters}$). Demystifies the concentric circles of AI, classical ML (dependent on human feature engineering), and Deep Learning (end-to-end representation learning). Examines the tri-factor enabling deep learning breakthroughs: Big Data, parallel GPU hardware, and non-saturating algorithmic improvements.
- **Code Core**: Verifies Python and NumPy runtime environments and demonstrates the behavioral contrast between a fixed rule-based function and an adaptive numerical parameter learner.

### Chapter 2: Fundamental Concepts — How Machines Learn
- **Theoretical Core**: Explores learning taxonomies (Supervised, Unsupervised, Parametric vs. Non-Parametric) and establishes the universal engine of deep learning: **Predict $\to$ Compare $\to$ Learn**. Introduces practical linear algebra for neural computing, interpreting the vector dot product ($\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta$) simultaneously as a projection of geometric similarity and a weighted accumulation of evidence.
- **Code Core**: Builds vector operations, scalar multiplication, dot products, and matrix transformations from pure Python loops, contrasted against vectorized NumPy BLAS acceleration routines.

### Chapter 3: Forward Propagation — Intro to Neural Prediction
- **Theoretical Core**: Dissects the first stage of the cycle: **Predict**. Interprets neural weights as directional sensitivity knobs that scale, invert, or mute information. Explores 5 progressive network topologies: single input to single output, multi-input (weighted sum), multi-output (signal broadcasting), full matrix transformations, and multi-layer deep networks where intermediate hidden representations enable "predicting on predictions."
- **Code Core**: Evaluates a baseball toy dataset (`toes`, `wlrec`, `nfans`) across all five forward architectures using custom primitives (`w_sum`, `vect_mat_mul`) and verifies numerical equivalence against NumPy.

### Chapter 4: Introduction to Gradient Descent — Learning to Reduce Error
- **Theoretical Core**: Focuses on the **Compare** and **Learn** stages. Establishes why Mean Squared Error (MSE: $(\hat{y} - y)^2$) is preferred over absolute error (strictly positive, quadratically penalizes catastrophic errors, smooth parabolic derivative). Analyzes the failure of heuristic *Hot and Cold* perturbation ($3^N$ combinatorial explosion) and derives analytical *Gradient Descent*:
  $$\text{delta} = \hat{y} - y, \quad \text{weight\_delta} = \text{delta} \cdot \text{input}$$
  Examines how the learning rate hyperparameter ($\alpha$) balances convergence speed against catastrophic overshooting (divergence).
- **Code Core**: Compares Hot & Cold oscillation against Gradient Descent, implements one-step updates, and demonstrates exponential divergence when $\alpha$ is set too aggressively.

### Chapter 5: Generalizing Gradient Descent — Learning Multiple Weights at a Time
- **Theoretical Core**: Expands single-variable optimization into multi-dimensional parameter spaces. Explains why all weights feeding into a single neuron share the identical output `delta`, while each maintains a distinct `weight_delta` scaled by its feature input. Visualizes high-dimensional error bowls and the dynamics of "frozen weights." Derives full matrix parameter updates using the vector **Outer Product**: $\Delta \mathbf{W} = \boldsymbol{\delta} \mathbf{x}^T$.
- **Code Core**: Pure Python and NumPy multi-input gradient descent, simulated frozen-weight compensation experiments, and concurrent multi-input multi-output matrix training.

### Chapter 6: Intro to Backpropagation — Building Your First DEEP Neural Network
- **Theoretical Core**: Identifies the fundamental geometric limitation of shallow linear models on non-linear datasets (the Streetlights / XOR problem), demonstrating that stacking linear layers without activations mathematically collapses into a single linear layer: $\mathbf{W}_2 \mathbf{W}_1 \mathbf{x} = \mathbf{W}_{net} \mathbf{x}$. Introduces the Rectified Linear Unit (**ReLU**: $\max(0, z)$) and derives the **Backpropagation** algorithm via the Chain Rule, projecting error deltas back to hidden layers through transposed weight matrices ($\mathbf{W}_2^T$).
- **Code Core**: Demonstrates single-layer perceptron failure on non-linear data, followed by building and training a complete 2-layer Deep Neural Network from scratch with NumPy, achieving 100% classification accuracy and near-zero cumulative loss.

### Chapter 7: Picture Perfect — Neural Networks That Can See (MNIST)
- **Theoretical Core**: Formulates the computer vision pipeline: flattening 2D pixel grids ($28 \times 28$) into 784-dimensional vectors and scaling values to $[0.0, 1.0]$. Explains **One-Hot Encoding** for categorical target representation to prevent spurious distance relationships. Details a 3-layer architecture ($784 \to 40 \text{ ReLU} \to 10 \text{ Outputs}$) trained with **Stochastic Gradient Descent (SGD)**.
- **Code Core**: Constructs the full image classification architecture, tracks training loss and epoch accuracy, and evaluates generalization performance on an independent out-of-sample test split.

### Chapter 8: Intro to Regularization — Learning Signal and Ignoring Noise
- **Theoretical Core**: Addresses the bias-variance tradeoff and the pathology of **Overfitting**, wherein high-capacity networks memorize dataset-specific noise rather than generalizable signal. Examines Early Stopping, L2 Weight Decay, and **Dropout** (Srivastava et al., 2014) with inverted scaling ($\frac{1}{1-p}$), which randomly severs hidden activations during training to dismantle co-adaptation and simulate an ensemble of $2^N$ sub-networks.
- **Code Core**: Evaluates an unregularized baseline on a noisy synthetic dataset (observing memorization of noise), then implements Inverted Dropout from scratch, demonstrating significant gains in out-of-sample test accuracy.

### Chapter 9: Intro to Activation Functions — Modeling Probabilities
- **Theoretical Core**: Reconciles neural outputs with Kolmogorov probability axioms ($0 \le P \le 1$ and $\sum P = 1$). Derives the **Sigmoid** activation function ($\sigma(z) = \frac{1}{1 + e^{-z}}$), its derivative ($\sigma(1 - \sigma)$), and the mechanics of the Vanishing Gradient Problem. Formulates the **Softmax** function for multi-class probability normalization alongside **Categorical Cross-Entropy Loss**, deriving the elegant analytical gradient: $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$.
- **Code Core**: Implements numerically stable Sigmoid, Softmax (preventing exponential overflow via maximum logit subtraction), and Cross-Entropy loss, followed by multi-class categorical classification from scratch.

### Chapter 10: Intro to Convolutional Neural Networks — Learning Edges and Corners
- **Theoretical Core**: Analyzes why fully connected layers fail on visual imagery (parameter explosion, loss of translation invariance, and destruction of 2D spatial topology). Introduces the core principles of **Convolutional Neural Networks (CNNs)**: local receptive fields and weight sharing via sliding kernels. Explains feature map output dimension calculations, spatial downsampling via Max Pooling, and kernel gradient accumulation across patches during backpropagation.
- **Code Core**: Builds 2D convolution from scratch with sliding window dot products, demonstrates classical edge detection (Sobel filter), implements Max Pooling, and trains a Mini-ConvNet classifier using pure NumPy.

---

## 4. Repository Structure

```text
GrokkingDeepLearning/
├── README.md                                    # Main repository documentation and student identity
├── requirements.txt                             # Python dependencies specification
├── 01.introduction.ipynb                        # Chapter 1: Introducing Deep Learning & Paradigms
├── 02.Fundamental-Concepts.ipynb                # Chapter 2: Predict-Compare-Learn & Practical Linear Algebra
├── 03.making-predictions.ipynb                  # Chapter 3: Forward Propagation & Prediction Topologies
├── 04.gradient-descent.ipynb                    # Chapter 4: Error Measurement (MSE) & Gradient Descent
├── 05.generalizing-gradient-descent.ipynb       # Chapter 5: Multi-Weight GD & Matrix Outer Product Updates
├── 06.backpropagation.ipynb                     # Chapter 6: Non-Linearity (ReLU) & 2-Layer Deep Backpropagation
├── 07.picture-perfect-mnist.ipynb               # Chapter 7: Computer Vision & MNIST Handwritten Digit Classification
├── 08.regularization-and-dropout.ipynb          # Chapter 8: Overfitting Mitigation & Inverted Dropout from Scratch
├── 09.activation-functions.ipynb                # Chapter 9: Sigmoid, Softmax, & Categorical Cross-Entropy Loss
└── 10.convolutional-neural-networks.ipynb       # Chapter 10: 2D Convolution, Max Pooling, & Mini-ConvNet from Scratch
```

---

## 5. How to Run the Notebooks

### Prerequisites
Ensure Python 3.8+ is installed, then install all project dependencies:
```bash
pip install -r requirements.txt
```

### Running Locally
1. Clone or navigate to the `GrokkingDeepLearning` directory:
   ```bash
   cd GrokkingDeepLearning
   ```
2. Launch the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
3. Open your browser and select any desired chapter notebook (e.g., `01.introduction.ipynb` or `06.backpropagation.ipynb`).
4. Select **Cell $\to$ Run All** to execute all code cells and verify the numerical outputs.

### Running on Google Colab
All notebooks in this repository are formatted with standard `.ipynb` specifications and are fully compatible with **Google Colab**. You can upload the notebooks directly to Google Drive / Colab and run them with zero setup required.

---

## 6. References
1. Trask, Andrew W. (2019). *Grokking Deep Learning*. Manning Publications. ISBN: 9781617293702.
2. Goodfellow, Ian, Bengio, Yoshua, & Courville, Aaron. (2016). *Deep Learning*. MIT Press.
3. Assignment Reference Repository: [farrelrassya/GrokkingDeepLearning](https://github.com/farrelrassya/GrokkingDeepLearning).
