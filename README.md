# Model analysis for hybrid quantum-classical architectures

This repository contains several Jupyter notebooks used to design, train and evaluate multiple machine learning models on the RetinaMNIST dataset. The goal is to compare classical convolutional networks with hybrid quantum-classical neural networks (HQNN), parallel hybrid architectures, and quantum feature extractors such as quanvolution filters and amplitude embedding circuits.

Technologies used:
- PyTorch
- PennyLane (quantum simulation, QNodes)
- MedMNIST (RetinaMNIST dataset)


## Repository structure

### 1.Basic_HQNN vs CNN.ipynb
Compares the performance of two baseline models:
- Standard CNN classifier
- Basic HQNN where a small quantum circuit is used as a feature transformation layer at the end
In different training situations,starting from 10% of the dataset till 100% 

### 2.Resnet.ipynb
Implements a lightweight ResNet-style architecture adapted for small grayscale images.  
Serves as a stronger classical baseline to compare with hybrid and quantum-enhanced models.


###  3.Quanvolution.ipynb
The first stage of the network replaces a standard convolution layer with a *quanvolution layer*: image patches are flattened and passed through a small quantum circuit. The quantum outputs act as learned feature channels.

After this quantum front-end, a normal CNN continues the feature extraction and performs classification. The model structure is:

- Quantum layer (AngleEmbedding + StronglyEntanglingLayers)
- CNN feature extractor (Conv → ReLU → BatchNorm → Pool)
- Final linear classifier


###  4.Amplitude_Embedding.ipynb
This notebook builds a hybrid model where a CNN extracts features and compresses them into a vector that can be encoded into a quantum state using amplitude embedding. The quantum circuit replaces the final classifier layer.The model structure is:

- CNN feature extractor
- Linear layer → reduces to 2ⁿ features
- Amplitude-embedded quantum layer producing the final outputs



###  5.Parallel_quantum_model.ipynb
This notebook builds a parallel hybrid model where a CNN produces a compact feature vector that is then split and processed by multiple quantum circuits in parallel. Each quantum block receives a slice of the CNN output, processes it through a variational circuit, and returns quantum features. The outputs of all quantum paths are then combined and passed through a final classifier.

- CNN feature extractor → 32-dim vector
- Linear compression to 6 features
- Split into 3 parts → each passed through its own quantum layer
- Concatenated quantum outputs → final linear classifier


###  6.Parallel pathing CNN+HQNN.ipynb
This notebook builds a hybrid model where a CNN produces a 128-dim feature vector that is sent through two parallel heads: a classical linear classifier and a quantum classifier. The quantum head first reduces the features, feeds them into an AngleEmbedding + entangler circuit, and outputs quantum-derived logits. The model is designed to combine both paths through a residual fusion mechanism.

- CNN feature extractor → 128-dim vector
- Classical head: linear projection to class logits
- Quantum head: linear → quantum circuit → quantum logits
- Residual fusion: classical output + α · quantum output (ablation uses classical only)


### Results and visualization

#### View_results.ipynb
Centralizes training results and architecture visualizations from all models.  
Includes:
- Accuracy plots
- Loss curves
- Architecture diagrams



## Goals of the project

The notebooks aim to determine:
1. Whether quantum layers improve performance on small datasets.
2. How classical CNNs compare with HQNN models.
3. Whether parallel hybrid approaches provide additional performance gains.
4. How effective quantum feature extractors (quanvolution, amplitude embedding) are in practice.


## Dataset

All notebooks use RetinaMNIST from the MedMNIST collection.  
This dataset contains small retinal images labeled for classification tasks.


## How to run

### Install dependencies
```bash
pip install -r requirements.txt


