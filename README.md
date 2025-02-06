# ProMethylNet
ProMethylNet is an advanced deep learning framework designed for the accurate prediction of protein methylation sites. Protein methylation is a key post-translational modification (PTM) that influences gene regulation, protein function, and signal transduction pathways. Identifying methylation sites is critical for understanding protein mechanisms and disease pathways.

# Model
## Model Architecture
![Architecture Of ProMethylNet](Images/Model_structure_diagram/globalstructurechart.png)

ProMethylNet integrates multiple sophisticated components to capture complex sequence-structure-function relationships:

1. **Multi-Scale Convolution-Attention Network (MSCANet):**  
   - Utilizes convolutional layers with kernel sizes of 3, 5, and 7 to extract short-, medium-, and long-range sequence dependencies.
   - Incorporates multi-head self-attention mechanisms to capture global contextual features.
   - Enhances feature diversity and hierarchical representation by combining convolutional and attention-based feature extraction.

2. **Graph Neural Network (GNN):**  
   - Employs Graph Attention Networks (GAT) to model long-range sequence correlations.
   - Constructs adjacency matrices to represent local dependencies, enabling the model to interpret interactions between non-adjacent residues.
   - Utilizes multi-head attention for improved feature aggregation and interaction modeling.

3. **Bidirectional Long Short-Term Memory (BiLSTM):**  
   - Processes sequences in both forward and backward directions, effectively capturing temporal dependencies.
   - Enhances the model's ability to understand context and long-range interactions within protein sequences.

4. **Multi-Modal Feature Integration:**  
   - **One-Hot Encoding:** Represents amino acid sequences for basic sequence features.
   - **Physicochemical Properties:** Captures molecular characteristics such as hydrophobicity and molecular weight.
   - **Position-Specific Scoring Matrices (PSSM):** Models evolutionary conservation and mutation sensitivity.
   - **ProtBERT Embeddings:** Derived from a pre-trained protein language model, capturing semantic and contextual sequence information.

5. **Classifier:**  
   - A multi-layer feedforward neural network serves as the final classifier.
   - Optimized using weighted binary cross-entropy loss to handle class imbalances.
   - Employs early stopping mechanisms to prevent overfitting.

## Multi-Scale Convolution-Attention Network Architecture
![Multi-Scale Convolution-Attention Network Architecture](Images/Model_structure_diagram/Multiscale_Convolutional_Neural_Networks.png)

## Graph Neural Network Architecture
![Graph Neural Network Architecture](Images/Model_structure_diagram/graph_neural_network.png)

## BiLSTM Architecture
![BiLSTM Architecture](Images/Model_structure_diagram/BidirectionalLSTM.png)

# Evaluation Metrics (on an independent test set)
ProMethylNet demonstrates state-of-the-art performance across various metrics:
- **Accuracy:** 99.75%
- **Precision:** 92.48%
- **Recall:** 90.36%
- **F1 Score:** 91.41%
- **Matthews Correlation Coefficient (MCC):** 91.29%
- **Area Under the Curve (AUC):** 99.36%

## Performance Results of Ablation Experiments
| Experiment | Accuracy (%) | Recall (%) | Precision (%) | F1 (%) | AUC (%) | MCC (%) |
|------------|------------|------------|------------|------------|------------|------------|
| **Baseline** | 99.752 | 90.361 | 92.480 | 91.408 | 99.359 | 91.289 |
| **No MSCANet** | 99.396 | 76.633 | 80.917 | 78.717 | 99.322 | 78.441 |
| **No GAT** | 99.692 | 91.600 | 87.815 | 89.668 | 99.315 | 89.532 |
| **No Attention** | 99.726 | 86.187 | 94.546 | 90.173 | 99.518 | 90.134 |
| **No BiLSTM** | 83.142 | 96.810 | 7.744 | 14.342 | 98.223 | 24.760 |
| **No MSCANet + No Attention** | 99.364 | 72.747 | 81.626 | 76.931 | 99.217 | 76.741 |
| **No MSCANet + No GAT** | 99.740 | 26.947 | 4.386 | 7.544 | 66.320 | 7.663 |
| **No MSCANet + No GAT + ProtBERT_PSSM** | 99.740 | 90.989 | 91.159 | 91.074 | 99.523 | 90.942 |
| **No Attention + No BiLSTM + OneHot** | 98.689 | 53.080 | 55.245 | 54.141 | 97.681 | 53.487 |
| **Features_OneHot_Properties** | 98.805 | 63.771 | 58.236 | 60.878 | 98.373 | 60.337 |
| **Features_ProtBERT_PSSM** | 99.818 | 91.193 | 96.153 | 93.607 | 98.916 | 93.549 |

## Observations
- **Baseline achieves the highest F1 score (91.408%)**, maintaining a good balance between Precision and Recall.
- **Removing BiLSTM drastically reduces F1-score (14.342%)**, indicating its critical role in long-sequence modeling.
- **No MSCANet + No GAT significantly drops AUC (66.320%)**, showing the importance of feature extraction and graph modeling.
- **ProtBERT_PSSM enhances classification performance (F1: 93.607%)**, demonstrating the benefits of deep feature representation.

# Applications
ProMethylNet is a versatile tool applicable in various biological research domains:
- **Protein Function Analysis**
- **Disease Mechanism Research**
- **Drug Discovery and Precision Medicine**

# Conclusion
ProMethylNet represents a transformative advancement in computational biology. Its integration of multi-modal features and sophisticated neural network architectures establishes a new benchmark for protein methylation site prediction. The model's exceptional accuracy, robustness, and generalizability underscore its potential in diverse biological and medical research applications.

# Requirement
## Hardware and Software Parameters
| **Component**       | **Specification**                         |
|--------------------|-------------------------------------|
| **CPU**           | Intel(R) Core(TM) i9-14900K 3.20 GHz |
| **GPU**           | NVIDIA GeForce RTX 4090 24GB        |
| **Memory**        | 64GB                                 |
| **Disk**          | 1TB SSD                              |
| **Operating System** | Windows 10 Enterprise LTSC           |
| **CUDA**          | 11.2.2                               |
| **cuDNN**         | 8.2.0                                |
| **IDE**           | Pycharm 2024.3.1.1                  |
| **Python**        | 3.8.20                               |

### Python Libraries
| **Library**        | **Version**   |
|--------------------|--------------|
| **TensorFlow**    | 2.10.0        |
| **NumPy**         | 1.24.3        |
| **Bio**           | 1.6.2         |
| **Pandas**        | 2.0.3         |
| **Torch**         | 2.4.1         |
| **Scikit-learn**  | 1.3.2         |
| **SciPy**         | 1.10.1        |
| **Matplotlib**    | 3.7.3         |

