# ==============================================================================
# RECYCLABLE WASTE CLASSIFICATION — COMPUTER VISION RESEARCH LAB
# PROJECT README (SUBMISSION FORMAT: README.txt)
# Course: CAP 4630 - Introduction to Artificial Intelligence (Spring 2026)
# University of North Florida — School of Computing
# ==============================================================================

================================================================================
1. PROJECT OVERVIEW
================================================================================
This repository contains the full research framework and implementation code for 
a comparative study on how Convolutional Neural Network (CNN) architectural depth 
affects multi-class recyclable material classification. 

Using a localized dataset of 12,875 colored images (150x150 pixels) sourced from 
the UNF CCEC AI Laboratory, the project designs, trains, and evaluates two distinct 
models to isolate the visual feature recognition capabilities across 4 material 
classes: Glass, Metal, Paper, and Plastic.

  * Model A (Baseline): A compact 2-layer CNN utilizing 16 and 32 filters. 
    Optimized for rapid feature localization (edges, basic shapes).
  * Model B (Experimental): An expanded 3-layer CNN utilizing 32, 64, and 128 
    filters. Designed with higher capacity to extract abstract textures, micro-
    structures, and non-linear geometric patterns.

Research Outcome:
Increasing architectural depth from 2 to 3 layers successfully raised overall 
test accuracy from 49.20% to 51.22%. Notably, Model B demonstrated a localized 
performance surge of +4.14% in "Paper" and +4.10% in "Plastic", confirming that 
deeper convolutional layers are vital for capturing the highly irregular, 
deformable textures typical of compressed consumer recyclables.

================================================================================
2. DIRECTORY & FILE STRUCTURE
================================================================================
The project architecture is organized as follows:

├── data/
│   ├── train/            # Training data subset (8,641 images)
│   ├── val/              # Validation data subset (2,104 images)
│   └── test/             # Test data subset (2,130 images)
│
├── DataLoader.py         # Data preprocessing, pipeline mapping, and normalization
├── Models.py             # Architectural definitions for Model A and Model B
├── train.py              # Iterative training execution routine and weight checkpointing
├── test_model.py         # Test partition evaluator reporting macro accuracy
├── class_accuracy.py     # Class-by-class granular accuracy tracking script
├── evaluate_model.py     # Diagnostic analyzer generating research matrices and metrics
├── master_run.py         # Unified automated experimental orchestration framework
└── README.txt            # Operational guidelines and infrastructure documentation (this file)

================================================================================
3. DETAILED FILE ROLES
================================================================================

* DataLoader.py
  Handles image ingestion, scaling, and array processing. Standardizes all data 
  to 150x150 pixels, converts images to PyTorch Tensors, and applies a min-max 
  channel normalization centered around [-1.0, 1.0] across all 3 RGB channels. 
  Constructs PyTorch DataLoaders with a batch size of 32, enabling memory-efficient 
  mini-batch streaming.

* Models.py
  Houses the neural topology definitions built on torch.nn.
  - ModelA: Outlines a 2-stage feature extraction block [Conv -> ReLU -> MaxPool] 
    transitioning from 3 input channels to 16, then 32 hidden feature maps. 
    Flattens into a 128-neuron dense layer leading to a 4-class output.
  - ModelB: Implements an extended 3-stage feature block transitioning from 3 
    channels to 32, 64, and finally 128 feature maps before flattening into a 
    256-neuron dense hidden layer.

* train.py
  Executes independent training passes for both networks over a fixed 10-epoch 
  horizon. Employs the Adam optimization algorithm (learning rate = 0.001) paired 
  with a Cross-Entropy Loss objective function. Conducts post-epoch validation 
  audits and automatically serializes the final learned network states into 
  'Model_A_weights.pth' and 'Model_B_weights.pth'.

* test_model.py
  A clean testing script that re-instantiates Model B, sets its operational mode 
  to evaluation (model.eval()), maps saved weights, and computes the absolute 
  unweighted accuracy over the unseen 2,130-image test set.

* class_accuracy.py
  Iterates over the test data loader to tabulate raw counts of correct versus 
  incorrect predictions indexed per class. Prints an aligned layout detailing the 
  precise percentage-based classification accuracy achieved for each category.

* evaluate_model.py
  Calculates and prints professional scientific metrics using scikit-learn. 
  Generates a formal classification report itemizing Precision, Recall, and 
  F1-Score for each material class. Outputs a comprehensive confusion matrix 
  mapping true versus predicted classes to expose structural vulnerabilities, 
  such as low-resolution inter-class overlap.

* master_run.py
  The core automated script designed for complete reproducibility. It combines 
  the functionality of the entire pipeline into a single operational interface. 
  Running main() executes the full lifecycle for both Model A and Model B 
  sequentially: dynamic hardware targeting, network compilation, training loop 
  execution, cross-validation logging, weight serialization, and final 
  comparative research reports (classification metrics and confusion matrices).

================================================================================
4. TECHNICAL REQUIREMENTS & DEPENDENCIES
================================================================================
Execution requires a Python 3.8+ environment along with the following primary 
libraries:

  * torch >= 2.0.0      (Deep learning core Engine)
  * torchvision         (Image folder processing and standard transforms)
  * numpy               (Linear algebra and multi-dimensional array structures)
  * scikit-learn        (Statistical metrics collection and parsing)

The architecture features a dynamic hardware-agnostic design. At runtime, the 
scripts evaluate system resources to automatically utilize specialized hardware 
acceleration platforms, prioritizing Apple Silicon GPUs (Metal Performance Shaders 
- 'mps') or NVIDIA CUDA cores if available, before falling back to a standard 
Windows/macOS CPU backend.

================================================================================
5. PIPELINE INSTRUCTIONS & EXECUTION
================================================================================

Step 1: Environment Setup
Verify your runtime library installations. To install missing dependencies via 
terminal, execute:
$ pip install torch torchvision numpy scikit-learn

Step 2: Dataset Verification
Ensure the CCEC AI Laboratory image partitions are unpacked within your local 
working directory matching this file structure:
./data/train/
./data/val/
./data/test/
Each subfolder must contain 4 class-named directories ('glass', 'metal', 
'paper', 'plastic') hosting the respective raw image data files.

Step 3: Comprehensive Automation (Recommended)
To run the entire comparative research experiment from scratch, execute the 
centralized master controller:
$ python master_run.py

This script will sequentially:
1. Initialize training for Model A over 10 epochs.
2. Track and output validation performance profiles per epoch.
3. Save Model A weights to 'Model_A_weights.pth'.
4. Output the full evaluation metrics and confusion matrix for Model A.
5. Initialize training for Model B over 10 epochs.
6. Track and output validation performance profiles per epoch.
7. Save Model B weights to 'Model_B_weights.pth'.
8. Output the full evaluation metrics and confusion matrix for Model B.

Step 4: Granular Independent Execution (Optional)
If you prefer to interface with separate components of the pipeline individually:

- To execute standalone training and generate weight files:
  $ python train.py

- To review broad macro accuracy scores on the test set:
  $ python test_model.py

- To view a clean class-by-class accuracy breakdown:
  $ python class_accuracy.py

- To extract detailed Precision, Recall, F1-Scores, and confusion statistics:
  $ python evaluate_model.py

================================================================================
6. EXPERIMENTAL RESULTS SUMMARY
================================================================================
Below is the definitive performance profile logged during the evaluation cycle 
over the 2,130-image test set:

-----------------------------------------------------
  Class   |  Model A (2-Layer)  |  Model B (3-Layer)
-----------------------------------------------------
  Glass   |       54.70%        |       55.00%
  Metal   |       47.50%        |       45.20%
  Paper   |       64.30%        |       68.40%
  Plastic |       28.80%        |       32.90%
-----------------------------------------------------
  TOTAL   |       49.20%        |       51.22%
-----------------------------------------------------

Key Findings and Error Diagnostics:
1. Deformable Textures: Model B's +4.14% and +4.10% increases in Paper and Plastic 
   confirm that the added third convolutional layer succeeds in mapping irregular 
   geometric shapes, creases, and tears.
2. Inter-Class Overlap Bottleneck: Granular error analysis via the confusion matrix 
   indicates a strong visual similarity between low-resolution plastic and paper. 
   Model B misclassified 258 plastic items as paper due to the visual ambiguity of 
   clear plastic materials at a 150x150 pixel scale.
3. Over-parameterization: The slight reduction in Metal accuracy (-2.30%) suggests 
   that highly uniform, reflective surfaces can experience slight overfitting when 
   network depth is extended without introducing explicit contrast-based data 
   augmentation techniques.
=============================================================================
