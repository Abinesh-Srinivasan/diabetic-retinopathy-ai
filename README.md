# Diabetic Retinopathy Detection using CNN, Vision Transformer & Hybrid Models

## Overview
This project focuses on automated **Diabetic Retinopathy (DR) classification** from retinal fundus images using deep learning.  
We designed, trained, and evaluated multiple architectures to analyze how local and global visual features contribute to disease severity classification.

The models were trained and evaluated on a multi-class DR dataset with five severity levels.

---

## Project Team
- **ABINESH S**
- **DILJAZ R.S**
- **SURESH S**

---

## Models Implemented
- **CNN Baseline (EfficientNet)**  
  Learns local lesion-level features such as microaneurysms and hemorrhages.

- **Vision Transformer (ViT)**  
  Captures global retinal context and long-range spatial dependencies.

- **Hybrid CNN–ViT Model**  
  Combines CNN-based local features with ViT-based global representations using a parallel fusion architecture.

---

## How the Model Works

This project addresses diabetic retinopathy classification as a **five-class image classification problem** using retinal fundus images.  
The workflow follows a standard deep learning pipeline with progressively stronger model architectures.

---

### 1. Input Processing
- Retinal fundus images are resized to **224 × 224**
- Pixel values are normalized
- Images are fed into the network as RGB tensors

This ensures consistent input format across all models.

---

### 2. CNN Baseline Model
The CNN baseline uses an **EfficientNet-based architecture**.

**How it works:**
- Convolutional layers learn **local visual patterns**
- These include microaneurysms, hemorrhages, exudates, and vessel textures
- Global Average Pooling summarizes spatial features
- Fully connected layers produce class probabilities

**Limitation:**
- CNNs focus mainly on local features
- Long-range spatial relationships across the retina are not explicitly modeled

---

### 3. Vision Transformer (ViT)
The Vision Transformer treats the image as a sequence of patches.

**How it works:**
- The image is split into fixed-size patches
- Each patch is embedded and passed through transformer encoder blocks
- Self-attention allows the model to learn **global relationships**
- The class token aggregates information for final prediction

**Advantage:**
- Captures long-range dependencies
- Learns global retinal structure and context

---

### 4. Hybrid CNN–ViT Model
The hybrid model combines the strengths of both CNN and ViT using a **parallel architecture**.

**How it works:**
- The same input image is fed simultaneously into:
  - A CNN branch (EfficientNet) for local feature extraction
  - A ViT branch for global contextual understanding
- Features from both branches are concatenated
- Fully connected layers fuse the information
- The final layer outputs class probabilities

**Why this works better:**
- CNN captures fine-grained lesion details
- ViT captures global retinal patterns
- Feature fusion improves generalization on unseen data

---

### 5. Model Training and Evaluation
- Models are trained using categorical cross-entropy loss
- Evaluation follows a **train / validation / test split**
- Final performance is reported on the **unseen test set**
- This ensures reliable measurement of real-world performance

---

### 6. Explainability
To improve transparency and trust:
- **Grad-CAM** highlights lesion-level regions influencing CNN and Hybrid predictions
- **Attention Rollout** visualizes global regions influencing ViT predictions

These explainability techniques help verify that the models focus on clinically relevant retinal regions.

---


## How to Run This Project

### 1️⃣ Environment Setup (dr_ai)

This project was developed and tested inside a dedicated **Conda environment** named `dr_ai`.  
Cloners are strongly recommended to use the same setup to avoid dependency and compatibility issues.

---

### Step 1: Install Miniconda or Anaconda
If not already installed, download and install **Miniconda** or **Anaconda**:

- https://docs.conda.io/en/latest/miniconda.html

---

### Step 2: Create the `dr_ai` Environment
Open **Anaconda Prompt** or terminal and run:

```bash
conda create -n dr_ai python=3.11 -y
conda activate dr_ai
```
---

### Step 3: Select the Interpreter
If using VS Code:
  1. Open the project folder in VS Code
  2. Press Ctrl + Shift + P
  3. Select Python: Select Interpreter
  4. Choose: Python 3.11 (dr_ai)
  5. Open terminal and run: python --version
  6. You should see: Python 3.11.x
  7. Open new terminal and ensure the terminal shows (dr_ai) before running any script


### 2️⃣ Clone the repository
```bash
git clone https://github.com/Abinesh-Srinivasan/diabetic-retinopathy-ai.git
cd diabetic-retinopathy-ai
pip install -r requirements.txt
```

### Dataset Download and Setup

The dataset used in this project is **not included in the repository** due to its large size and licensing restrictions.

### Step 1: Download the Dataset
Download the dataset from the following link:

👉 **[ADD YOUR DATASET DOWNLOAD LINK HERE]**

---

### Step 2: Extract the Dataset
After downloading:
1. Extract the compressed file (ZIP/RAR)
2. You should obtain the raw dataset files

---

### Step 3: Place Dataset in Project Root
Move the extracted dataset contents into the **project root directory**

### 3️⃣ Model Training
```bash
python train/train_cnn.py
python train/train_vit.py
python train/train_hybrid.py
```

### 4️⃣ Evaluating Model
```bash
python eval/eval_cnn.py
python eval/eval_vit.py
python eval/eval_hybrid.py
```

### 5️⃣ Testing Model
```bash
python test/test_cnn.py
python test/test_vit.py
python test/test_hybrid.py
```

## Final Performance (Test Accuracy)

| Model | Test Accuracy |
|------|---------------|
| CNN | 61% |
| Vision Transformer (ViT) | 83% |
| Hybrid CNN–ViT | **85%** |

---