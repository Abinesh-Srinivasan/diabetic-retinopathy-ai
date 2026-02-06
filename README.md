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

## Final Performance (Test Accuracy)

| Model | Test Accuracy |
|------|---------------|
| CNN | 61% |
| Vision Transformer (ViT) | 83% |
| Hybrid CNN–ViT | **85%** |

---

## Explainability
To improve transparency and clinical trust:
- **Grad-CAM** was applied to CNN and Hybrid models to highlight lesion-level regions influencing predictions.
- **Attention Rollout** was applied to the Vision Transformer to visualize global regions contributing to classification decisions.

These techniques ensure that the models are not only accurate but also interpretable.

---

## How to Run This Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Abinesh-Srinivasan/diabetic-retinopathy-ai.git
cd diabetic-retinopathy-ai
pip install -r requirements.txt
```

### 1️⃣ Model Training
```bash
python train/train_cnn.py
python train/train_vit.py
python train/train_hybrid.py
```

### 1️⃣ Evaluating Model
```bash
python eval/eval_cnn.py
python eval/eval_vit.py
python eval/eval_hybrid.py
```

### 1️⃣ Testing Model
```bash
python test/test_cnn.py
python test/test_vit.py
python test/test_hybrid.py
