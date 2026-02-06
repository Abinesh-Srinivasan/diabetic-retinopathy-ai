# Diabetic Retinopathy Detection using CNN, ViT & Hybrid Models

## Overview
This project implements and compares:
- CNN baseline (EfficientNet)
- Vision Transformer (ViT)
- Hybrid CNN–ViT model

for multi-class diabetic retinopathy classification using fundus images.

## Project Structure
- models/ : model architectures
- train/  : training scripts
- eval/   : validation scripts
- test/   : testing & explainability
- utils/  : preprocessing, dataset loader, Grad-CAM

## Dataset
APTOS 2019 Blindness Detection  
(Data not included due to size & licensing)

## Results (Test Accuracy)
- CNN: 61%
- ViT: 83%
- Hybrid: 85%

## Explainability
- Grad-CAM for CNN & Hybrid
- Attention Rollout for ViT
