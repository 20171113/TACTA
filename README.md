# TACTA: Temporal-Aware Cross-Task Attention for Holistic Machinability Estimation

This repository contains the official implementation of the paper submitted to *Advanced Engineering Informatics*:

> **Towards holistic machinability estimation of titanium alloy: An integrated approach with enhanced feature extraction and physics-guided deep multi-task learning**

The proposed method jointly estimates multiple machinability factors (tool wear, surface roughness, and three-axis cutting forces) from vibration signals during end-milling of Ti-6Al-4V. Three core techniques are introduced:

- **Deep multi-task learning (MTL):** simultaneous estimation of multiple machinability factors with a single predictive model.
- **Physics-guided encoder (PGE):** incorporates Waldorf's wear-force model as a soft inductive bias, ensuring physically consistent representations.
- **Temporal-aware cross-task attention (TACTA):** captures inter-task dependencies and temporal dynamics in machining processes.

---

## Repository Structure
```
TACTA/
├── data/
│   └── sample_testloader.npz       # Sample subset of the test set for quick reproduction
├── src/
│   ├── module/
│   │   ├── proposed.py             # Main proposed model integrating PGE and TACTA
│   │   ├── backbone.py             # ResNet-based 1D-CNN backbone
│   │   ├── attention.py            # Cross-task and temporal attention modules
│   │   ├── components.py           # Building blocks (projection layers, prediction heads)
│   │   └── physics_loss.py         # Physics-guided force computation (Waldorf model)
│   └── Evaluate_sample.ipynb       # Evaluation notebook on the sample data
└── README.md
```
---

## Usage

### Quick reproduction with sample data

The notebook `src/Evaluate_sample.ipynb` demonstrates how to load the proposed model and evaluate it on a small sample of the test set provided in `data/sample_testloader.npz`. The notebook prints MAE and RMSE for each machinability factor in the order reported in the paper (VB, Ra, Fx, Fy, Fz).

The trained model weights are not uploaded due to file size limits and are available upon reasonable request.
