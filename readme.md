# DCMGA-YOLO


## 📌 Introduction

Steel surface defect detection is critical for ensuring product quality and production safety in industrial manufacturing. However, existing methods still face several challenges in complex industrial scenarios, including the lack of dynamic allocation of multi-scale feature contributions, insufficient local responses to weak-texture and low-contrast defects, and the attenuation of edge details and directional textures during feature propagation.

To address these issues, we propose **DCMGA-YOLO**, a lightweight steel surface defect detection network based on the YOLO11n framework.

---

## ✨ Key Contributions

* **DSCA-SPPF Module**: Developed to dynamically allocate multi-scale feature contributions through scale-aware and channel-aware attention mechanisms.
* **C2DCF Module**: Designed to strengthen local responses to weak-texture and low-contrast defects by collaboratively modeling contextual information and fine-detail cues.
* **AGU-DGF Architecture**: Constructed to alleviate the attenuation of edge details and directional textures during feature propagation through asymmetric gating and dynamic guided fusion.

---

## 📊 Experimental Results

Experimental results on the **NEU-DET** dataset are shown below (DCMGA-YOLO achieves an excellent balance between accuracy and efficiency):

| Model | mAP50 (%) ↑ | Params (M) ↓ | P (%) ↑ | R (%) ↑ | FLOPs (G) ↓ | FPS ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv5n | 73.5 | 2.50 | 72.7 | 72.9 | 7.1 | 165.11 |
| YOLOv8n | 73.8 | 3.01 | 71.7 | 73.5 | 8.1 | 153.65 |
| YOLOv10n | 71.5 | 2.70 | **81.26** | 63.85 | 8.2 | 124.51 |
| YOLO11n | 78.6 | 2.58 | 70.9 | 78.6 | 6.3 | 137.39 |
| YOLOv12n | 77.8 | 2.51 | 73.2 | 72.2 | 5.8 | 178.98 |
| YOLOv13n | 77.7 | 2.45 | 74.6 | 72.6 | 6.1 | 185.99 |
| YOLO26 | 77.0 | 2.38 | 73.6 | 70.4 | 5.2 | **261.42** |
| ADMA-YOLO | 80.4 | **1.14** | 69.4 | 75.1 | **3.7** | 208.6 |
| **DCMGA-YOLO (Ours)** | **82.3** | **2.68** | **74.2** | **78.8** | **6.8** | **242.20** |

---

## Environment Setup

* Deep Learning Framework: PyTorch
* GPU: NVIDIA RTX 3090 with 24GB memory

*(Note: You can add OS, Python version, and CUDA versions here if needed, similar to your screenshot)*

## Training Parameters

* Epochs: 300
* Batch Size: 32
* Initial Learning Rate: 0.01
* Momentum: 0.937
* Weight Decay: 0.0005
* Optimizer: SGD
* Data Augmentation: Mosaic
