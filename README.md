# Freestyle Wrestling Action Recognition (FSW-Net)

Official repository for the paper:

**A CNN–Bi-LSTM Pipeline and Open FSW Dataset for Freestyle Wrestling Action Recognition**

Published in **Scientific Reports**, 2026.
DOI: https://doi.org/10.1038/s41598-026-44782-0

---

## Overview

Human action recognition in close-contact sports is challenging due to rapid body movements, severe occlusion, inter-athlete interactions, camera-view variations, and the visual similarity between different techniques. This repository provides the official implementation and dataset resources for freestyle wrestling action recognition using a foreground-aware deep learning pipeline.

The proposed framework combines:

1. **DeepLabV3+** for athlete foreground segmentation,
2. **EfficientNet-B7** for spatial feature extraction,
3. **Bidirectional LSTM (Bi-LSTM)** for temporal modeling,
4. A final classification layer for recognizing freestyle wrestling techniques.

The method is evaluated on the **Open FSW Dataset**, which contains seven freestyle wrestling action classes.

---

## Paper

**Title:** A CNN–Bi-LSTM pipeline and open FSW dataset for freestyle wrestling action recognition
**Authors:** Milad Rostamian, Ali Mottaghi, Mohsen Soryani
**Journal:** Scientific Reports
**Year:** 2026
**Volume:** 16
**Article number:** 14632
**DOI:** https://doi.org/10.1038/s41598-026-44782-0

---

## Architecture

The proposed pipeline consists of the following stages:

### 1. Foreground Segmentation

DeepLabV3+ is used to extract athlete foreground regions from video frames. This step helps reduce background noise and focuses the model on the wrestlers and their movements.

### 2. Spatial Feature Extraction

A fine-tuned EfficientNet-B7 network is used to extract discriminative spatial features from the segmented frames.

### 3. Temporal Modeling

The frame-level features are passed to a Bidirectional LSTM to model temporal dependencies across video sequences.

### 4. Action Classification

The final temporal representation is fed into a dense classification layer to predict one of the seven freestyle wrestling action classes.

---

## Open FSW Dataset

The Open FSW Dataset contains freestyle wrestling video clips categorized into seven action classes:

1. **Hip headlock throw**
2. **Single leg tackle**
3. **Gut wrench / rolling side sweep**
4. **Arm spin**
5. **Side sweep and over-under**
6. **Ankle lace**
7. **From head and arm**

The dataset is designed to support research on action recognition in close-contact combat sports, where occlusion, body overlap, and rapid motion make classification difficult.

---

## Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Organize the dataset using the following structure:

```text
data/
  ├── 01-Hip_headlock_throw/
  │    ├── video_001.mp4
  │    ├── video_002.mp4
  │    └── ...
  ├── 02-Single_leg_tackle/
  │    ├── video_001.mp4
  │    └── ...
  ├── 03-Gut_wrench_or_rolling_side_sweep/
  │    └── ...
  ├── 04-Arm_spin/
  │    └── ...
  ├── 05-Side_sweep_and_over_under/
  │    └── ...
  ├── 06-Ankle_lace/
  │    └── ...
  └── 07-From_head_and_arm/
       └── ...
```

---

## Training

Run the training script:

```bash
python train.py
```

Depending on your local configuration, you may need to adjust dataset paths, batch size, number of frames, learning rate, and other training parameters inside the training script or configuration file.

---

## Evaluation

The proposed foreground-aware CNN–Bi-LSTM pipeline achieves **82.9% accuracy** on the Open FSW Dataset using **6-fold cross-validation**.

---

## Project Structure

```text
Freestyle-Wrestling-Dataset/
│
├── data/                   # Dataset directory
├── models/                 # Model definitions
│   └── architecture.py     # CNN + Bi-LSTM architecture
├── utils/                  # Utility functions
│   └── segmentation.py     # DeepLabV3+ segmentation utilities
├── dataset.py              # Dataset loading and preprocessing
├── train.py                # Training script
├── requirements.txt        # Python dependencies
├── LICENSE                 # License file
└── README.md               # Project documentation
```

---

## Citation

If you use this code, dataset, or any part of this repository in your research, please cite the following paper:

```bibtex
@article{rostamian2026cnn,
  title={A CNN--Bi-LSTM pipeline and open FSW dataset for freestyle wrestling action recognition},
  author={Rostamian, Milad and Mottaghi, Ali and Soryani, Mohsen},
  journal={Scientific Reports},
  volume={16},
  pages={14632},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s41598-026-44782-0},
  url={https://doi.org/10.1038/s41598-026-44782-0}
}
```

---

## Featured

This work has also been featured on ClearSkyScience:

https://www.clearskyscience.com/en/10.1038/s41598-026-44782-0/

---

## License

This project is licensed under the MIT License. Please see the `LICENSE` file for details.

---

## Contact

For questions, comments, or research collaboration, please contact:

**Milad Rostamian**
Email: [milad.chessmaster@gmail.com](mailto:milad.chessmaster@gmail.com)

---

## Note

This repository is provided for academic and research use. If you use the dataset, code, or results, please cite the original paper.
