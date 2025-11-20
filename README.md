# Freestyle Wrestling Action Recognition (FSW-Net)

This repository contains the official PyTorch implementation of the paper:
**"A CNN-Bi-LSTM Pipeline and Open FSW Dataset for Freestyle Wrestling Action Recognition"**.

## 📌 Abstract
Human action recognition in close-contact sports is hindered by occlusion and rapid pose changes. We introduce a foreground-aware pipeline that uses **DeepLabV3+** for segmentation, **EfficientNet-B7** for feature extraction, and a **Bi-LSTM** for temporal modeling. The model is trained on the Open FSW dataset and achieves state-of-the-art results.

## 🏗 Architecture
1. **Segmentation:** DeepLabV3+ extracts athlete foregrounds.
2. **Feature Extractor:** Fine-tuned EfficientNet-B7 (Global Average Pooling).
3. **Temporal Aggregation:** Bidirectional LSTM (concatenating final states).
4. **Classification:** Dense layer (7 classes).

## 📊 Dataset
The Freestyle Wrestling Dataset contains 7 action classes:
- 01-Hip headlock throw
- 02-Single leg tackle
- 03-Gut wrench or rolling Side sweep
- 04-Arm spin
- 05-Side sweep and over-under
- 06-Ankle lace
- 07-From head and arm

## 🚀 Getting Started

### Prerequisites
Install the required packages:
```bash
pip install -r requirements.txt
```

### Data Preparation
Organize your dataset as follows:
```
data/
  ├── Class1/
  │    ├── video_01.mp4
  │    └── ...
  ├── Class2/
  │    └── ...
  ...
```

### Training
Run the training script:
```bash
python train.py
```

## 📁 Project Structure
```
FSW-Action-Recognition/
│
├── data/                   # Video dataset
├── models/
│   └── architecture.py     # CNN + LSTM model
├── utils/
│   └── segmentation.py     # DeepLabV3+ segmentation
├── dataset.py              # Dataset preprocessing
├── train.py                # Training script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## 📈 Results
The proposed method achieves **82.9% accuracy** on the FSW dataset using 6-fold cross-validation.

## 📜 Citation
If you use this code or dataset, please cite our paper:
```
@article{fsw2024,
  title={A CNN-Bi-LSTM Pipeline and Open FSW Dataset for Freestyle Wrestling Action Recognition},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 📝 License
This project is licensed under the MIT License.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact
For any questions, please contact: milad.chessmaster@gmail.com

---
**Note:** This is an academic research project. The dataset and code are provided for research purposes only.
