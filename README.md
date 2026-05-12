# 😷 Face Mask Detector

Real-time face mask detection using Deep Learning and OpenCV.

**Live Demo →** [huggingface.co/spaces/Mr0Diablo/face-mask-detector](https://huggingface.co/spaces/Mr0Diablo/face-mask-detector)

---

## What it does

Detects whether a person is wearing a face mask — in real time via webcam, or from an uploaded image through the web dashboard. It locates faces using OpenCV's Haar Cascade, then classifies each face using a trained CNN.

---

## Models & Results

Three architectures were trained and compared on the same dataset (~1,500 images):

| Model | Training Accuracy | Validation Accuracy |
|---|---|---|
| MobileNetV2 (Transfer Learning) | 99.16% | 98.97% |
| VGG16 (Transfer Learning) | 95.21% | 97.94% |
| Custom CNN | 96.58% | 94.85% |

**Takeaway:** MobileNetV2 leads on both metrics while being the lightest model — making it the best choice for real-time deployment. Transfer learning outperforms training from scratch on this dataset size.

---

## Project Structure

```
face-mask-detector/
│
├── src/
│   ├── config.py              ← all constants and paths
│   ├── model.py               ← custom CNN architecture
│   ├── model_mobilenet.py     ← MobileNetV2 transfer learning
│   ├── model_vgg16.py         ← VGG16 transfer learning
│   ├── utils.py               ← preprocessing and drawing helpers
│   ├── train.py               ← training pipeline
│   ├── detector.py            ← live webcam detection
│   └── streamlit_app.py       ← web dashboard
│
├── models/                    ← saved model weights (.h5)
├── main.py                    ← single entry point
├── requirements.txt
└── hear_cascade_frontal_face_default.xml
```

---

## Dataset

1,314 training images and 194 test images across two classes: `with_mask` and `without_mask`.

Download: [Face Mask Dataset](https://data-flair.training/blogs/download-face-mask-data/)

Place as:
```
train/
    with_mask/
    without_mask/
test/
    with_mask/
    without_mask/
```

---

## How to Use

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python main.py --mode train --model custom
python main.py --mode train --model mobilenet
python main.py --mode train --model vgg16
```

### Live Detection (webcam)
```bash
python main.py --mode detect --model models/mymodel_mobilenet.h5
```
Press **q** to quit.

### Web Dashboard (local)
```bash
streamlit run src/streamlit_app.py
```

---

## Enhancements over baseline

- Modular file structure replacing a single-script approach
- BatchNormalization and Dropout added to custom CNN
- In-memory face preprocessing (no temp file disk writes)
- Histogram equalization for lighting robustness
- Confidence percentage displayed per detection
- Transfer learning with MobileNetV2 and VGG16
- Streamlit dashboard deployed on HuggingFace Spaces