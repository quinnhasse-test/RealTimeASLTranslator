# Real-time ASL Translator

Classifies American Sign Language hand gestures from a webcam feed and outputs the corresponding letter in real time. Built with OpenCV for capture and TensorFlow for inference.

## Architecture

```
Webcam frame
    └── OpenCV preprocessing (crop ROI, resize to 64×64, normalize, grayscale)
            └── TensorFlow CNN classifier
                    └── Softmax over 26 letter classes
                            └── Top-1 prediction rendered as overlay
```

**Pipeline:**
1. OpenCV captures frames at 30 fps from the default webcam device.
2. The region of interest (hand bounding box) is cropped, resized to the model input shape (64×64), converted to grayscale, and pixel-normalized to [0, 1].
3. A CNN trained on labeled ASL gesture images outputs a class probability vector over 26 letters (A–Z).
4. The top-1 prediction and its confidence score are rendered as a text overlay on the live frame.

## Model

**Architecture:** Three convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool) followed by two dense layers with dropout.

| Layer block | Filters | Kernel | Output shape |
|-------------|---------|--------|--------------|
| Conv1 | 32 | 3×3 | 62×62×32 |
| Conv2 | 64 | 3×3 | 29×29×64 |
| Conv3 | 128 | 3×3 | 13×13×128 |
| Dense1 | 512 units | — | 512 |
| Dense2 (output) | 26 units | — | 26 |

**Training:**
- Dataset: labeled ASL hand gesture images (A–Z static letters)
- Optimizer: Adam (lr=1e-3, decay to 1e-5)
- Loss: categorical cross-entropy
- Augmentation: random horizontal flip, rotation ±15°, brightness jitter

## Usage

### Prerequisites

- Python 3.9+
- A webcam connected to your machine

### Install

```bash
git clone https://github.com/quinnhasse-test/RealTimeASLTranslator.git
cd RealTimeASLTranslator
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Press `q` to exit the live feed window.

### Training

```bash
python train.py --data_dir data/asl_alphabet_train --epochs 30
```

Saves the trained model to `model/asl_cnn.keras`. Training logs accuracy and loss per epoch to stdout.

## Project structure

```
RealTimeASLTranslator/
├── main.py           # Webcam loop + inference + overlay rendering
├── train.py          # Model training entry point
├── model/
│   ├── model.py      # CNN architecture definition
│   └── asl_cnn.keras # Saved model weights
├── data/
│   └── asl_alphabet_train/  # Training images (A–Z subdirs)
├── tests/
│   ├── test_preprocessing.py  # Resize, normalize, ROI crop
│   └── test_labels.py         # Class label mapping A–Z
└── requirements.txt
```

## Testing

```bash
pytest tests/
```

| File | What it covers |
|------|----------------|
| `test_preprocessing.py` | Resize to 64×64, normalization to [0, 1], grayscale conversion |
| `test_labels.py` | Class index → letter mapping, 26-class coverage |

## Dependencies

See `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

Key packages: `tensorflow>=2.13.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`.
