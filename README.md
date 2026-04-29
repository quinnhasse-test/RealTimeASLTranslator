# Real-time ASL Translator

Classifies American Sign Language hand gestures from a webcam feed and outputs the corresponding letter or word in real time. Built with OpenCV for capture and TensorFlow for inference.

## Architecture

```
Webcam frame
    └── OpenCV preprocessing (resize, normalize, grayscale)
            └── TensorFlow CNN classifier (hand landmark features)
                    └── Letter/word prediction (text overlay on frame)
```

**Pipeline:**
1. OpenCV captures frames at 30 fps from the default webcam.
2. Each frame is cropped to the region of interest, resized to the model's input shape, and normalized.
3. A CNN trained on ASL gesture images outputs a class probability vector.
4. The top prediction is rendered as an overlay on the live frame.

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

## Dependencies

See `requirements.txt`:

```
matplotlib
numpy
opencv-python
tensorflow
pandas
seaborn
```

## Testing

```bash
pytest tests/
```

Unit tests cover the preprocessing pipeline (resize, normalize) and class label mappings.
