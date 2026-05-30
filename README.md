# Gesture Vibe Control

Real-time hand gesture recognition built with Python, OpenCV, and MediaPipe. Control media playback, swap images, and trigger system actions — all through gestures captured from your webcam.

## Modules

| File | Description |
|------|-------------|
| `hand_tracker.py` | Core `HandDetector` class — wraps MediaPipe hand detection |
| `image_vibe.py` | Swap display images based on detected gesture |
| `media_vibe.py` | Overlay video or a static image on the camera feed |
| `scroll_control.py` | Scroll through content using peace sign / index finger |
| `area_counter.py` | Increment/decrement a counter by extending index or pinky |

## Gesture Reference

| Gesture | Action (varies by module) |
|---------|--------------------------|
| Fist | Trigger / show image |
| Peace sign (index + middle) | Scroll up / play video |
| Index finger only | Scroll down |
| Open hand (4 fingers) | Open state |

## Setup

**Requirements:** Python 3.10+, a webcam

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python image_vibe.py      # Gesture-triggered image swapping
python scroll_control.py  # Scroll with gestures
python area_counter.py    # Finger-based counter
python media_vibe.py      # Video overlay (requires peace.mp4 in project root)
```

Press `q` to quit any module.
