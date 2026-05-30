# Gesture Vibe Control

Real-time hand gesture recognition built with Python, OpenCV, and MediaPipe. Control media playback, swap images, scroll pages, and trigger actions — all through hand gestures captured from your webcam.

## Project Structure

| File | Description |
|------|-------------|
| `hand_tracker.py` | Core `HandDetector` class — wraps MediaPipe Hands; provides `find_hands()` and `get_position()` |
| `main.py` | Minimal prototype — detects fist vs open hand using raw MediaPipe landmarks |
| `image_vibe.py` | Maps fist / peace sign / open hand to different food images in a separate output window |
| `media_vibe.py` | Overlays a video clip (peace sign) or image (fist) in the corner of the live camera feed |
| `scroll_control.py` | Sends keyboard up/down arrow presses via pyautogui based on gesture, with 4-frame debounce |
| `area_counter.py` | Increments (index finger) / decrements (pinky) a counter by measuring finger-stretch distance from the wrist |

## Gesture Reference

| Gesture | `image_vibe` | `media_vibe` | `scroll_control` | `area_counter` |
|---------|-------------|-------------|-----------------|---------------|
| Fist (no fingers up) | Show sushi image | Overlay bowl image | — | — |
| Peace sign (index + middle) | Show bowl image | Play video overlay | Scroll up (↑) | — |
| Index only | — | — | Scroll down (↓) | — |
| Open hand (4 fingers) | Show fries image | — | — | — |
| Index finger fully stretched | — | — | — | Counter +1 |
| Pinky finger fully stretched | — | — | — | Counter −1 |

## Setup

**Requirements:** Python 3.10+, a webcam

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run any module independently:

```bash
python image_vibe.py      # Gesture → food image (sushi / bowl / fries)
python media_vibe.py      # Gesture → video or image overlay on camera feed (requires peace.mp4)
python scroll_control.py  # Gesture → scroll up/down with keyboard arrow keys
python area_counter.py    # Finger stretch → increment / decrement a counter
python main.py            # Minimal prototype — fist vs open hand detection
```

Press `q` to quit any module.

> **Note for `area_counter.py`:** The stretch thresholds (`IDX_MIN/MAX`, `PNK_MIN/MAX`) are calibrated for a specific hand size. Tune these constants at the top of the file if detection feels off.

> **Note for `media_vibe.py`:** Requires a `peace.mp4` video file in the project root to use the peace-sign overlay.

## Tech Stack

- [OpenCV](https://opencv.org/) — camera capture and frame rendering
- [MediaPipe](https://mediapipe.dev/) — hand landmark detection (21 keypoints)
- [pyautogui](https://pyautogui.readthedocs.io/) — system-level keyboard/mouse control (`scroll_control.py`)
- [NumPy](https://numpy.org/) — array operations for image compositing

---

## GitHub About Section

> Control your computer with hand gestures — real-time webcam gesture recognition using Python, OpenCV, and MediaPipe. Includes modules for image swapping, video overlays, page scrolling, and a finger-stretch counter.

**Topics to add:** `python`, `opencv`, `mediapipe`, `hand-tracking`, `gesture-recognition`, `computer-vision`, `webcam`, `real-time`
