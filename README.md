# Real-Time Vehicle & Pedestrian Detection (ADAS)

A real-time object detection system built with YOLOv8 and OpenCV that detects 
vehicles, pedestrians and road objects from video footage — simulating a core 
component of Advanced Driver Assistance Systems.

---

## Demo

![Demo](output_image.png)

---

## About

I built this to explore how real world ADAS systems work. The pipeline reads 
video frame by frame, runs each frame through YOLOv8, draws colored bounding 
boxes around detected objects and saves the annotated output.

---

## Features

- Real time detection at 29–30 FPS on CPU
- 11 object classes — people, cars, trucks, buses, motorcycles,
  bicycles, traffic lights, stop signs and more
- Color coded bounding boxes per class
- Confidence score on each detection
- Live object count per class
- FPS counter
- Saves annotated output as output.mp4

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| YOLOv8 nano | Object detection model |
| OpenCV | Video processing and drawing |
| PyTorch | Deep learning backend |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/shaikaltaaf123/adas-detection.git
cd adas-detection
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add a video file**

Place any traffic or street video in the project folder and rename it `traffic.mp4`

**5. Verify setup**
```bash
python verify_setup.py
```

**6. Run**
```bash
python detect.py
```
Press `Q` to quit.

---

## Configuration

```python
VIDEO_PATH = "traffic.mp4"   # swap in any video
CONFIDENCE = 0.4             # detection confidence threshold
CLASSES    = [0, 1, 2, ...]  # COCO class IDs to detect
```

---

## Limitations & future improvements

- Currently runs on CPU only — GPU support would significantly increase FPS
- No object tracking — detected objects do not have persistent IDs across frames
- Model not fine tuned on dashcam data — may miss some road specific scenarios
- Fixed camera angle — performance may vary on aerial or unusual angles
- Distance estimation not yet implemented

---

## Project structure
adas-detection/

├── detect.py            # main detection script
├── verify_setup.py      # verifies installation
├── requirements.txt     # dependencies
├── README.md            # project documentation
└── .gitignore           # ignores videos, model weights, venv


---

## License

MIT License





