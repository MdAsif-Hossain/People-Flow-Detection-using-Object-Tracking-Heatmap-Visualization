<div align="center">

# People Flow Detection
### Object Tracking · Directional Counting · Heatmap Visualization

![demo](demo.gif)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=github)](https://github.com/ultralytics/ultralytics)
[![Supervision](https://img.shields.io/badge/Supervision-0.28-orange?logo=roboflow)](https://supervision.roboflow.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What it does

This system processes a video feed to:

- **Detect** every person in each frame using YOLOv8
- **Track** each individual across frames with a persistent unique ID
- **Count** entries (IN) and exits (OUT) using virtual line crossings
- **Visualize** cumulative foot-traffic as a real-time heatmap

---

## Live Demo

The GIF above shows the system running on a real crowd scene.  
The full annotated output video is available at [`output_flow_tracking.mp4`](output_flow_tracking.mp4).

| Feature | Preview |
|---------|---------|
| Bounding box + ID tracking | ![demo](demo.gif) |

---

## Heatmap Output

<div align="center">

| Cumulative Foot-Traffic Heatmap | Heatmap Overlaid on Last Frame |
|---|---|
| ![heatmap](final_heatmap.png) | ![overlay](final_heatmap_overlay.png) |

*Red/yellow = high density paths · Blue = low activity zones*

</div>

---

## Crossing Event Verification

Every single IN/OUT event is logged with frame number and tracker ID and can be visually verified.

<div align="center">

![crossings](crossing_verification.png)

*Each panel = one crossing event. Green circle = IN · Red circle = OUT*

</div>

**Result on `people-walking.mp4` (341 frames, 1920×1080):**

| Metric | Value |
|--------|-------|
| Total IN | **12** |
| Total OUT | **7** |
| Unique tracked IDs | 75+ |
| Processing speed | ~13 fps (CPU) |

---

## How It Works

### 1. Detection
YOLOv8n runs on every frame, filtering detections to **class 0 (person)** only.

### 2. Tracking
ByteTrack assigns a **persistent ID** to each detected person across frames, maintaining identity through occlusions and re-entries.

### 3. Line Crossing Logic

Two virtual lines are drawn across the frame using [polygonzone.roboflow.com](https://polygonzone.roboflow.com/):

```
Frame height H = 1080 px

  y = 432  (40% of H)  ━━━━━━━━━━━━━━━━━━  Upper Line [GREEN]  →  IN
  y = 648  (60% of H)  ━━━━━━━━━━━━━━━━━━  Lower Line [RED]    →  OUT
```

For each tracked person, the **previous center-y** is stored in `track_prev_y`.  
Every frame, the direction of movement determines the count:

| Condition | Interpretation | Action |
|-----------|---------------|--------|
| `prev_y < 432` and `cy >= 432` | Moved **down** past upper line | `in_count += 1` |
| `prev_y > 648` and `cy <= 648` | Moved **up** past lower line | `out_count += 1` |

Each tracker ID is added to a `counted_in` / `counted_out` set after counting — preventing double-counts on borderline frames.

### 4. Heatmap
At every frame, a filled circle is painted at each person's center into a `float32` accumulation buffer.  
At the end, the buffer is Gaussian-blurred and mapped to a JET colormap — producing a density-weighted foot-traffic heatmap.

---

## Tech Stack

| Library | Role |
|---------|------|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Person detection |
| [Roboflow Supervision](https://supervision.roboflow.com/) | ByteTrack, annotators, video I/O |
| [OpenCV](https://opencv.org/) | Frame processing, drawing, heatmap |
| [NumPy](https://numpy.org/) | Accumulation buffer math |
| [Matplotlib](https://matplotlib.org/) | Final visualisation |
| [FFmpeg](https://ffmpeg.org/) | H.264 re-encoding for inline Colab playback |

---

## Project Structure

```
├── people_flow.py                          # Standalone Python script (CLI)
├── People Flow Detection ... .ipynb        # Colab/Jupyter notebook
├── demo.gif                                # Live detection demo
├── output_flow_tracking.mp4               # Full annotated output video
├── final_heatmap.png                       # Cumulative heatmap
├── final_heatmap_overlay.png              # Heatmap blended on last frame
├── crossing_verification.png              # Visual proof of every crossing event
├── Output.png                              # Summary figure
└── README.md
```

---

## Quickstart

```bash
# Install dependencies
pip install ultralytics supervision opencv-python numpy tqdm matplotlib

# Run on the default sample video (auto-downloaded)
python people_flow.py

# Run on your own video
python people_flow.py --video my_video.mp4 --output result.mp4 --heatmap heat.png
```

**Or open the notebook in Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MdAsif-Hossain/People-Flow-Detection-using-Object-Tracking-Heatmap-Visualization/blob/main/People%20Flow%20Detection%20using%20Object%20Tracking%20%26%20Heatmap%20Visualization.ipynb)

---

## Customisation

| Parameter | Location | Effect |
|-----------|----------|--------|
| `UPPER_LINE_RATIO` | `people_flow.py` line 37 | Move the IN line (0.0–1.0 of height) |
| `LOWER_LINE_RATIO` | `people_flow.py` line 38 | Move the OUT line |
| `HEATMAP_RADIUS` | `people_flow.py` line 41 | Footprint size per detection |
| `--video` | CLI flag | Use any input video |
| Model size | `YOLO("yolov8n.pt")` | Swap to `yolov8s/m/l` for higher accuracy |

---

<div align="center">
Made with Python · YOLOv8 · ByteTrack · OpenCV
</div>
