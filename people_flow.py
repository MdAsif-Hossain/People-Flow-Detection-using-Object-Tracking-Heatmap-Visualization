"""
People Flow Detection using Object Tracking & Heatmap Visualization
-------------------------------------------------------------------
Detects, tracks, and counts people crossing two virtual lines in a video,
then generates a cumulative motion heatmap.

Usage:
    python people_flow.py [--video VIDEO_PATH] [--output OUTPUT_PATH]

Detection:  YOLOv8n (COCO class 0 = person)
Tracker:    ByteTrack via supervision
Heatmap:    Gaussian-blurred center-point accumulation, JET colormap
"""

import argparse
import os
import urllib.request
import warnings
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)
import supervision as sv  # noqa: E402  (after warning filter)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/people-walking.mp4"
DEFAULT_VIDEO_PATH = "people-walking.mp4"
DEFAULT_OUTPUT_VIDEO = "output_flow_tracking.mp4"
DEFAULT_OUTPUT_HEATMAP = "final_heatmap.png"

# Line positions as a fraction of frame height (from polygonzone.roboflow.com)
UPPER_LINE_RATIO = 0.40   # IN  line  – people crossing downward are counted IN
LOWER_LINE_RATIO = 0.60   # OUT line  – people crossing upward are counted OUT

# Heatmap circle radius painted at each detection center
HEATMAP_RADIUS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_video(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"Video already exists: {dest}")
        return
    print(f"Downloading video from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print("Download complete.")


def build_heatmap_overlay(
    heatmap_accum: np.ndarray,
    background: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blend a JET-mapped heatmap onto a background frame."""
    blurred = cv2.GaussianBlur(heatmap_accum, (99, 99), 0)
    normed = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    return cv2.addWeighted(background, 1 - alpha, colored, alpha, 0)


def draw_counter(frame: np.ndarray, in_count: int, out_count: int) -> None:
    cv2.rectangle(frame, (10, 10), (270, 125), (0, 0, 0), -1)
    cv2.putText(frame, f"IN : {in_count}",  (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT: {out_count}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)


def draw_lines(
    frame: np.ndarray,
    upper_y: int,
    lower_y: int,
    width: int,
) -> None:
    cv2.line(frame, (0, upper_y), (width, upper_y), (0, 255, 0), 3)
    cv2.putText(frame, "Upper Line [IN  crossing down]",
                (10, upper_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.line(frame, (0, lower_y), (width, lower_y), (0, 0, 255), 3)
    cv2.putText(frame, "Lower Line [OUT crossing up]",
                (10, lower_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run(
    video_path: str,
    output_video: str,
    output_heatmap: str,
) -> None:
    # ---- Load model & video ------------------------------------------------
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    video_info = sv.VideoInfo.from_video_path(video_path)
    W, H = video_info.width, video_info.height
    upper_y = int(H * UPPER_LINE_RATIO)
    lower_y = int(H * LOWER_LINE_RATIO)

    print(f"Resolution : {W} x {H}  |  Frames: {video_info.total_frames}")
    print(f"Upper (IN) : y = {upper_y}  ({UPPER_LINE_RATIO:.0%} of height)")
    print(f"Lower (OUT): y = {lower_y}  ({LOWER_LINE_RATIO:.0%} of height)")

    # ---- Tracker & annotators ----------------------------------------------
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.WHITE,
        text_color=sv.Color.BLACK,
        text_scale=0.7,
        text_thickness=2,
        text_padding=8,
        border_radius=4,
    )

    # ---- State -------------------------------------------------------------
    # Maps tracker_id -> last recorded center-y for crossing detection
    track_prev_y = {}
    # Set of IDs already counted to avoid double-counting on noisy frames
    counted_in = set()
    counted_out = set()

    in_count = 0
    out_count = 0
    heatmap_accum = np.zeros((H, W), dtype=np.float32)

    # Capture last frame for heatmap overlay background
    last_frame: Optional[np.ndarray] = None

    # ---- Processing loop ---------------------------------------------------
    print("\nProcessing frames...")
    with sv.VideoSink(output_video, video_info) as sink:
        frames = sv.get_video_frames_generator(video_path)
        for frame in tqdm(frames, total=video_info.total_frames):
            # Detection (class 0 = person in COCO)
            results = model(frame, classes=[0], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            labels = []
            for i in range(len(detections)):
                xyxy = detections.xyxy[i]
                tid = int(detections.tracker_id[i])

                cx = int((xyxy[0] + xyxy[2]) / 2)
                cy = int((xyxy[1] + xyxy[3]) / 2)

                labels.append(f"#{tid}")

                # Accumulate presence for heatmap
                cv2.circle(heatmap_accum, (cx, cy), HEATMAP_RADIUS, 1, thickness=-1)

                # Crossing logic
                if tid in track_prev_y:
                    prev_y = track_prev_y[tid]

                    # Moving DOWN (y increases) across the upper line -> IN
                    if prev_y < upper_y <= cy and tid not in counted_in:
                        in_count += 1
                        counted_in.add(tid)

                    # Moving UP (y decreases) across the lower line -> OUT
                    if prev_y > lower_y >= cy and tid not in counted_out:
                        out_count += 1
                        counted_out.add(tid)

                track_prev_y[tid] = cy

            # Annotate frame
            frame = box_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            draw_lines(frame, upper_y, lower_y, W)
            draw_counter(frame, in_count, out_count)

            sink.write_frame(frame)
            last_frame = frame.copy()

    # ---- Generate heatmap --------------------------------------------------
    print("\nGenerating heatmap...")
    blurred = cv2.GaussianBlur(heatmap_accum, (99, 99), 0)
    normed = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap_color = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    cv2.imwrite(output_heatmap, heatmap_color)

    # Also save a blended overlay on the last frame for context
    if last_frame is not None:
        overlay = build_heatmap_overlay(heatmap_accum, last_frame)
        overlay_path = output_heatmap.replace(".png", "_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"Overlay saved : {overlay_path}")

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 40)
    print(f"  Total IN  : {in_count}")
    print(f"  Total OUT : {out_count}")
    print(f"  Output video : {output_video}")
    print(f"  Heatmap      : {output_heatmap}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="People Flow Detection")
    p.add_argument("--video", default=DEFAULT_VIDEO_PATH,
                   help="Path to input video (downloaded if missing)")
    p.add_argument("--output", default=DEFAULT_OUTPUT_VIDEO,
                   help="Path for annotated output video")
    p.add_argument("--heatmap", default=DEFAULT_OUTPUT_HEATMAP,
                   help="Path for final heatmap image")
    p.add_argument("--url", default=DEFAULT_VIDEO_URL,
                   help="URL to download video from if not found locally")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_video(args.url, args.video)
    run(args.video, args.output, args.heatmap)
