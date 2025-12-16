import math
import tempfile
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
import requests


def _save_uploaded_file_to_temp(uploaded_file) -> str:
    """Save Streamlit UploadedFile to a temp file path and return path."""
    suffix = ".mp4"
    if uploaded_file.name.lower().endswith(".mov"):
        suffix = ".mov"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def download_video_from_url(url: str) -> str:
    """
    Download a *direct* video URL (mp4/mov) to a temporary file.
    Returns the local file path.
    """
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()

    suffix = ".mp4"
    if ".mov" in url.lower():
        suffix = ".mov"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                tmp.write(chunk)
        return tmp.name


def load_first_frame(uploaded_file=None, video_url: Optional[str] = None):
    """Load the first frame of a provided clip (UploadedFile or direct URL).

    Returns: (frame_rgb, width, height) or (None, None, None) on failure.
    """
    if uploaded_file is not None:
        path = _save_uploaded_file_to_temp(uploaded_file)
    elif video_url:
        path = download_video_from_url(video_url)
    else:
        return None, None, None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, None, None

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, w, h


def _estimate_motion_roi(cap: cv2.VideoCapture, fps: float, seconds: float = 3.0) -> Optional[Dict[str, float]]:
    """Estimate an ROI around the most-moving region using simple frame differencing.

    Returns ROI in percent coordinates: {x0,y0,x1,y1} or None if insufficient motion.
    NOTE: This is heuristic and works best on broadcast angles.
    """
    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    ok, prev = cap.read()
    if not ok or prev is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
        return None

    H, W, _ = prev.shape
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion = np.zeros((H, W), dtype=np.float32)

    frames_to_scan = int(max(10, min(5 * fps, seconds * fps)))
    for _ in range(frames_to_scan):
        ok, frm = cap.read()
        if not ok or frm is None:
            break
        g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g, prev_g)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        motion += diff.astype(np.float32)
        prev_g = g

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)

    if float(np.max(motion)) <= 0:
        return None

    thr = np.percentile(motion, 97)
    mask = (motion >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    pad_x = int(0.12 * (x1 - x0 + 1) + 30)
    pad_y = int(0.18 * (y1 - y0 + 1) + 40)

    x0 = max(0, x0 - pad_x)
    x1 = min(W - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(H - 1, y1 + pad_y)

    return {
        "x0": (x0 / W) * 100.0,
        "y0": (y0 / H) * 100.0,
        "x1": (x1 / W) * 100.0,
        "y1": (y1 / H) * 100.0,
    }


def compute_burst_signal(
    uploaded_file=None,
    video_url: Optional[str] = None,
    max_seconds: int = 12,
    preview_frames: int = 0,
    roi_pct: Optional[Dict[str, float]] = None,
    auto_roi: bool = False,
) -> Dict:
    """
    Compute a simple, interpretable 'burst' proxy from a short clip.

    Approach (v1):
    - Use MediaPipe Pose to estimate hip center (midpoint of left/right hip)
    - Track hip center across frames
    - Compute per-frame speed and acceleration
    - Burst score = max acceleration in first ~2 seconds (normalized 0–100)
    """
    if uploaded_file is not None:
        path = _save_uploaded_file_to_temp(uploaded_file)
    elif video_url:
        path = download_video_from_url(video_url)
    else:
        return {"error": "No video provided."}

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fps)

    auto_roi_used = None
    if auto_roi and roi_pct is None:
        auto_roi_used = _estimate_motion_roi(cap, fps=fps, seconds=3.0)
        if auto_roi_used is not None:
            roi_pct = auto_roi_used

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    points: List[Tuple[float, float]] = []
    preview_images = []
    frame_count = 0

    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # Optional ROI crop (percent-based) to force tracking to the intended player
        x_off, y_off = 0, 0
        frame_for_pose = frame
        if roi_pct is not None:
            H, W, _ = frame.shape
            x0 = int(np.clip(roi_pct.get("x0", 0.0), 0.0, 100.0) / 100.0 * W)
            y0 = int(np.clip(roi_pct.get("y0", 0.0), 0.0, 100.0) / 100.0 * H)
            x1 = int(np.clip(roi_pct.get("x1", 100.0), 0.0, 100.0) / 100.0 * W)
            y1 = int(np.clip(roi_pct.get("y1", 100.0), 0.0, 100.0) / 100.0 * H)

            # Ensure valid box
            if x1 <= x0:
                x1 = min(W, x0 + 1)
            if y1 <= y0:
                y1 = min(H, y0 + 1)

            x_off, y_off = x0, y0
            frame_for_pose = frame[y0:y1, x0:x1]

        rgb = cv2.cvtColor(frame_for_pose, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            points.append((math.nan, math.nan))
            continue

        h, w, _ = frame_for_pose.shape
        lm = res.pose_landmarks.landmark
        left_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]

        x = ((left_hip.x + right_hip.x) / 2.0) * w
        y = ((left_hip.y + right_hip.y) / 2.0) * h
        x = x + x_off
        y = y + y_off
        points.append((x, y))

        # Capture a few preview frames so the user can see what was tracked
        if preview_frames and len(preview_images) < preview_frames:
            frame_vis = frame.copy()

            # Draw ROI box if enabled
            if roi_pct is not None:
                H, W, _ = frame.shape
                x0 = int(np.clip(roi_pct.get("x0", 0.0), 0.0, 100.0) / 100.0 * W)
                y0 = int(np.clip(roi_pct.get("y0", 0.0), 0.0, 100.0) / 100.0 * H)
                x1 = int(np.clip(roi_pct.get("x1", 100.0), 0.0, 100.0) / 100.0 * W)
                y1 = int(np.clip(roi_pct.get("y1", 100.0), 0.0, 100.0) / 100.0 * H)
                cv2.rectangle(frame_vis, (x0, y0), (x1, y1), (255, 0, 255), 2)
                cv2.putText(frame_vis, "ROI (AUTO)" if auto_roi_used is not None else "ROI", (x0 + 6, max(18, y0 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            # Draw pose landmarks on the *full* frame by temporarily shifting them
            # (We draw on a cropped frame first, then paste it back if ROI is enabled)
            if roi_pct is None:
                mp_drawing.draw_landmarks(frame_vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                # Draw on cropped, then paste
                cropped_vis = frame_for_pose.copy()
                mp_drawing.draw_landmarks(cropped_vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                frame_vis[y_off:y_off + cropped_vis.shape[0], x_off:x_off + cropped_vis.shape[1]] = cropped_vis

            # Pose bounding box from landmarks (approx)
            lm = res.pose_landmarks.landmark
            xs = [int(l.x * w) + x_off for l in lm]
            ys = [int(l.y * h) + y_off for l in lm]
            bx0, by0 = max(0, min(xs)), max(0, min(ys))
            bx1, by1 = min(frame.shape[1] - 1, max(xs)), min(frame.shape[0] - 1, max(ys))
            cv2.rectangle(frame_vis, (bx0, by0), (bx1, by1), (0, 255, 255), 2)
            cv2.putText(frame_vis, "POSE", (bx0 + 6, max(18, by0 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Hip center marker
            cv2.circle(frame_vis, (int(x), int(y)), 6, (0, 255, 0), -1)

            preview_images.append(cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB))

    cap.release()
    pose.close()

    if len(points) < int(2 * fps):
        return {"error": "Clip too short. Use a 5–15 second clip with the player visible."}

    # Forward-fill missing points
    filled = []
    last = None
    for x, y in points:
        if math.isnan(x) or math.isnan(y):
            if last is not None:
                filled.append(last)
        else:
            last = (x, y)
            filled.append((x, y))

    if len(filled) < int(2 * fps):
        return {"error": "Not enough usable frames. Try a clearer clip (player larger in frame)."}

    # Speed (pixels/frame)
    speeds = [0.0]
    for i in range(1, len(filled)):
        x1, y1 = filled[i - 1]
        x2, y2 = filled[i]
        speeds.append(math.hypot(x2 - x1, y2 - y1))

    speeds_np = np.array(speeds, dtype=np.float32)

    # Smooth to reduce jitter
    if len(speeds_np) >= 9:
        kernel = np.ones(9) / 9.0
        speeds_np = np.convolve(speeds_np, kernel, mode="same")

    # Acceleration (delta speed)
    accels_np = np.diff(speeds_np, prepend=speeds_np[0])

    # Burst window ~ first 2 seconds
    burst_frames = int(min(len(accels_np), 2.0 * fps))
    max_accel = float(np.nanmax(accels_np[:burst_frames]))

    # Normalize to 0–100 using clip percentile scale
    clip_p95 = float(np.percentile(np.abs(accels_np), 95))
    denom = clip_p95 if clip_p95 > 1e-6 else 1.0
    burst_score = int(np.clip((max_accel / denom) * 70 + 30, 0, 100))

    return {
        "burst_score": burst_score,
        "max_accel": max_accel,
        "speeds": speeds_np.tolist(),
        "accels": accels_np.tolist(),
        "fps": float(fps),
        "preview_frames": preview_images,
        "roi_pct": roi_pct,
        "roi_auto": bool(auto_roi_used is not None),
        "notes": "Burst is a motion proxy derived from hip-center acceleration in the first ~2s of the clip."
    }