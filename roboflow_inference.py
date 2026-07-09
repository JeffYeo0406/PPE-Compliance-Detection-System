"""
roboflow_inference.py
---------------------
Roboflow serverless API wrapper — zero pip deps (stdlib urllib only).
Returns same format as inference.PPEInferenceEngine.run().

Set ROBOFLOW_API_KEY env var before use.
"""

import os
import time
import base64
import json
from urllib import request as urlrequest

import cv2
import numpy as np

CLASS_ID_MAP: dict[int, str] = {
    0:  "Fall-Detected",  1: "Gloves",      2: "Goggles",
    3:  "Hardhat",        4: "Ladder",      5: "Mask",
    6:  "NO-Gloves",      7: "NO-Goggles",  8: "NO-Hardhat",
    9:  "NO-Mask",       10: "NO-Safety Vest", 11: "Person",
    12: "Safety Cone",   13: "Safety Vest",
}

MODEL_ID = "personal-protective-equipment-combined-model/8"
API_URL  = "https://detect.roboflow.com"


def run_inference(
    image_rgb: np.ndarray,
    confidence_threshold: float = 0.35,
) -> tuple[list[dict], float]:
    """Run PPE detection via Roboflow serverless API.

    Returns (detections, elapsed_ms) — compatible with inference.py format.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY not set. "
            "HF Spaces: Settings → Variables and secrets → New secret."
        )

    endpoint = f"{API_URL}/{MODEL_ID}?api_key={api_key}"

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode image")
    b64 = base64.b64encode(buf).decode('utf-8')

    start = time.perf_counter()
    req = urlrequest.Request(
        endpoint,
        data=b64.encode('utf-8'),
        headers={
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'PPE-Detection/2.0',
        },
    )
    with urlrequest.urlopen(req, timeout=20) as resp:
        raw = json.loads(resp.read().decode('utf-8'))

    elapsed_ms = (time.perf_counter() - start) * 1000
    height, width = image_rgb.shape[:2]

    detections: list[dict] = []
    for p in raw.get("predictions", []):
        conf = float(p.get("confidence", 0))
        if conf < confidence_threshold:
            continue

        cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(width, int(cx + bw / 2))
        y2 = min(height, int(cy + bh / 2))

        if x2 <= x1 or y2 <= y1:
            continue

        class_name = p.get("class", "") or CLASS_ID_MAP.get(
            int(p.get("class_id", -1)), "unknown")
        class_name = class_name.replace("_", " ")

        detections.append({
            "class_name": class_name,
            "confidence": round(conf, 3),
            "bbox": (x1, y1, x2, y2),
        })

    return detections, round(elapsed_ms, 1)
