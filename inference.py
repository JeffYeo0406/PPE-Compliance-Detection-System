import os
import time
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


CLASS_NAMES = [
    'Fall-Detected',
    'Gloves',
    'Goggles',
    'Hardhat',
    'Ladder',
    'Mask',
    'NO-Gloves',
    'NO-Goggles',
    'NO-Hardhat',
    'NO-Mask',
    'NO-Safety Vest',
    'Person',
    'Safety Cone',
    'Safety Vest',
]

CRITICAL_CLASSES = {'Hardhat', 'Safety Vest', 'NO-Hardhat', 'NO-Safety Vest'}
WARNING_CLASSES = {'Gloves', 'Goggles', 'Mask', 'NO-Gloves', 'NO-Goggles', 'NO-Mask'}
EMERGENCY_CLASSES = {'Fall-Detected'}
NEUTRAL_CLASSES = {'Person', 'Ladder', 'Safety Cone'}

CLASS_COLORS_RGB = {
    'Fall-Detected':    (220, 50, 50),    # Red — emergency
    'Gloves':           (186, 117, 29),   # Amber
    'Goggles':          (123, 178, 224),  # Light blue
    'Hardhat':          (29, 158, 117),   # Green — compliant
    'Ladder':           (160, 160, 160),  # Grey
    'Mask':             (88, 147, 196),   # Blue
    'NO-Gloves':        (200, 120, 40),   # Dark amber — warning violation
    'NO-Goggles':       (180, 80, 80),    # Dark red — warning violation
    'NO-Hardhat':       (220, 50, 50),    # Red — critical violation
    'NO-Mask':          (200, 80, 80),    # Red — warning violation
    'NO-Safety Vest':   (220, 50, 50),    # Red — critical violation
    'Person':           (165, 117, 29),   # Brown/amber
    'Safety Cone':      (255, 165, 0),    # Orange
    'Safety Vest':      (24, 95, 165),    # Blue — compliant
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check root first (HF Spaces flat layout), then models/ (local dev layout)
_alt_int8 = os.path.join(BASE_DIR, 'best_int8.onnx')
_alt_fp32 = os.path.join(BASE_DIR, 'best.onnx')
_models_int8 = os.path.join(BASE_DIR, 'models', 'best_int8.onnx')
_models_fp32 = os.path.join(BASE_DIR, 'models', 'best.onnx')
MODEL_INT8 = _alt_int8 if os.path.exists(_alt_int8) else _models_int8
MODEL_FP32 = _alt_fp32 if os.path.exists(_alt_fp32) else _models_fp32
INPUT_SIZE = 320


class PPEInferenceEngine:
    def __init__(self, model_path: Optional[str] = None, input_size: int = INPUT_SIZE):
        self.input_size = input_size
        self.session, self.input_name, self.model_label = self._load_session(model_path)

    def _preferred_providers(self) -> list[str]:
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _load_session(self, model_path: Optional[str]):
        if not ONNX_AVAILABLE:
            return None, None, 'DEMO'

        candidate_paths = []
        if model_path:
            candidate_paths.append(model_path)
        candidate_paths.extend([MODEL_FP32, MODEL_INT8])  # FP32 first (INT8 has known quantization issues)

        for path in candidate_paths:
            if os.path.exists(path):
                session = ort.InferenceSession(path, providers=self._preferred_providers())
                input_name = session.get_inputs()[0].name
                return session, input_name, os.path.basename(path)

        return None, None, 'DEMO'

    def preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox resize preserving aspect ratio → (tensor, scale, pad_x, pad_y)."""
        h, w = image_rgb.shape[:2]
        scale = self.input_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h))

        pad_h = self.input_size - new_h
        pad_w = self.input_size - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        return tensor[np.newaxis, :], scale, pad_left, pad_top

    def _demo_detections(self, image_rgb: np.ndarray) -> list[dict]:
        height, width = image_rgb.shape[:2]
        return [
            {'class_name': 'Person', 'confidence': 0.95, 'bbox': (int(width * 0.18), int(height * 0.08), int(width * 0.82), int(height * 0.92))},
            {'class_name': 'Hardhat', 'confidence': 0.93, 'bbox': (int(width * 0.36), int(height * 0.06), int(width * 0.54), int(height * 0.20))},
            {'class_name': 'Safety Vest', 'confidence': 0.89, 'bbox': (int(width * 0.34), int(height * 0.28), int(width * 0.58), int(height * 0.72))},
            {'class_name': 'Gloves', 'confidence': 0.81, 'bbox': (int(width * 0.38), int(height * 0.78), int(width * 0.58), int(height * 0.92))},
        ]

    def _normalize_output(self, raw_output: np.ndarray) -> np.ndarray:
        preds = raw_output[0] if raw_output.ndim == 3 else raw_output
        preds = np.squeeze(preds)
        if preds.ndim != 2:
            raise ValueError(f'Unexpected model output shape: {raw_output.shape}')

        if preds.shape[0] < preds.shape[1]:
            preds = preds.T
        return preds

    def postprocess(
        self,
        raw_output: np.ndarray,
        orig_w: int,
        orig_h: int,
        conf_thresh: float,
        scale: float = 1.0,
        pad_x: int = 0,
        pad_y: int = 0,
        iou_thresh: float = 0.45,
    ) -> list[dict]:
        preds = self._normalize_output(raw_output)

        boxes, scores, class_ids = [], [], []

        for row in preds:
            if row.shape[0] < 5:
                continue

            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if confidence < conf_thresh:
                continue

            cx, cy, bw, bh = row[:4]
            # Undo letterbox: remove padding, then scale to original size
            cx = (cx - pad_x) / scale
            cy = (cy - pad_y) / scale
            bw /= scale
            bh /= scale

            x1 = max(0, int(cx - bw / 2))
            y1 = max(0, int(cy - bh / 2))
            x2 = min(orig_w, int(cx + bw / 2))
            y2 = min(orig_h, int(cy + bh / 2))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(confidence)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                class_name = CLASS_NAMES[class_ids[i]] if class_ids[i] < len(CLASS_NAMES) else 'unknown'
                detections.append({
                    'class_name': class_name,
                    'confidence': round(scores[i], 3),
                    'bbox': (x, y, x + bw, y + bh),
                })

        return detections

    def run(self, image_rgb: np.ndarray, conf_thresh: float) -> tuple[list[dict], float]:
        if self.session is None:
            time.sleep(0.05)
            return self._demo_detections(image_rgb), 18.0

        tensor, scale, pad_x, pad_y = self.preprocess(image_rgb)
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - start) * 1000

        height, width = image_rgb.shape[:2]
        detections = self.postprocess(outputs[0], width, height, conf_thresh, scale, pad_x, pad_y)
        return detections, round(elapsed_ms, 1)
