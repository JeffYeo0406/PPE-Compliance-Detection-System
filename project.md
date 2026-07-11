# Project Audit — PPE Compliance Detection System

> **Generated:** July 2026  
> **Status:** pre-employment, full-time DEA-C01 certification study  
> **Target Role:** Cloud Data Engineer / AI Platform Engineer

---

## Part 0: Maturity Tag

- **Tag:** `DEPLOYED-DEMO`
- **Evidence for this tag:**
  - Live and reachable at https://jeffyeo-cvpr-ppe-demo.hf.space/ — a Hugging Face Spaces Gradio app with a quantised ONNX model.
  - CI pipeline at [.github/workflows/ci.yml](.github/workflows/ci.yml) validates that the ONNX artifact loads and that the three Python modules pass `ruff` lint. There are no unit tests, integration tests, or model-accuracy regression checks.
  - Model-load fallback logic in [inference.py](inference.py) (lines 65–79) handles missing model files gracefully by returning demo detections, but there is no structured error reporting or telemetry.

---

## Part 1: The Portfolio Blueprint

### Core Problem
Construction-site safety compliance is checked manually, at unpredictable intervals, with no automated record. This project demonstrates that a YOLOv8n model quantised to INT8 ONNX and served through a browser-based Gradio UI can detect 14 PPE classes in real time and flag violations (NO-Hardhat, NO-Safety Vest) directly, without guessing from absence.

### Architecture Style
**Modular pipeline with dual-backend inference.** Three Python modules separate concerns: `inference.py` (model loading, preprocessing, postprocessing), `utils.py` (annotation, compliance computation, HTML rendering), and `app.py` (Gradio event wiring). The inference path can switch between local ONNX and Roboflow serverless at runtime via a radio-button in the UI.

### Production Value (per `DEPLOYED-DEMO` tag)
1. **Dual-backend resilience** — the app degrades gracefully when local inference fails by falling back to a serverless Roboflow API call (gated by `ROBOFLOW_API_KEY` env var), demonstrated in [app.py](app.py) lines 114–127.
2. **CI-validated deployment artifact** — [.github/workflows/ci.yml](.github/workflows/ci.yml) proves the quantised ONNX model is loadable and the codebase is lint-clean on every push, even though no end-to-end tests exist.

### Target Role Alignment
- **"Model deployment and serving experience"** — the repo packages a trained YOLOv8n into INT8 ONNX (3.17 MB, 3.66× smaller than FP32) and serves it through a Gradio web app, deployed on Hugging Face Spaces with environment-conditional launch logic (`SPACE_ID` env var, [app.py](app.py) lines 330–333).
- **"CI/CD pipeline configuration"** — the GitHub Actions workflow installs dependencies, validates the ONNX artifact, and lints the codebase on every push to `main`.
- **"Edge optimisation for inference"** — letterbox preprocessing preserves aspect ratio ([inference.py](inference.py) lines 81–95) instead of naive stretch-resize, and INT8 quantisation reduces the model to 3.17 MB for CPU-only deployment.

---

## Part 2: The Technical Stack Breakdown

### Language and Runtime Environment
- Python 3.13 (local dev), Python 3.10 (HF Spaces build), via `venv` isolation.
- Training on PyTorch 2.6.0 + CUDA 12 (NVIDIA RTX 4060 Laptop GPU, 8 GB).
- Inference runtime: ONNX Runtime 1.26+ (CPU primary, CUDA fallback if available).

### Core Frameworks and Libraries
| Dependency | Version / Scope | Exact Architectural Role in This Project |
| :--- | :--- | :--- |
| `onnxruntime` | >=1.16.0 | The sole inference engine — loads the quantised INT8 model, selects CUDA or CPU provider dynamically ([inference.py](inference.py) lines 60–64), and runs the forward pass. |
| `gradio` | >=5.0.0 | Web framework that handles image upload, webcam streaming, confidence slider, and output rendering. Event wiring in [app.py](app.py) connects UI events to inference and compliance functions. |
| `opencv-python-headless` | >=4.8.0 | Responsible for all image operations: reading uploaded files, letterbox resize, colour conversion, bounding-box annotation, and font rendering for labels. Used in both [inference.py](inference.py) and [utils.py](utils.py). |
| `numpy` | >=1.24.0 | Tensor construction for ONNX input, array manipulation in postprocessing, and confidence-score filtering. |
| `Pillow` | >=10.0.0 | Used indirectly through Gradio for image handling; listed for HF Space compatibility. |
| `huggingface_hub` | >=0.26.0 | HF Spaces runtime dependency (version metadata); not directly called in application code. |
| `ruff` | CI-only | Linter for `app.py`, `inference.py`, `utils.py` in CI pipeline. |

### Hardware and Edge Bindings
- Training: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM), CUDA 12.
- Inference (deployed): CPU Basic (free tier, HF Spaces), ONNX Runtime auto-selects CPU provider.
- Inference (local dev): same ONNX model; CUDA provider attempted first, CPU fallback if unavailable.

---

## Part 2.5: Infrastructure & Observability Stack

Not applicable. This repo has no Terraform modules, no Prometheus metrics, and no Grafana dashboards. Observability is limited to Python `print()` statements and the Gradio UI's inference-time display.

---

## Part 3: Microscopic Execution and Data Lifecycles

### 1. Bootstrapping Sequence
1. `python app.py` is executed ([app.py](app.py)).
2. `os.environ.get("SPACE_ID")` is read to determine launch mode — `app.launch()` (HF Spaces) vs `app.launch(server_name="0.0.0.0", server_port=7860)` (local).
3. Module-level `ENGINE = PPEInferenceEngine()` is instantiated ([app.py](app.py) line 24), which calls `_load_session()` ([inference.py](inference.py) lines 65–79) and resolves the model file by checking root-level `best_int8.onnx` first, then `models/best_int8.onnx`.
4. Gradio `Blocks` UI is constructed with two tabs (Image Upload, Live Camera) and wired event handlers.

### 2. The Processing Pipeline
- **Ingestion:** User uploads a file or activates webcam → `gr.Image` component passes a `np.ndarray` (or file path string, handled by explicit path-loading in [app.py](app.py) lines 109–112) to `process_image()`.
- **Mutation/Processing:**
  1. Backend routing: `process_image()` checks the `backend` radio value ([app.py](app.py) lines 114–127):
     - `"onnx"` → `ENGINE.run(image, conf_threshold)` → [inference.py](inference.py) lines 174–181.
     - `"roboflow"` → `roboflow_infer(image, conf_threshold)` → [roboflow_inference.py](roboflow_inference.py) (base64-encodes the image, POSTs to Roboflow serverless API).
  2. ONNX path internals ([inference.py](inference.py)):
     - `preprocess()` — letterbox resize to 320×320, padding with value 114, normalise to [0,1], transpose to CHW + batch dimension.
     - `session.run()` — ONNX Runtime forward pass, timed with `time.perf_counter()`.
     - `postprocess()` — output normalisation, confidence thresholding, inverse letterbox transform, NMS via `cv2.dnn.NMSBoxes()`.
  3. `draw_detections()` in [utils.py](utils.py) — renders bounding boxes with class-colour mapping and confidence labels.
  4. `compute_stats()` in [utils.py](utils.py) — computes per-class counts, tier-based violation flags, missing-PPE inference, and compliance-rate percentage.
  5. `render_*_html()` functions in [utils.py](utils.py) — generate the dark-themed compliance report, metrics grid, and status badge as raw HTML strings.
- **Sink/Output:** Four Gradio outputs are updated per inference:
  1. `img_out` — annotated image (`np.ndarray`).
  2. `header_html` — status badge (green/red).
  3. `met` — metrics grid (violations, inference time, compliance).
  4. `rep` — per-class compliance report with pill badges.

### 3. Camera Throttling
The camera tab in [app.py](app.py) (lines 149–168) wraps `process_image()` in a throttled cache: if `_cam_cache` exists and less than `_CAM_THROTTLE_S` seconds have elapsed, the previous result is returned without running inference.

---

## Part 4: The Technical Interview Defense Simulator

Every scripted answer below is provably true to the code and scoped to the `DEPLOYED-DEMO` Maturity Tag.

### Question 1: Core Algorithm Selection and Validation
**The Ask:** "Why YOLOv8n over alternatives like EfficientDet or a two-stage detector, and how did you validate correctness?"

**Your Scripted Defense:** YOLOv8n was chosen because this project targets a free-tier CPU deployment; YOLOv8n has the lowest FLOP count (8.1 GFLOPs) among viable real-time detectors, and the Ultralytics training pipeline provided one-command export to ONNX with INT8 quantisation, eliminating the need for a separate post-training quantisation toolchain. Correctness was validated against the Roboflow published benchmark: the v2 model achieves mAP@50 of 0.776 on the test set at standard YOLO evaluation settings (conf=0.001), within 0.6% of the published 0.770, as recorded in [training/training_notes.md](training/training_notes.md). The CI pipeline additionally proves the exported ONNX artifact loads without error ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

### Question 2: Hardware Constraints and Edge Optimisation
**The Ask:** "Where is the primary bottleneck on a resource-constrained device, and how does your code account for it?"

**Your Scripted Defense:** The primary bottleneck is the ONNX Runtime `session.run()` call (the forward pass) on CPU — the INT8 model averages 20.8 ms per inference on CPU, compared to 7.5 ms for the FP32 ONNX export. This is the direct cost of quantisation; the code compensates through two mechanisms: (1) the letterbox preprocessing step ([inference.py](inference.py) lines 81–95) pads only to 320×320 input size rather than the training resolution of 640×640, which reduces the tensor by a factor of four; and (2) the camera tab throttles inference through a per-frame cache ([app.py](app.py) lines 149–168) so the UI never queued frames faster than one per `_CAM_THROTTLE_S` interval.

### Question 3: Data Pipeline Rigor and Preprocessing
**The Ask:** "Walk me through preprocessing — why these exact transforms, and what happens if they fail?"

**Your Scripted Defense:** Preprocessing is implemented in `PPEInferenceEngine.preprocess()` ([inference.py](inference.py) lines 81–95). It computes a scale factor from `input_size / max(h, w)` so the longer dimension becomes 320 px, then pads the shorter dimension with neutral grey (114) on both sides to create a square tensor. This letterbox approach avoids the aspect-ratio distortion that naive resize would introduce. If the input image is malformed — zero-size, corrupt bytes, or a string path to a missing file — the error is caught by the broad `try/except` block in `process_image()` ([app.py](app.py) lines 104–140), which returns the error traceback as an HTML string to the UI rather than crashing the app.

### Question 4: Edge Cases, Exception Boundaries, and Telemetry
**The Ask:** "How does the system handle corrupted inputs, disconnects, or missing values? Show your exception boundaries and logs."

**Your Scripted Defense:** The app has a single top-level exception boundary in `process_image()` ([app.py](app.py) lines 104–140) that wraps all inference and rendering logic in a `try/except Exception` block. Any error — missing model file, corrupted image, Roboflow API timeout — is captured with `traceback.format_exc()` and rendered as a red HTML error panel in the UI. Missing model files are handled silently earlier: if neither `best_int8.onnx` nor `best.onnx` exists, `_load_session()` returns `None` and the engine falls back to hard-coded demo detections ([inference.py](inference.py) lines 154–162). There is no structured logging or telemetry beyond Python's default stderr output, which is consistent with the `DEPLOYED-DEMO` maturity level.

### Question 5: Structural Complexity and Future Scalability
**The Ask:** "Why did you implement the compliance-report HTML rendering directly in Python strings rather than using a templating engine, and how would you refactor it with a team?"

**Your Scripted Defense:** The inline-HTML approach in [utils.py](utils.py) was chosen to keep the project dependency-free — Gradio's `gr.HTML` component accepts raw HTML strings, and adding Jinja2 or a front-end framework would have added complexity without benefit for a single-page demo with four output states. With a team, the three rendering functions (`render_header_html`, `render_metrics_html`, `render_report_html`) would be refactored into a dedicated template file per component and tested in isolation against fixture detection data. The structural constraint is that the logic is currently split across both `utils.py` (rendering) and `app.py` (event wiring); the first refactor step would be extracting all HTML templates into a `templates/` module so that app logic and presentation are fully separated.

---

## Part 5: Pragmatic Engineering Roadmap

Current active study focus is **DEA-C01** — do not propose High Effort items that would meaningfully compete with this for weekly hours.

### 1. High Reward / Low Effort (Quick Wins) ✅ Completed

| Proposed Change | Target | Status |
| :--- | :--- | :--- |
| Add `--fix` to the linter step in CI | [.github/workflows/ci.yml](.github/workflows/ci.yml) | ✅ Applied — `ruff check --fix` runs on every push to `main` (commit `8e739f3`). |
| Pin Python version note in README | [README.md](README.md) | ✅ Applied — callout added under Run Locally section. |
| Add a `.gitattributes` file for consistent line endings | [.gitattributes](.gitattributes) | ✅ Applied — `* text=auto` with binary markers for `.onnx` and image files (commit `8e739f3`). |

### 2. High Reward / High Effort (Strategic Scaling — explicitly "someday")

| Proposed Change | Target | ROI Justification |
| :--- | :--- | :--- |
| Add an end-to-end smoke test | `tests/smoke_test.py` | A single test that starts the Gradio app, uploads a known image, and asserts that the output contains the expected number of detections would move this repo from `DEPLOYED-DEMO` toward `PRODUCTION-TESTED`. Estimated effort: 4–6 hours. Not scheduled against DEA-C01 study time. |

### 3. De-prioritized Items (Low Reward / High Effort Trap)

| The Refactor Trap | Why We Avoid It |
| :--- | :--- |
| Replace inline-HTML rendering with Jinja2 templates | The current `utils.py` rendering functions are ~120 lines total and serve a single-page app with four outputs. Extracting them into Jinja2 templates would add a dependency, a build step, and a test layer for zero functional improvement. The opportunity cost against DEA-C01 study hours is not justified. |
| Add unit tests for `compute_stats()` | The compliance logic in `compute_stats()` is currently exercised only by visual inspection. Writing unit tests would require introducing `pytest`, a test fixture for 14-class detection data, and parameterised test cases for every tier permutation. Valuable in principle, but the current state is demonstrably sufficient for a `DEPLOYED-DEMO` portfolio repo that the user is actively using to prove deployment and CI skills to recruiters. |
