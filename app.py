import os
import time as _time
import warnings

import gradio as gr
import numpy as np

from inference import PPEInferenceEngine
from roboflow_inference import run_inference as roboflow_infer
from utils import (
    compute_stats,
    draw_detections,
    render_footer_html,
    render_header_html,
    render_metrics_html,
    render_report_html,
)


ENGINE = PPEInferenceEngine()

warnings.filterwarnings(
    'ignore',
    message=r'The parameters have been moved from the Blocks constructor to the launch\(\) method in Gradio 6.0: css.*',
)


CSS = """
body, .gradio-container {
  background-color: #0f0f1a !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}

footer { display: none !important; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; padding: 16px !important; }

#ppe-app {
  background: #13132a;
  border-radius: 12px;
  border: 0.5px solid #2a2a40;
  overflow: hidden;
}

.ppe-card {
  background: #1a1a2e !important;
  border: 0.5px solid #2a2a40 !important;
  border-radius: 10px !important;
  padding: 14px !important;
}

.upload-zone .wrap { background: #22223a !important; border: 1px dashed #3a3a55 !important; border-radius: 8px !important; }
.upload-zone .wrap:hover { border-color: #5555aa !important; }
.upload-zone svg { color: #6060a0 !important; }
.upload-zone .upload-text { color: #a0a0c0 !important; }

.output-image { border-radius: 8px !important; overflow: hidden !important; background: #22223a !important; }

input[type=range] { accent-color: #185FA5 !important; }
.gradio-slider label { color: #a0a0c0 !important; font-size: 12px !important; }
.gradio-slider .value-indicator { color: #e0e0f0 !important; font-weight: 500 !important; }

#run-btn {
  background: #185FA5 !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 10px !important;
  cursor: pointer !important;
}
#run-btn:hover { background: #1a72c4 !important; }

.step-header { font-size: 16px; font-weight: 500; color: #e0e0f0; }
.step-sub    { font-size: 11px; color: #7070a0; margin-top: 2px; }

.html-no-label > label { display: none !important; }
.gap-row { gap: 10px !important; }
"""


_STEP = (
    '<span style="width:22px;height:22px;border-radius:50%;display:inline-flex;'
    'align-items:center;justify-content:center;font-size:11px;font-weight:500;'
    'margin-right:8px;flex-shrink:0;background:{bg};color:{fg}">{n}</span>'
)


def _step_html(n, bg, fg, title, sub):
    num = _STEP.format(bg=bg, fg=fg, n=n)
    return (
        f'<div style="display:flex;align-items:flex-start;margin-bottom:10px">'
        f'{num}'
        f'<div><div class="step-header">{title}</div>'
        f'<div class="step-sub">{sub}</div></div></div>'
    )


def process_image(image: np.ndarray | None, conf_threshold: float, backend: str = "onnx"):
    try:
        if image is None:
            return (
                None,
                render_header_html(None, ENGINE.model_label),
                '<div style="color:#7070a0;font-size:13px;text-align:center;padding:24px">Upload an image to see results</div>',
                '<div style="color:#7070a0;font-size:13px;text-align:center;padding:24px">No image processed yet</div>',
            )

        # Gradio 5 may pass a file path string instead of numpy array
        if isinstance(image, str):
            import cv2
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
            if image is None:
                raise ValueError(f'Could not load image from path: {image}')

        # Route to selected backend
        if backend == "roboflow":
            if not os.environ.get("ROBOFLOW_API_KEY"):
                raise RuntimeError(
                    "ROBOFLOW_API_KEY not set. "
                    "HF Spaces: Settings → Variables and secrets → New secret."
                )
            detections, inference_ms = roboflow_infer(image, conf_threshold)
            model_label = "Roboflow personal-protective-equipment-combined-model/8"
        else:
            detections, inference_ms = ENGINE.run(image, conf_threshold)
            model_label = ENGINE.model_label

        annotated = draw_detections(image, detections)
        stats = compute_stats(detections)

        return (
            annotated,
            render_header_html(stats, model_label),
            render_metrics_html(stats, inference_ms),
            render_report_html(stats),
        )
    except Exception as exc:
        import traceback
        err_msg = f'<div style="color:#ff6b6b;padding:12px;font-size:13px"><b>Error:</b> {exc}<br><pre style="font-size:10px">{traceback.format_exc()}</pre></div>'
        return (
            image if isinstance(image, np.ndarray) else None,
            render_header_html(None, ENGINE.model_label),
            err_msg,
            err_msg,
        )


# ── Camera throttling: skip frames more frequent than CAM_THROTTLE_S ─────────
_CAM_THROTTLE_S = 0.0                # seconds between live-camera inferences
_cam_cache: dict = {}                # last result from camera tab
_cam_last_ts: float = 0.0            # timestamp of last inference


def process_camera_frame(image: np.ndarray | None, conf_threshold: float, backend: str = "onnx"):
    """Throttled wrapper — returns cached result when called too soon."""
    global _cam_cache, _cam_last_ts

    if image is None:
        _cam_cache = {}
        return (
            None,
            render_header_html(None, ENGINE.model_label),
            '<div style="color:#7070a0;font-size:13px;text-align:center;padding:24px">'
            'Start webcam to see live report</div>',
            '<div style="color:#7070a0;font-size:13px;text-align:center;padding:24px">'
            'Awaiting camera stream</div>',
        )

    now = _time.perf_counter()
    if _cam_cache and (now - _cam_last_ts) < _CAM_THROTTLE_S:
        # Return cached result — skip this frame
        return _cam_cache["result"]

    result = process_image(image, conf_threshold, backend)
    _cam_cache = {"result": result}
    _cam_last_ts = now
    return result


_METRICS_PLACEHOLDER = (
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px">'
    + ''.join(
        f'<div style="background:#0e0e1c;border-radius:8px;padding:10px 13px">'
        f'<div style="font-size:22px;font-weight:500;color:#3a3a5a;line-height:1.1">—</div>'
        f'<div style="font-size:11px;color:#4a4a70;margin-top:4px">{lbl}</div></div>'
        for lbl in ['Violations', 'Inference', 'Warnings', 'Compliance']
    )
    + '</div>'
)


def _build_upload_tab():
    """Return (img_input, conf_slider, run_btn, img_output, metrics_html, report_html)."""
    with gr.TabItem("📁 Image Upload"):
        with gr.Row(equal_height=False, elem_classes="gap-row"):
            with gr.Column(scale=1, elem_classes="ppe-card"):
                gr.HTML(_step_html(1, '#1e2d48', '#7ab3e0', 'Input', 'Upload a site image'))

                img_in = gr.Image(
                    type="numpy", label="",
                    sources=["upload", "clipboard"],
                    elem_classes="upload-zone", show_label=False, height=200,
                )

                conf_sl = gr.Slider(
                    minimum=0.20, maximum=0.90, value=0.30, step=0.05,
                    label="Confidence threshold",
                    info="Lower = more detections · Higher = more confident",
                )

                backend = gr.Radio(
                    choices=["onnx", "roboflow"], value="onnx",
                    label="Inference backend",
                    info="ONNX (local, ~11ms) · Roboflow (serverless API, ~1.5s)",
                )

                btn = gr.Button("Run detection", variant="primary", elem_id="run-btn")

            with gr.Column(scale=1, elem_classes="ppe-card"):
                gr.HTML(_step_html(2, '#0e2a1e', '#5DCAA5', 'Annotated output', 'Bounding boxes + labels'))

                img_out = gr.Image(
                    type="numpy", label="", show_label=False,
                    interactive=False, elem_classes="output-image", height=200,
                )

                met = gr.HTML(value=_METRICS_PLACEHOLDER, elem_classes="html-no-label")

            with gr.Column(scale=1, elem_classes="ppe-card"):
                gr.HTML(_step_html(3, '#2a2010', '#BA7517', 'Compliance report', 'Per-class breakdown'))

                rep = gr.HTML(
                    value='<div style="color:#4a4a70;font-size:13px;text-align:center;padding:20px 0">'
                          'Run detection to see compliance report</div>',
                    elem_classes="html-no-label",
                )

        btn.click(
            fn=process_image,
            inputs=[img_in, conf_sl, backend],
            outputs=[img_out, header_html, met, rep],
        )
        conf_sl.change(
            fn=process_image,
            inputs=[img_in, conf_sl, backend],
            outputs=[img_out, header_html, met, rep],
        )
        return img_in, conf_sl, backend, btn, img_out, met, rep


def _build_camera_tab():
    """Return (cam_input, conf_slider, cam_output, metrics_html, report_html)."""
    with gr.TabItem("📷 Live Camera"):
        gr.Markdown("> Click **Start** to enable webcam. Detection runs on every frame.")

        with gr.Row(equal_height=False, elem_classes="gap-row"):
            with gr.Column(scale=1, elem_classes="ppe-card"):
                gr.HTML(_step_html(1, '#1e2d48', '#7ab3e0', 'Webcam feed', 'Real-time frame capture'))

                cam_in = gr.Image(
                    type="numpy", label="",
                    sources=["webcam"], streaming=True,
                    elem_classes="upload-zone", show_label=False, height=200,
                )

                conf_sl = gr.Slider(
                    minimum=0.20, maximum=0.90, value=0.30, step=0.05,
                    label="Confidence threshold",
                    info="Lower = more detections · Higher = more confident",
                )

                backend = gr.Radio(
                    choices=["onnx", "roboflow"], value="onnx",
                    label="Inference backend",
                    info="ONNX (local, ~11ms) · Roboflow (serverless API, ~1.5s)",
                )

            with gr.Column(scale=1, elem_classes="ppe-card"):
                gr.HTML(_step_html(2, '#0e2a1e', '#5DCAA5', 'Detected output', 'Bounding boxes + labels'))

                cam_out = gr.Image(
                    type="numpy", label="", show_label=False,
                    interactive=False, elem_classes="output-image", height=200,
                )

                met = gr.HTML(value=_METRICS_PLACEHOLDER, elem_classes="html-no-label")

            with gr.Column(scale=1, elem_classes="ppe-card"):
                gr.HTML(_step_html(3, '#2a2010', '#BA7517', 'Live compliance report', 'Per-class breakdown'))

                rep = gr.HTML(
                    value='<div style="color:#4a4a70;font-size:13px;text-align:center;padding:20px 0">'
                          'Start webcam to see live report</div>',
                    elem_classes="html-no-label",
                )

        cam_in.stream(
            fn=process_camera_frame,
            inputs=[cam_in, conf_sl, backend],
            outputs=[cam_out, header_html, met, rep],
            concurrency_limit=1,
        )
        conf_sl.change(
            fn=process_camera_frame,
            inputs=[cam_in, conf_sl, backend],
            outputs=[cam_out, header_html, met, rep],
        )
        return cam_in, conf_sl, backend, cam_out, met, rep


with gr.Blocks(css=CSS, title="Real-Time PPE Detection System") as app:
    gr.HTML(
        value=(
            '<div style="padding:12px 20px 0 20px;color:#e0e0f0;font-size:18px;font-weight:600">'
            'Real-Time PPE Detection System</div>'
        )
    )

    header_html = gr.HTML(value=render_header_html(None, ENGINE.model_label), elem_id="ppe-header")

    with gr.Tabs():
        _build_upload_tab()
        _build_camera_tab()

    footer_html = gr.HTML(value=render_footer_html())


if __name__ == "__main__":
    if os.environ.get("SPACE_ID"):
        app.launch()
    else:
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)