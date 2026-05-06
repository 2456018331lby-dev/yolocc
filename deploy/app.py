"""
Streamlit Demo — Industrial Grade
Features:
  - Image upload + camera capture
  - Adjustable parameters
  - Detection result table
  - Model info panel
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import YOLODetector

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="YOLO Garbage Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stMetric {background: #f0f2f6; padding: 15px; border-radius: 10px;}
    .detection-box {border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; margin: 5px 0;}
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_detector(weights_path: str, imgsz: int, conf: float, iou: float):
    return YOLODetector(weights=weights_path, imgsz=imgsz, conf=conf, iou=iou)


def find_weights():
    """Find best available weights."""
    candidates = [
        PROJECT_ROOT / "results/runs/train/garbage_detect/weights/best.onnx",
        PROJECT_ROOT / "results/runs/train/garbage_detect/weights/best.pt",
        PROJECT_ROOT / "weights/best.onnx",
        PROJECT_ROOT / "weights/best.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")

    default_weights = find_weights()
    weights_path = st.text_input("Model Path", value=default_weights or "")

    st.markdown("---")
    imgsz = st.slider("Image Size", 320, 1280, 640, step=32,
                       help="Input size. ONNX models auto-detect this from model metadata.")
    conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("NMS IoU", 0.0, 1.0, 0.45, 0.01)

    st.markdown("---")
    st.markdown("**Classes:**")
    st.markdown("- 🟢 Recyclable (可回收物)")
    st.markdown("- 🔴 Hazardous (有害垃圾)")
    st.markdown("- 🔵 Kitchen (厨余垃圾)")
    st.markdown("- ⚫ Other (其他垃圾)")

# ── Main ─────────────────────────────────────────────────────────────────────

st.title("YOLO Garbage Classification Detection")
st.markdown("Real-time garbage classification using YOLOv8 — Upload an image or use your camera")

if not weights_path or not Path(weights_path).exists():
    st.warning("Provide a valid model path in the sidebar.")
    st.info("Train first: `python -m src.train` or `yolocc train`")
    st.stop()

try:
    detector = load_detector(weights_path, imgsz, conf, iou)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# Tabs
tab_upload, tab_camera, tab_info = st.tabs(["Image Upload", "Camera", "Model Info"])

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded:
        col_orig, col_result = st.columns(2)

        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:
            with col_orig:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original")

            t0 = time.time()
            detections = detector.detect_frame(frame)
            elapsed = time.time() - t0

            annotated = detector.draw(frame.copy(), detections)

            with col_result:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result")

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects", len(detections))
            m2.metric("Inference", f"{elapsed*1000:.1f}ms")
            m3.metric("FPS", f"{1/elapsed:.1f}")

            # Results table
            if detections:
                st.markdown("**Detections:**")
                for i, d in enumerate(detections):
                    st.markdown(
                        f"{i+1}. **{d.class_name}** — "
                        f"Confidence: `{d.confidence:.3f}` — "
                        f"Box: `[{d.x1}, {d.y1}, {d.x2}, {d.y2}]`"
                    )
            else:
                st.info("No objects detected.")

with tab_camera:
    camera_input = st.camera_input("Take a photo")

    if camera_input:
        file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:
            detections = detector.detect_frame(frame)
            annotated = detector.draw(frame.copy(), detections)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result")

            if detections:
                for d in detections:
                    st.markdown(f"- **{d.class_name}** ({d.confidence:.3f})")

with tab_info:
    st.subheader("Model Information")

    model_path = Path(weights_path)
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)

        info_data = {
            "Property": ["Model", "Size", "Backend", "Image Size (actual)", "Confidence", "IoU"],
            "Value": [
                model_path.name,
                f"{size_mb:.2f} MB",
                detector.backend,
                f"{detector.imgsz} (slider: {imgsz})" if detector.imgsz != imgsz else str(imgsz),
                str(conf),
                str(iou),
            ],
        }
        st.table(info_data)

    bench_path = PROJECT_ROOT / "results/benchmark_results.json"
    if bench_path.exists():
        import json
        with open(bench_path) as f:
            bench = json.load(f)
        st.subheader("Benchmark Results")
        for r in bench:
            st.markdown(
                f"- **{r['backend']}**: {r['mean_ms']:.2f}ms "
                f"({r['fps']:.1f} FPS) — {r['model_size_mb']:.2f} MB"
            )
