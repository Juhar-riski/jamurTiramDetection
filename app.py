"""
Author : Juhar Riski Ahmadi (22017164)
Jalankan dengan:
    streamlit run app.py
"""

import io
import re
import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO

# Konfigurasi dasar
BASE_DIR = Path(__file__).resolve().parent
CNN_PATH = BASE_DIR / "models" / "cnnModel.pth"
YOLO_PATH = BASE_DIR / "models" / "yoloModel.pt"

CLASS_NAMES = {0: "Belum Layak Panen", 1: "Layak Panen"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 640
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Helper: render HTML tanpa kena bug "indentasi dianggap code block"
def render_html(html: str):
    """Streamlit markdown menganggap baris berindentasi 4+ spasi sebagai
    code block. Karena HTML di file ini ditulis dengan indentasi rapi,
    ratakan dulu sebelum dikirim ke st.markdown."""
    flat = re.sub(r"\n[ \t]+", "\n", html).strip()
    st.markdown(flat, unsafe_allow_html=True)


# Loader model (cache supaya tidak reload tiap interaksi)
@st.cache_resource(show_spinner=False)
def load_cnn_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(CNN_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_yolo_model():
    return YOLO(str(YOLO_PATH))


def predict_cnn(model, pil_image: Image.Image):
    start = time.perf_counter()
    tensor = PREPROCESS(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    return {
        "label": CLASS_NAMES[int(idx.item())],
        "confidence": float(conf.item()) * 100,
        "time_ms": elapsed,
        "probs": {CLASS_NAMES[i]: float(p) * 100 for i, p in enumerate(probs)},
    }


def predict_yolo(model, pil_image: Image.Image):
    start = time.perf_counter()
    results = model.predict(pil_image.convert("RGB"), verbose=False, conf=0.25)
    elapsed = (time.perf_counter() - start) * 1000  # ms

    r = results[0]
    boxes = r.boxes
    detections = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item()) * 100
            label = model.names.get(cls_id, str(cls_id))
            detections.append({"label": label, "confidence": conf})
        detections.sort(key=lambda d: d["confidence"], reverse=True)

    annotated_bgr = r.plot()  # numpy array BGR
    annotated_rgb = annotated_bgr[:, :, ::-1]

    return {
        "detections": detections,
        "time_ms": elapsed,
        "annotated_image": annotated_rgb,
    }


# Tampilan / styling
st.set_page_config(
    page_title="Perbandingan Model CNN & YOLO",
    layout="wide",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #1b1f2a 0%, #0e1117 45%, #0a0c10 100%);
    color: #e6e6e6;
}

#MainMenu, header, footer {visibility: hidden;}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

/* --- TAMBAHAN: samakan tinggi antar kolom (versi lengkap) --- */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
div[data-testid="column"] {
    display: flex;
    height: auto;
}
div[data-testid="column"] > div {
    width: 100%;
    display: flex;
    flex-direction: column;
    flex: 1;
}
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"],
div[data-testid="element-container"],
div[data-testid="stMarkdown"],
div[data-testid="stMarkdownContainer"] {
    display: flex;
    flex-direction: column;
    flex: 1;
    width: 100%;
    height: 100%;
}
/* --- akhir tambahan --- */

.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.6rem;
}
.hero h1 {
    margin: 0;
    font-size: 1.7rem;
    font-weight: 700;
    color: #f4f4f4;
}
.hero p {
    margin: 0.4rem 0 0 0;
    color: #9aa3b2;
    font-size: 0.95rem;
}

.card {
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
.card h3 {
    margin-top: 0;
    font-size: 1.05rem;
    color: #d8dde6;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.badge {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.2px;
    align-self: flex-start;
}
.badge-layak {
    background: rgba(56, 189, 119, 0.15);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.35);
}
.badge-belum {
    background: rgba(248, 113, 113, 0.12);
    color: #f87171;
    border: 1px solid rgba(248, 113, 113, 0.35);
}

.metric-row {
    display: flex;
    gap: 0.6rem;
    margin-top: auto;
    padding-top: 0.9rem;
    flex-wrap: wrap;
}
.metric-pill {
    flex: 1;
    min-width: 110px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 0.55rem 0.7rem;
}
.metric-pill .label {
    font-size: 0.72rem;
    color: #828a99;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-pill .value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    color: #e8eaf0;
    margin-top: 0.1rem;
}

.bar-track {
    height: 7px;
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    overflow: hidden;
    margin-top: 0.25rem;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
}

.detect-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px dashed rgba(255,255,255,0.07);
    font-size: 0.92rem;
}
.detect-row:last-child { border-bottom: none; }

.empty-note {
    color: #828a99;
    font-size: 0.9rem;
    padding: 0.6rem 0;
}

.summary-pill {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    text-align: center;
}
.summary-pill .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8eaf0;
}
.summary-pill .cap {
    font-size: 0.78rem;
    color: #828a99;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.15rem;
}
.summary-pill.green .num { color: #4ade80; }
.summary-pill.red .num { color: #f87171; }

.row-thumb {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 0.6rem;
    background: rgba(255,255,255,0.02);
}
.row-thumb .fname {
    font-size: 0.78rem;
    color: #828a99;
    margin-top: 0.45rem;
    word-break: break-all;
}

div[role="radiogroup"] {
    gap: 0.5rem;
}
div[role="radiogroup"] label {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.4rem 0.9rem;
    margin-right: 0.3rem;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.025);
    border: 1.5px dashed rgba(255,255,255,0.15);
    border-radius: 14px;
}

div.stButton > button {
    background: linear-gradient(135deg, #3a7bd5, #2e5fa3);
    border: none;
    border-radius: 10px;
    padding: 0.55rem 1.3rem;
    font-weight: 600;
    color: white;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #5a9bf5, #4b7fc3);
    color: white;
}

hr { border-color: rgba(255,255,255,0.07); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

render_html(
    """
    <div class="hero">
        <h1>Perbandingan Model CNN(ResNet50) & YOLOv11</h1>
        <p>Program ini berfungsi menampilkan hasil pendeteksian dari model CNN dan model YOLOv11</p>
    </div>
    """
)


def confidence_color(label: str) -> str:
    return "#4ade80" if "Layak" in label and "Belum" not in label else "#f87171"


def render_cnn_card(result):
    label = result["label"]
    is_layak = "Belum" not in label
    badge_class = "badge-layak" if is_layak else "badge-belum"

    rows_html = ""
    for cls_label, pct in sorted(result["probs"].items(), key=lambda x: -x[1]):
        c = confidence_color(cls_label)
        rows_html += f"""
        <div style="margin-top:0.55rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:#9aa3b2;">
                <span>{cls_label}</span><span>{pct:.1f}%</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%; background:{c};"></div></div>
        </div>
        """

    render_html(
        f"""
        <div class="card">
            <h3>Model CNN <span style="font-size:0.75rem; color:#828a99; font-weight:400;">(ResNet-50)</span></h3>
            <span class="badge {badge_class}">{label}</span>
            {rows_html}
            <div class="metric-row">
                <div class="metric-pill">
                    <div class="label">Confidence</div>
                    <div class="value">{result['confidence']:.2f}%</div>
                </div>
                <div class="metric-pill">
                    <div class="label">Waktu Deteksi</div>
                    <div class="value">{result['time_ms']:.1f} ms</div>
                </div>
            </div>
        </div>
        """
    )


def render_yolo_card(result):
    detections = result["detections"]

    if detections:
        top = detections[0]
        is_layak = "Belum" not in top["label"]
        badge_class = "badge-layak" if is_layak else "badge-belum"
        top_label_display = top["label"].replace("_", " ").title()
        rows_html = ""
        for d in detections:
            c = confidence_color(d["label"])
            disp = d["label"].replace("_", " ").title()
            rows_html += f"""
            <div class="detect-row">
                <span>{disp}</span>
                <span style="color:{c}; font-family:'JetBrains Mono', monospace;">{d['confidence']:.1f}%</span>
            </div>
            """
        top_conf_display = f"{top['confidence']:.2f}%"
    else:
        badge_class = ""
        top_label_display = "Tidak ada objek terdeteksi"
        rows_html = '<div class="empty-note">Coba foto dengan jarak/cahaya lebih jelas.</div>'
        top_conf_display = "-"

    badge_html = (
        f'<span class="badge {badge_class}">{top_label_display}</span>'
        if detections
        else f'<span style="color:#828a99;">{top_label_display}</span>'
    )

    render_html(
        f"""
        <div class="card">
            <h3>Model YOLO <span style="font-size:0.75rem; color:#828a99; font-weight:400;">(YOLOv11)</span></h3>
            {badge_html}
            <div style="margin-top:0.7rem;">
                {rows_html}
            </div>
            <p> </p>
            <div>
            </div>
            <div class="metric-row">
                <div class="metric-pill">
                    <div class="label">Top Confidence</div>
                    <div class="value">{top_conf_display}</div>
                </div>
                <div class="metric-pill">
                    <div class="label">Waktu Deteksi</div>
                    <div class="value">{result['time_ms']:.1f} ms</div>
                </div>
                <div class="metric-pill">
                    <div class="label">Objek Terdeteksi</div>
                    <div class="value">{len(detections)}</div>
                </div>
            </div>
        </div>
        """
    )


# main funcionl
def main():
    uploaded_file = st.file_uploader(
        "Upload foto jamur tiram", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        col_left, col_center, col_right = st.columns([1, 3, 1])
        with col_center:
            st.image(pil_image, caption="Gambar yang diupload", use_column_width=True)
            run = st.button("Jalankan Deteksi", use_container_width=True)
        
        if run:
            with st.spinner("Memuat model & menjalankan deteksi..."):
                cnn_model = load_cnn_model()
                yolo_model = load_yolo_model()

                cnn_result = predict_cnn(cnn_model, pil_image)
                yolo_result = predict_yolo(yolo_model, pil_image)

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                render_cnn_card(cnn_result)
            with col2:
                render_yolo_card(yolo_result)

            if yolo_result["detections"]:
                st.markdown("<br>", unsafe_allow_html=True)
                render_html(
                    "<div class='card'><h3>Bounding Box dari YOLO</h3></div>"
                )
                st.image(
                    yolo_result["annotated_image"],
                    use_column_width=True,
                )

            total_time = cnn_result["time_ms"] + yolo_result["time_ms"]
            st.caption(f"Total waktu pemrosesan kedua model: {total_time:.1f} ms")
    else:
        render_html(
            "<div class='empty-note'>Belum ada gambar. Upload foto rumpun jamur tiram untuk mulai deteksi.</div>"
        )

st.markdown("<br>", unsafe_allow_html=True)
main()
