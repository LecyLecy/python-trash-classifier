from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference.api import INSTRUCTIONS, PredictConfig, load_model, predict_pil_ui


ROOT_DIR = Path(__file__).resolve().parent
METRICS_PATH = ROOT_DIR / "metrics" / "metrics.json"
WEIGHTS_PATH = ROOT_DIR / "models" / "model.pth"
LABELS_PATH = ROOT_DIR / "models" / "labels.json"

CLASS_DETAILS = {
    "cardboard": {
        "icon": "▤",
        "color": "#B7794B",
        "bin": "Paper & cardboard",
        "description": "Boxes, cartons, and corrugated packaging",
    },
    "glass": {
        "icon": "◇",
        "color": "#45A99A",
        "bin": "Glass recycling",
        "description": "Bottles, jars, and other glass containers",
    },
    "metal": {
        "icon": "⬡",
        "color": "#718096",
        "bin": "Metal recycling",
        "description": "Cans, tins, and clean metal packaging",
    },
    "paper": {
        "icon": "▱",
        "color": "#D7A93E",
        "bin": "Paper recycling",
        "description": "Clean paper, magazines, and office sheets",
    },
    "plastic": {
        "icon": "♳",
        "color": "#4A86E8",
        "bin": "Plastic recycling",
        "description": "Plastic bottles, tubs, and containers",
    },
    "trash": {
        "icon": "⌫",
        "color": "#7B6F83",
        "bin": "General waste",
        "description": "Non-recyclable or contaminated waste",
    },
}

st.set_page_config(
    page_title="ReSort — AI Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink: #16312a;
                --muted: #61716c;
                --brand: #167d5a;
                --brand-dark: #0d5f42;
                --mint: #eaf6f0;
                --paper: #f7faf8;
                --line: #dce8e2;
                --amber: #aa6b05;
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 4%, rgba(63, 170, 126, .10), transparent 25rem),
                    radial-gradient(circle at 92% 12%, rgba(226, 184, 74, .08), transparent 24rem),
                    #fbfdfc;
                color: var(--ink);
            }

            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stToolbar"] { visibility: hidden; }
            #MainMenu, footer { visibility: hidden; }

            .block-container {
                max-width: 1180px;
                padding-top: 2.2rem;
                padding-bottom: 4rem;
            }

            h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
            p { color: var(--muted); }

            .brand-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 4.2rem;
            }

            .wordmark {
                display: flex;
                align-items: center;
                gap: .7rem;
                color: var(--ink);
                font-weight: 760;
                font-size: 1.08rem;
                letter-spacing: -.02em;
            }

            .brand-mark {
                display: grid;
                place-items: center;
                width: 2.2rem;
                height: 2.2rem;
                border-radius: .7rem;
                background: var(--brand);
                color: white;
                font-size: 1.15rem;
                box-shadow: 0 8px 20px rgba(22, 125, 90, .22);
            }

            .project-tag {
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: .45rem .8rem;
                color: var(--muted);
                background: rgba(255, 255, 255, .72);
                font-size: .78rem;
                font-weight: 650;
            }

            .hero { max-width: 780px; margin: 0 auto 3.2rem; text-align: center; }

            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: .45rem;
                border: 1px solid #b9ddcc;
                border-radius: 999px;
                padding: .42rem .78rem;
                background: var(--mint);
                color: var(--brand-dark);
                font-size: .75rem;
                font-weight: 750;
                letter-spacing: .06em;
                text-transform: uppercase;
            }

            .hero h1 {
                margin: 1rem 0 .9rem;
                font-size: clamp(2.5rem, 6vw, 4.6rem);
                line-height: .98;
                font-weight: 790;
            }

            .hero h1 span { color: var(--brand); }
            .hero p { max-width: 650px; margin: 0 auto; font-size: 1.08rem; line-height: 1.7; }

            .section-label {
                margin-bottom: .4rem;
                color: var(--brand-dark);
                font-size: .72rem;
                font-weight: 780;
                letter-spacing: .09em;
                text-transform: uppercase;
            }

            .section-title {
                margin: 0 0 1.2rem;
                font-size: 1.55rem;
                font-weight: 740;
                color: var(--ink);
            }

            [data-testid="stFileUploader"] {
                padding: .6rem;
                border: 1px solid var(--line);
                border-radius: 1rem;
                background: rgba(255,255,255,.75);
            }

            [data-testid="stFileUploaderDropzone"] {
                min-height: 12rem;
                border: 1.5px dashed #9cc9b5;
                border-radius: .8rem;
                background: #f5fbf8;
            }

            .stButton > button {
                min-height: 3rem;
                border-radius: .7rem;
                border: 1px solid var(--brand);
                font-weight: 720;
                transition: transform .15s ease, box-shadow .15s ease;
            }

            .stButton > button[kind="primary"] {
                background: var(--brand);
                color: white !important;
                box-shadow: 0 8px 22px rgba(22, 125, 90, .18);
            }

            .stButton > button[kind="primary"] p {
                color: white !important;
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 24px rgba(22, 125, 90, .20);
            }

            .result-card {
                min-height: 14.5rem;
                border: 1px solid var(--line);
                border-radius: 1rem;
                padding: 1.45rem;
                background: rgba(255,255,255,.9);
                box-shadow: 0 14px 36px rgba(26, 68, 54, .07);
            }

            .result-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; }
            .result-identity { display: flex; align-items: center; gap: 1rem; }

            .class-icon {
                display: grid;
                place-items: center;
                width: 3.6rem;
                height: 3.6rem;
                border-radius: 1rem;
                color: white;
                font-size: 1.65rem;
                font-weight: 700;
            }

            .class-name { margin: 0; color: var(--ink); font-size: 1.7rem; font-weight: 760; text-transform: capitalize; }
            .class-description { margin: .2rem 0 0; font-size: .86rem; }

            .status-pill {
                display: inline-flex;
                border-radius: 999px;
                padding: .4rem .7rem;
                font-size: .7rem;
                font-weight: 760;
                white-space: nowrap;
            }

            .status-ready { color: #09613f; background: #dff4e9; }
            .status-review { color: #8a5600; background: #fff0cf; }

            .confidence-row { display: flex; align-items: baseline; justify-content: space-between; margin-top: 1.8rem; }
            .confidence-label { color: var(--muted); font-size: .78rem; font-weight: 650; }
            .confidence-value { color: var(--ink); font-size: 1.5rem; font-weight: 760; }

            .confidence-track { height: .5rem; overflow: hidden; border-radius: 999px; background: #e8efeb; }
            .confidence-fill { height: 100%; border-radius: inherit; background: var(--brand); }

            .guidance {
                display: flex;
                align-items: flex-start;
                gap: .8rem;
                margin-top: 1rem;
                padding: 1rem;
                border: 1px solid #c8e4d6;
                border-radius: .85rem;
                background: var(--mint);
            }

            .guidance-icon {
                display: grid;
                place-items: center;
                flex: 0 0 auto;
                width: 2rem;
                height: 2rem;
                border-radius: .6rem;
                color: white;
                background: var(--brand);
                font-weight: 760;
            }

            .guidance strong { display: block; margin-bottom: .15rem; color: var(--ink); font-size: .78rem; }
            .guidance p { margin: 0; color: #315d4e; font-size: .85rem; line-height: 1.5; }

            .empty-state {
                display: grid;
                place-items: center;
                min-height: 25rem;
                border: 1px dashed #b9d3c7;
                border-radius: 1rem;
                padding: 2rem;
                background: rgba(247, 251, 249, .75);
                text-align: center;
            }

            .empty-icon {
                display: grid;
                place-items: center;
                width: 4rem;
                height: 4rem;
                margin: 0 auto 1rem;
                border-radius: 1.2rem;
                color: var(--brand);
                background: var(--mint);
                font-size: 1.7rem;
            }

            .empty-state strong { color: var(--ink); font-size: 1rem; }
            .empty-state p { max-width: 280px; margin: .45rem auto 0; font-size: .84rem; line-height: 1.55; }

            .metric-strip {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                margin: 4.5rem 0;
                border: 1px solid var(--line);
                border-radius: 1rem;
                background: white;
                box-shadow: 0 12px 30px rgba(26, 68, 54, .05);
            }

            .metric-item { padding: 1.25rem 1.4rem; text-align: center; }
            .metric-item + .metric-item { border-left: 1px solid var(--line); }
            .metric-value { color: var(--ink); font-size: 1.25rem; font-weight: 770; }
            .metric-label { margin-top: .2rem; color: var(--muted); font-size: .72rem; }

            .step-card {
                height: 100%;
                padding: 1.15rem;
                border-top: 2px solid #b9ddcc;
                background: transparent;
            }

            .step-number { color: var(--brand); font-size: .7rem; font-weight: 800; letter-spacing: .08em; }
            .step-title { margin: .6rem 0 .35rem; color: var(--ink); font-weight: 730; }
            .step-copy { margin: 0; font-size: .82rem; line-height: 1.55; }

            .disclaimer {
                margin-top: 4rem;
                padding-top: 1.25rem;
                border-top: 1px solid var(--line);
                color: #74827e;
                font-size: .72rem;
                line-height: 1.55;
                text-align: center;
            }

            @media (max-width: 760px) {
                .block-container { padding-top: 1.25rem; }
                .brand-row { margin-bottom: 2.8rem; }
                .project-tag { display: none; }
                .hero { margin-bottom: 2.4rem; }
                .hero h1 { font-size: 2.65rem; }
                .metric-strip { grid-template-columns: 1fr; margin: 3rem 0; }
                .metric-item + .metric-item { border-left: 0; border-top: 1px solid var(--line); }
                .result-top { flex-direction: column; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(weights_path=WEIGHTS_PATH, labels_path=LABELS_PATH)


def get_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def read_image(file) -> Image.Image | None:
    if file is None:
        return None
    try:
        return Image.open(file).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Please try a JPG or PNG.")
        return None


def image_signature(image: Image.Image | None) -> str | None:
    if image is None:
        return None
    digest = hashlib.blake2b(image.tobytes(), digest_size=8)
    digest.update(str(image.size).encode("ascii"))
    return digest.hexdigest()


def render_result(result: dict) -> None:
    label = result["label"]
    details = CLASS_DETAILS[label]
    confidence = result["confidence"]
    needs_review = result["needs_review"]
    status_class = "status-review" if needs_review else "status-ready"
    status_text = "Review recommended" if needs_review else "High confidence"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-top">
                <div class="result-identity">
                    <div class="class-icon" style="background:{details['color']}">{details['icon']}</div>
                    <div>
                        <p class="class-name">{label}</p>
                        <p class="class-description">{details['description']}</p>
                    </div>
                </div>
                <span class="status-pill {status_class}">{status_text}</span>
            </div>
            <div class="confidence-row">
                <span class="confidence-label">Model confidence</span>
                <span class="confidence-value">{confidence:.0%}</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{confidence * 100:.1f}%"></div>
            </div>
            <div class="guidance">
                <div class="guidance-icon">✓</div>
                <div>
                    <strong>{details['bin']}</strong>
                    <p>{result['instruction']}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Prediction details")
    for candidate in result["top"]:
        candidate_label = candidate["label"].title()
        candidate_confidence = candidate["confidence"]
        left, right = st.columns([4, 1])
        left.progress(candidate_confidence, text=candidate_label)
        right.markdown(f"**{candidate_confidence:.1%}**")

    if needs_review:
        st.warning(
            "The top predictions are close or belong to a commonly confused pair. "
            "Try a clearer photo, or select the correct category below."
        )
        choices = [item["label"] for item in result["top"]]
        manual = st.selectbox(
            "Manual category",
            choices,
            format_func=str.title,
            help="Use your judgment when the model is uncertain.",
        )
        manual_details = CLASS_DETAILS[manual]
        st.info(f"**{manual_details['bin']}:** {INSTRUCTIONS[manual]}")


inject_styles()
metrics = get_metrics()

st.markdown(
    """
    <div class="brand-row">
        <div class="wordmark"><span class="brand-mark">↻</span> ReSort</div>
        <span class="project-tag">AI for responsible consumption · SDG 12</span>
    </div>
    <section class="hero">
        <span class="eyebrow">● Computer vision for better sorting</span>
        <h1>Know where your <span>waste belongs.</span></h1>
        <p>
            Upload or photograph one waste item. ReSort identifies its material
            and gives you a clear next step for responsible disposal.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Classify an item</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Add a clear, well-lit photo</div>', unsafe_allow_html=True)

input_column, result_column = st.columns([1, 1], gap="large")
image: Image.Image | None = None

with input_column:
    input_source = st.segmented_control(
        "Image source",
        ["Upload a photo", "Use camera"],
        default="Upload a photo",
        label_visibility="collapsed",
    )

    if input_source == "Upload a photo":
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            help="JPG, PNG, or WebP. For best results, show one item against a simple background.",
        )
        image = read_image(uploaded_file)
    else:
        camera_file = st.camera_input(
            "Take a photo",
            label_visibility="collapsed",
            help="Center one item in the frame and use even lighting.",
        )
        image = read_image(camera_file)

    current_signature = image_signature(image)
    if st.session_state.get("image_signature") != current_signature:
        st.session_state.image_signature = current_signature
        st.session_state.pop("prediction", None)

    if image is not None:
        st.image(image, caption="Selected image", use_container_width=True)
        classify = st.button(
            "Classify this item",
            type="primary",
            use_container_width=True,
            icon=":material/search:",
        )
        if classify:
            if not WEIGHTS_PATH.exists():
                st.error(
                    "The trained model is not available. Add `models/model.pth` "
                    "or follow the training instructions in the README."
                )
            else:
                with st.spinner("Analyzing material…"):
                    model, labels, device = get_model()
                    config = PredictConfig(
                        confidence_threshold=0.60,
                        margin_threshold=0.15,
                        topk=3,
                    )
                    st.session_state.prediction = predict_pil_ui(
                        image, model, labels, device, config
                    )
                    st.session_state.prediction_image = image.copy()

with result_column:
    prediction = st.session_state.get("prediction")
    if prediction is None:
        st.markdown(
            """
            <div class="empty-state">
                <div>
                    <div class="empty-icon">⌁</div>
                    <strong>Your result will appear here</strong>
                    <p>Use one item per photo, keep it centered, and avoid busy backgrounds.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        render_result(prediction)

validation_accuracy = metrics.get("best_val_acc", 0.834)
st.markdown(
    f"""
    <div class="metric-strip">
        <div class="metric-item">
            <div class="metric-value">6</div>
            <div class="metric-label">material categories</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{validation_accuracy:.1%}</div>
            <div class="metric-label">best validation accuracy</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">ResNet-18</div>
            <div class="metric-label">transfer-learning model</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">From photo to disposal guidance</div>', unsafe_allow_html=True)

steps = [
    (
        "01 · CAPTURE",
        "Show one item",
        "Photograph a single waste item in good light against a simple background.",
    ),
    (
        "02 · CLASSIFY",
        "Analyze the material",
        "A fine-tuned ResNet-18 compares visual features across six TrashNet categories.",
    ),
    (
        "03 · ACT",
        "Sort with confidence",
        "Follow the disposal guidance, or review alternatives when confidence is low.",
    ),
]
step_columns = st.columns(3, gap="large")
for column, (number, title, copy) in zip(step_columns, steps):
    with column:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-number">{number}</div>
                <div class="step-title">{title}</div>
                <p class="step-copy">{copy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="disclaimer">
        ReSort is an educational portfolio project trained on the TrashNet dataset.
        Recycling rules vary by location; always follow your local waste authority's guidance.
    </div>
    """,
    unsafe_allow_html=True,
)
