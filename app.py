"""Streamlit demo for the DR uncertainty-aware classifier.

Upload a retinal fundus image; the model reports a DR severity grade with
MC Dropout confidence and predictive entropy, and defers to a specialist
recommendation when entropy crosses a data-driven "uncertain" threshold
instead of silently presenting a prediction that's more likely to be wrong.

Run with: streamlit run app.py
"""

import streamlit as st
import torch
from PIL import Image

from src.data_loader import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from src.gradcam import GradCAM, find_target_layer, overlay_heatmap
from src.model import build_model, predict_with_uncertainty

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
CHECKPOINT_PATH = "results/checkpoints/best_model.pt"
IMG_SIZE = 224
N_MC_SAMPLES = 30

# 75th percentile of mc_entropy over the 549-image validation set
# (results/metrics/per_image_predictions.csv, real run) -- roughly the top
# quartile of most-uncertain predictions.
ENTROPY_THRESHOLD = 0.83


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def preprocess(image: Image.Image):
    """Same resize+normalize pipeline as the validation split
    (src/data_loader.py's get_transforms), plus a denormalized display copy
    for the Grad-CAM overlay."""
    tensor = get_transforms("val", IMG_SIZE)(image.convert("RGB"))

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    display = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    return tensor, display


def main():
    st.set_page_config(page_title="DR Severity Grading Demo", page_icon="🩺")
    st.title("DR Severity Grading — Uncertainty-Aware Demo")
    st.warning(
        "**Research/educational demo only — not a diagnostic tool.** "
        "This model was trained on the public APTOS 2019 dataset for a portfolio "
        "project and must not be used to make or support real clinical decisions."
    )

    try:
        model, device = load_model()
    except FileNotFoundError:
        st.error(
            f"No checkpoint found at `{CHECKPOINT_PATH}`. Train a model first with "
            "`python -m src.train`, or place a checkpoint at that path."
        )
        st.stop()

    uploaded = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG or PNG retinal image to get a graded prediction.")
        return

    image = Image.open(uploaded)
    tensor, display_image = preprocess(image)

    with st.spinner(f"Running {N_MC_SAMPLES} MC Dropout samples..."):
        mean_probs, entropy = predict_with_uncertainty(model, tensor, n_samples=N_MC_SAMPLES)
    model.eval()  # predict_with_uncertainty leaves Dropout in train mode

    pred_class = int(mean_probs.argmax())
    confidence = float(mean_probs[pred_class])

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_image, caption="Uploaded image (preprocessed)", use_container_width=True)
    with col2:
        st.subheader(f"Prediction: {CLASS_NAMES[pred_class]}")
        st.metric("Confidence (MC Dropout)", f"{confidence:.1%}")
        st.metric("Predictive entropy", f"{entropy:.3f}")

        if entropy >= ENTROPY_THRESHOLD:
            st.error(
                "⚠️ **Low confidence — recommend specialist review**\n\n"
                f"Predictive entropy ({entropy:.3f}) is at or above the top-quartile "
                f"threshold ({ENTROPY_THRESHOLD}) seen on the validation set. This "
                "prediction is substantially more likely to be wrong than a typical one."
            )
        else:
            st.success("Entropy is within the model's typical range for this dataset.")

        with st.expander("Class probabilities"):
            for i, name in enumerate(CLASS_NAMES):
                st.write(f"{name}: {float(mean_probs[i]):.1%}")

    st.subheader("Grad-CAM")
    st.caption(
        "Highlights the image regions driving the prediction above, from a single "
        "deterministic (dropout-off) pass targeting the predicted class. Heatmaps "
        "are diffuse -- EfficientNet-B0's final feature map is 7x7 -- so treat this "
        "as a coarse attention check, not lesion-level localization."
    )
    target_layer = find_target_layer(model)
    gradcam = GradCAM(model, target_layer)
    image_batch = tensor.unsqueeze(0).to(device)
    image_batch.requires_grad_(True)
    cam = gradcam(image_batch, target_class=pred_class)
    gradcam.remove()
    model.eval()  # restore a clean deterministic state after the backward pass

    overlay = overlay_heatmap(display_image, cam)
    st.image(overlay, caption=f"Grad-CAM for predicted class: {CLASS_NAMES[pred_class]}", use_container_width=True)


if __name__ == "__main__":
    main()
