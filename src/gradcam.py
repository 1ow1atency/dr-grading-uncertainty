"""Grad-CAM visualization for the DR severity classifier.

Generates class-activation heatmaps for 8 representative validation
images (high-confidence correct/incorrect, high-uncertainty
correct/incorrect, by MC Dropout confidence/entropy), using a single
deterministic (dropout-off) forward/backward pass targeting each image's
MC Dropout predicted class.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from .data_loader import IMAGENET_MEAN, IMAGENET_STD, get_transforms
    from .model import build_model
except ImportError:
    from data_loader import IMAGENET_MEAN, IMAGENET_STD, get_transforms
    from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for the DR severity classifier.")
    parser.add_argument("--per_image_csv", type=str, default="results/metrics/per_image_predictions.csv")
    parser.add_argument("--img_dir", type=str, default="data/raw/train_images")
    parser.add_argument("--img_ext", type=str, default=".png")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--checkpoint_path", type=str, default="results/checkpoints/best_model.pt")
    parser.add_argument("--n_per_group", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/plots/gradcam")
    return parser.parse_args()


def find_target_layer(model):
    """Locates the backbone's final convolutional block -- the one whose
    output feeds avgpool -> classifier.

    torchvision's EfficientNet-B0 exposes its conv trunk as
    model.features, a 9-stage nn.Sequential whose last child is a
    Conv2dNormActivation block (1x1 Conv2d -> BatchNorm2d -> SiLU,
    confirmed by inspection). That's the natural Grad-CAM hook point:
    the last spatial feature map before global pooling, so we take
    model.features[-1] rather than a hardcoded stage index.
    """
    return model.features[-1]


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017): a target class's activation map
    is ReLU(sum_k alpha_k * A_k), where A_k is the target layer's k-th
    output channel and alpha_k is the global-average-pooled gradient of
    the target class's logit w.r.t. A_k.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, image, target_class):
        """image: (1, C, H, W) tensor. Returns a 2D numpy CAM (the target
        layer's spatial resolution), normalized to [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        peak = cam.max()
        if peak > 1e-12:
            cam /= peak
        return cam


def load_image(img_path, img_size):
    """Loads an image with the same resize+normalize pipeline used for
    validation, plus a denormalized display copy (float RGB in [0, 1])
    for the Grad-CAM overlay."""
    image = Image.open(img_path).convert("RGB")
    tensor = get_transforms("val", img_size)(image)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    display = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    return tensor, display


def overlay_heatmap(display_image, cam, alpha=0.45):
    """Upsamples the (small) CAM to the display image's resolution,
    colors it with the 'jet' colormap, and alpha-blends it over the
    original image."""
    h, w = display_image.shape[:2]
    cam_tensor = torch.tensor(cam, dtype=torch.float32).view(1, 1, *cam.shape)
    cam_resized = F.interpolate(cam_tensor, size=(h, w), mode="bilinear", align_corners=False)
    cam_resized = cam_resized.squeeze().numpy()

    heatmap = matplotlib.colormaps["jet"](cam_resized)[..., :3]
    overlay = (1 - alpha) * display_image + alpha * heatmap
    return np.clip(overlay, 0, 1)


def select_representative_images(df: pd.DataFrame, n_per_group: int = 2):
    """Selects images for four groups -- high-confidence correct/incorrect
    and high-uncertainty correct/incorrect -- ranked by MC Dropout
    confidence / predictive entropy respectively (correctness per
    mc_correct). Groups are disjoint: an image already picked for one
    group is excluded from the rest.

    Returns a list of (group_label, row) pairs.
    """
    used_ids = set()
    selections = []

    def pick(pool, sort_col, label):
        pool = pool[~pool["id_code"].isin(used_ids)].sort_values(sort_col, ascending=False)
        picked = pool.head(n_per_group)
        used_ids.update(picked["id_code"])
        if len(picked) < n_per_group:
            print(f"Warning: only {len(picked)}/{n_per_group} images available for '{label}'")
        return [(label, row) for _, row in picked.iterrows()]

    correct = df[df["mc_correct"] == 1]
    incorrect = df[df["mc_correct"] == 0]

    selections += pick(correct, "mc_confidence", "high-confidence correct")
    selections += pick(incorrect, "mc_confidence", "high-confidence incorrect")
    selections += pick(correct, "mc_entropy", "high-uncertainty correct")
    selections += pick(incorrect, "mc_entropy", "high-uncertainty incorrect")

    return selections


def save_gradcam_figure(display_image, overlay, row, group_label, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), dpi=150)

    axes[0].imshow(display_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    caption = (
        f"{group_label}  |  True: {int(row['true_label'])}  Pred: {int(row['mc_predicted_label'])}  "
        f"Confidence: {row['mc_confidence']:.3f}  Entropy: {row['mc_entropy']:.3f}"
    )
    fig.suptitle(caption, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.per_image_csv)

    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # dropout OFF: one clean, deterministic gradient, not MC Dropout's stochastic mode

    target_layer = find_target_layer(model)
    print(f"Grad-CAM target layer: {target_layer.__class__.__name__} (model.features[-1])")
    gradcam = GradCAM(model, target_layer)

    selections = select_representative_images(df, args.n_per_group)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, (group_label, row) in enumerate(selections, start=1):
        img_path = os.path.join(args.img_dir, f"{row['id_code']}{args.img_ext}")
        tensor, display_image = load_image(img_path, args.img_size)

        image = tensor.unsqueeze(0).to(device)
        image.requires_grad_(True)
        cam = gradcam(image, target_class=int(row["mc_predicted_label"]))
        overlay = overlay_heatmap(display_image, cam)

        safe_label = group_label.replace(" ", "_").replace("-", "_")
        out_path = os.path.join(args.output_dir, f"{i:02d}_{safe_label}_{row['id_code']}.png")
        save_gradcam_figure(display_image, overlay, row, group_label, out_path)
        print(f"[{i}/{len(selections)}] {group_label}: {row['id_code']} -> {out_path}")

    gradcam.remove()
    print(f"Saved {len(selections)} Grad-CAM visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()
