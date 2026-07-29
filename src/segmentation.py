"""Sky segmentation utilities using U-Net and OpenCV Canny-edge guidance."""
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from PIL import Image


def _median_filter_1d(values, kernel_size=7):
    """Apply a simple 1D median filter to stabilize skyline boundaries."""
    if kernel_size <= 1 or len(values) == 0:
        return values

    pad = kernel_size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    filtered = np.empty_like(values)

    for idx in range(len(values)):
        filtered[idx] = np.median(padded[idx:idx + kernel_size])

    return filtered


def _prepare_inference_image(orig_img, input_size=256):
    """Resize with aspect ratio preserved and reflective padding for inference."""
    width, height = orig_img.size
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized_img = orig_img.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    resized_np = np.array(resized_img)

    pad_left = (input_size - resized_width) // 2
    pad_right = input_size - resized_width - pad_left
    pad_top = (input_size - resized_height) // 2
    pad_bottom = input_size - resized_height - pad_top

    padded_np = cv2.copyMakeBorder(
        resized_np,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    return padded_np, (pad_left, pad_top, resized_width, resized_height)


def _has_edge_support(edges, row, col, row_radius=1, col_radius=2, min_count=3):
    """Return True when a candidate Canny edge has enough local support."""
    row_min = max(0, row - row_radius)
    row_max = min(edges.shape[0], row + row_radius + 1)
    col_min = max(0, col - col_radius)
    col_max = min(edges.shape[1], col + col_radius + 1)
    window = edges[row_min:row_max, col_min:col_max]
    return np.count_nonzero(window == 255) >= min_count


def refine_sky_mask_with_guidance(img_np, raw_unet_mask):
    """
    Refines raw U-Net sky mask using OpenCV Canny edges to snap 
    boundaries to exact ridgelines, removing snow/rock false positives.
    """
    H, W = raw_unet_mask.shape
    
    # 1. Connected components to keep only sky touching the top boundary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_unet_mask, connectivity=8)
    top_sky = np.zeros_like(raw_unet_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_TOP] == 0 and stats[i, cv2.CC_STAT_AREA] > 100:
            top_sky[labels == i] = 1

    # 2. Extract sharp structural lines (Canny edges)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    
    refined = np.zeros_like(top_sky)
    boundaries = np.full(W, -1, dtype=np.int32)

    for col in range(W):
        sky_rows = np.where(top_sky[:, col] == 1)[0]
        if len(sky_rows) == 0:
            continue
        boundary = sky_rows[-1]
        
        # Snap boundary to the closest Canny edge in a smaller local window
        search_min = max(0, boundary - 8)
        search_max = min(H - 1, boundary + 8)
        edge_rows = np.where(edges[search_min:search_max, col] == 255)[0]
        
        if len(edge_rows) > 0:
            closest_edge = edge_rows[np.argmin(np.abs(edge_rows - (boundary - search_min)))]
            candidate_boundary = search_min + closest_edge
            if _has_edge_support(edges, candidate_boundary, col):
                boundary = candidate_boundary

        boundaries[col] = boundary

    valid_boundaries = boundaries >= 0
    if np.any(valid_boundaries):
        boundaries[valid_boundaries] = _median_filter_1d(boundaries[valid_boundaries], kernel_size=7)

    for col in range(W):
        boundary = boundaries[col]
        if boundary >= 0:
            refined[:boundary + 1, col] = 1
        
    # Standard convention: Sky = 0 (Black), Terrain = 255 (White)
    return np.where(refined == 1, 0, 255).astype(np.uint8)


def load_segmentation_model(model_path, device):
    """Loads and returns the trained SMP U-Net model."""
    model = smp.Unet(
        encoder_name="tu-mobilenetv3_large_100",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    checkpoint = torch.load(model_path, map_location=device)
    clean_state = {k.replace("module.", "").replace("model.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(clean_state)
    return model.to(device).eval()


def segment_image(model, img_path, mask_output_path, device):
    """Processes a single image, refines it, and saves the resulting sky mask."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    orig_img = Image.open(img_path).convert("RGB")
    W, H = orig_img.size
    padded_img, crop_info = _prepare_inference_image(orig_img, input_size=256)
    pad_left, pad_top, resized_width, resized_height = crop_info
    
    # 1. Run U-Net inference
    tensor_img = transform(Image.fromarray(padded_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(tensor_img)).squeeze().cpu().numpy()
        
    # 2. Remove padding and resize probability map back to the original aspect ratio
    output_cropped = output[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width]
    prob_resized = cv2.resize(output_cropped.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    raw_mask = (prob_resized <= 0.5).astype(np.uint8)
    
    # 3. Apply edge guidance and save
    refined = refine_sky_mask_with_guidance(np.array(orig_img), raw_mask)
    Image.fromarray(refined).save(mask_output_path)


# =============================================================================
# Training utilities
# =============================================================================

import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class UnifiedDatasetAug(Dataset):
    """Albumentations-augmented dataset combining GeoPose3K and synthetic images."""

    def __init__(self, imgs, masks, is_train=True, train_transform=None):
        self.imgs = [str(p) for p in imgs]
        self.masks = [str(p) for p in masks]
        self.is_train = is_train

        if self.is_train:
            self.transform = train_transform if train_transform is not None else A.Compose([])
        else:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.imgs[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 10).astype(np.float32)
        # Resolve any aspect-ratio mismatches (e.g. GeoPose3K pinhole crops)
        h, w, _ = img.shape
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        aug = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].unsqueeze(0)


def find_photo_path(folder_path):
    """Return the first photo file found in a GeoPose3K sample folder."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = os.path.join(folder_path, f"photo{ext}")
        if os.path.exists(p):
            return p
    return None


def load_geopose_split(split_file, base_dir):
    """Load image/mask path lists from a GeoPose3K split text file."""
    images, masks = [], []
    split_file = Path(split_file)
    base_dir = Path(base_dir)
    if not os.path.exists(split_file):
        print(f"Warning: Split file {split_file} not found.")
        return images, masks
    publish_dir = base_dir / "geoPose3K_final_publish"
    if not publish_dir.exists():
        print(f"Warning: Publish directory {publish_dir} not found.")
        return images, masks
    # Single listing — 1000x faster than sequential os.path.exists calls
    existing_folders = set(os.listdir(publish_dir))
    with open(split_file) as f:
        folder_names = [line.strip().strip("/") for line in f if line.strip()]
    for folder in folder_names:
        if folder not in existing_folders:
            continue
        folder_path = publish_dir / folder
        img_p = folder_path / "photo.jpg"
        if not img_p.exists():
            img_p = find_photo_path(folder_path)
        mask_p = folder_path / "pinhole" / "labels_crop.png"
        if img_p and mask_p.exists():
            images.append(str(img_p))
            masks.append(str(mask_p))
    return images, masks


def build_training_loaders(geopose_dir, syn_img_dir, syn_mask_dir, batch_size=8, train_transform=None):
    """Build combined GeoPose3K + synthetic train/val DataLoaders."""
    train_images, train_masks = [], []
    val_images,   val_masks   = [], []
    geopose_dir = Path(geopose_dir)

    print("Loading GeoPose3K dataset files...")
    tr_img, tr_mask = load_geopose_split(
        geopose_dir / "splits" / "geoPose3K_final_train.txt", geopose_dir)
    va_img, va_mask = load_geopose_split(
        geopose_dir / "splits" / "geoPose3K_final_val.txt", geopose_dir)
    train_images.extend(tr_img);  train_masks.extend(tr_mask)
    val_images.extend(va_img);    val_masks.extend(va_mask)

    print("Checking for synthetic dataset...")
    syn_img_dir, syn_mask_dir = str(syn_img_dir), str(syn_mask_dir)
    if os.path.exists(syn_img_dir):
        all_syn = sorted(f for f in os.listdir(syn_img_dir) if f.lower().endswith(".png"))
        n = len(all_syn)
        if n > 0:
            print(f"  Found {n} synthetic samples.")
            split = int(n * 0.8)
            for f in all_syn[:split]:
                train_images.append(os.path.join(syn_img_dir, f))
                train_masks.append(os.path.join(syn_mask_dir, f))
            for f in all_syn[split:]:
                val_images.append(os.path.join(syn_img_dir, f))
                val_masks.append(os.path.join(syn_mask_dir, f))
        else:
            print("  Warning: Synthetic directory is empty.")
    else:
        print("  Warning: Synthetic directory not found. Training on GeoPose3K only.")

    train_loader = DataLoader(
        UnifiedDatasetAug(train_images, train_masks, is_train=True, train_transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(
        UnifiedDatasetAug(val_images, val_masks, is_train=False),
        batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"  Train samples: {len(train_images)} | Val samples: {len(val_images)}")
    return train_loader, val_loader


def build_sky_model(device):
    """Instantiate the MobileNetV3-backed U-Net for training."""
    import segmentation_models_pytorch as smp
    return smp.Unet(
        encoder_name="tu-mobilenetv3_large_100",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)


def compute_iou(pred_logits, true_masks, threshold=0.5):
    """Batch IoU from raw logits."""
    import torch
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (preds * true_masks).sum()
    union = preds.sum() + true_masks.sum() - intersection
    return (intersection / (union + 1e-6)).item()


def bce_dice_loss(pred, target):
    """Combined BCE + soft Dice loss."""
    import torch
    bce = nn.BCEWithLogitsLoss()(pred, target)
    smooth = 1e-6
    probs = torch.sigmoid(pred)
    intersection = (probs * target).sum()
    dice = 1.0 - (2.0 * intersection + smooth) / (probs.sum() + target.sum() + smooth)
    return bce + dice


def train_sky_model(model, train_loader, val_loader, device, save_path, epochs=15, lr=2e-4):
    """Full training loop. Returns (train_losses, val_losses, train_ious, val_ious, lrs)."""
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, train_ious = [], []
    val_losses,   val_ious   = [], []
    lrs = []
    best_val_iou = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = total_iou = 0.0
        lrs.append(optimizer.param_groups[0]["lr"])

        for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = bce_dice_loss(out, msks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_iou  += compute_iou(out, msks)

        train_losses.append(total_loss / len(train_loader))
        train_ious.append(total_iou  / len(train_loader))
        scheduler.step()

        model.eval()
        v_loss = v_iou = 0.0
        with torch.no_grad():
            for imgs, msks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                imgs, msks = imgs.to(device), msks.to(device)
                out = model(imgs)
                v_loss += bce_dice_loss(out, msks).item()
                v_iou  += compute_iou(out, msks)

        val_losses.append(v_loss / len(val_loader))
        val_ious.append(v_iou  / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs}  "
              f"Train Loss: {train_losses[-1]:.4f}  Train IoU: {train_ious[-1]*100:.2f}%  |  "
              f"Val Loss: {val_losses[-1]:.4f}  Val IoU: {val_ious[-1]*100:.2f}%")

        if val_ious[-1] > best_val_iou:
            best_val_iou = val_ious[-1]
            torch.save(model.state_dict(), str(save_path))
            print(f"  => Checkpoint saved (Val IoU: {best_val_iou*100:.2f}%)")

    print(f"\nTraining complete. Best Val IoU: {best_val_iou*100:.2f}%")
    return train_losses, val_losses, train_ious, val_ious, lrs


def show_augmentation_samples(img_paths, mask_paths, aug, n=6):
    indices = random.sample(range(len(img_paths)), min(n, len(img_paths)))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    
    # Position the suptitle higher to prevent overlap
    fig.suptitle("Data Augmentation Samples", fontsize=14, fontweight="bold", y=0.99)
    
    for row, idx in enumerate(indices):
        img  = cv2.cvtColor(cv2.imread(str(img_paths[idx])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 10).astype(np.uint8) * 255
        
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        result = aug(image=img, mask=mask)
        img_show = result["image"]
        mask_show = result["mask"]

        if torch.is_tensor(img_show):
            img_show = img_show.permute(1, 2, 0).cpu().numpy()
            img_show = (img_show * IMAGENET_STD + IMAGENET_MEAN) * 255
            img_show = np.clip(img_show, 0, 255).astype(np.uint8)

        if torch.is_tensor(mask_show):
            mask_show = mask_show.squeeze().cpu().numpy() * 255

        axes[row, 0].imshow(cv2.resize(img, (256, 256)));   axes[row, 0].set_title("Original");         axes[row, 0].axis("off")
        axes[row, 1].imshow(img_show);                      axes[row, 1].set_title("Augmented");        axes[row, 1].axis("off")
        axes[row, 2].imshow(mask_show, cmap="gray");        axes[row, 2].set_title("Augmented Mask");   axes[row, 2].axis("off")
        
    plt.tight_layout()
    # Add spacing at the top of the grid to clear the suptitle
    plt.subplots_adjust(top=0.96)
    plt.show()
        
def plot_training_curves(train_losses, val_losses, train_ious, val_ious, lrs):
    """Three-panel plot: loss curves, IoU curves, LR decay."""
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color="crimson",    lw=2)
    plt.plot(epochs_range, val_losses,   label="Val Loss",   color="royalblue",  lw=2, linestyle="--")
    plt.title("BCE + Dice Loss Curve"); plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, [v * 100 for v in train_ious], label="Train IoU", color="crimson",   lw=2)
    plt.plot(epochs_range, [v * 100 for v in val_ious],   label="Val IoU",   color="royalblue", lw=2, linestyle="--")
    plt.title("IoU Curve"); plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, lrs, label="Learning Rate", color="forestgreen", lw=2)
    plt.title("Cosine Annealing LR Decay"); plt.xlabel("Epochs"); plt.ylabel("Learning Rate")
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()