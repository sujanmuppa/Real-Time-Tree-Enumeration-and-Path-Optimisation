import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

########################################
# Define Dice Loss and Compound Loss
########################################
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        intersection = (inputs * targets_onehot).sum(dim=(2,3))
        total = (inputs + targets_onehot).sum(dim=(2,3))
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()

def compound_loss(outputs, targets, ce_weight=1.0, dice_weight=2.0):
    logits = F.interpolate(outputs.logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
    ce = nn.CrossEntropyLoss()(logits, targets)
    dice = DiceLoss()(logits, targets)
    return ce_weight * ce + dice_weight * dice

########################################
# Dataset Class
########################################
class DeepGlobeDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, mapping=None):
        """
        Args:
            df: DataFrame with CSV info.
            root_dir: Root folder for images/masks.
            transform: Albumentations transform.
            mapping: Dictionary mapping original mask pixel values to contiguous indices.
        """
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.mapping = mapping

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sat_img_path = os.path.join(self.root_dir, str(row['sat_image_path']))
        mask_path = os.path.join(self.root_dir, str(row['mask_path']))
        
        image = cv2.imread(sat_img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {sat_img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if self.mapping is not None:
            mask_mapped = np.copy(mask)
            for old_val, new_val in self.mapping.items():
                mask_mapped[mask == old_val] = new_val
            mask = mask_mapped
        if mask.max() >= len(self.mapping):
            raise ValueError(f"Mask at index {idx} has max value {mask.max()}, expected less than {len(self.mapping)}")
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).long()
        return image, mask

########################################
# Transforms (Resize to 512x512)
########################################
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.2),
        ])
    else:
        return A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        ])

########################################
# Utility: Compute mIoU for a Batch
########################################
def compute_iou(outputs, masks, num_classes):
    preds = torch.argmax(outputs, dim=1)
    iou_total = 0.0
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        mask_cls = (masks == cls).float()
        intersection = (pred_cls * mask_cls).sum()
        union = pred_cls.sum() + mask_cls.sum() - intersection
        if union == 0:
            iou = torch.tensor(1.0)
        else:
            iou = intersection / union
        iou_total += iou
    return iou_total / num_classes

########################################
# Utility: Display a Sample Prediction
########################################
def show_prediction(model, dataset, device, idx=0):
    model.eval()
    image, mask = dataset[idx]
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
    upsampled_logits = F.interpolate(outputs.logits, size=mask.shape[-2:], mode='bilinear', align_corners=False)
    pred = torch.argmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
    image_np = image.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    mask_np = mask.cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(mask_np, cmap="jet", vmin=0, vmax=num_classes-1)
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap="jet", vmin=0, vmax=num_classes-1)
    axs[2].set_title("Prediction")
    for ax in axs:
        ax.axis("off")
    plt.show()

########################################
# Utility: Evaluate Model
########################################
def evaluate_model(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_batches = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images)
            upsampled_logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = compound_loss(outputs, masks)
            total_loss += loss.item()
            batch_iou = compute_iou(upsampled_logits, masks, num_classes)
            total_iou += batch_iou.item()
            total_batches += 1
    avg_loss = total_loss / total_batches
    avg_iou = total_iou / total_batches
    return avg_loss, avg_iou

########################################
# Main Script: Split Single CSV, Train & Evaluate
########################################
if __name__ == "__main__":
    # Set the root directory where your dataset is located and CSV file path.
    root_dir = ""  # update this to your dataset folder
    csv_file = os.path.join(root_dir, "metadata.csv")
    df = pd.read_csv(csv_file)

    # Use only the rows with split == 'train' since these have masks.
    df_train = df[df['split'] == 'train']

    # Split df_train into train, valid, and test sets (e.g., 70/15/15).
    train_val_df, test_df = train_test_split(df_train, test_size=0.15, random_state=42)
    train_df, valid_df = train_test_split(train_val_df, test_size=0.15/0.85, random_state=42)
    
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}, Test samples: {len(test_df)}")

    # Compute unique classes from the training set and create mapping.
    unique_classes = set()
    for x in train_df['mask_path']:
        mask = cv2.imread(os.path.join(root_dir, str(x)), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_classes.update(np.unique(mask).tolist())
    print("Unique classes found:", unique_classes)
    mapping = {old_val: new_val for new_val, old_val in enumerate(sorted(unique_classes))}
    print("Mapping dictionary:", mapping)
    num_classes = len(mapping)
    print("Number of classes:", num_classes)

    # Define transforms.
    train_transform = get_transforms(train=True)
    valid_transform = get_transforms(train=False)

    # Create Dataset objects.
    train_dataset = DeepGlobeDataset(train_df, root_dir=root_dir, transform=train_transform, mapping=mapping)
    valid_dataset = DeepGlobeDataset(valid_df, root_dir=root_dir, transform=valid_transform, mapping=mapping)
    test_dataset  = DeepGlobeDataset(test_df, root_dir=root_dir, transform=valid_transform, mapping=mapping)

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # (Optional) Check a batch.
    for batch_idx, (images, masks) in enumerate(train_loader):
        print("Batch idx:", batch_idx)
        print("Images shape:", images.shape)  # (4, 3, 512, 512)
        print("Masks shape:", masks.shape)      # (4, 512, 512)
        print("Unique values in a mask:", torch.unique(masks[0]))
        break

    # Setup device (change to 'cuda' if available).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the SegFormer model (e.g., SegFormer-B4) with ignore_mismatched_sizes.
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b4-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    print("Model parameter device:", next(model.parameters()).device)

    # Setup loss, optimizer, and scheduler.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    # Early stopping parameters.
    best_val_iou = 0.0
    patience = 0
    early_stop_patience = 10

    # Full training loop.
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", unit="batch")
        for batch_idx, (images, masks) in enumerate(train_pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            upsampled_logits = F.interpolate(outputs.logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = compound_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Training Loss: {avg_train_loss:.4f}")

        # Validation phase.
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        total_batches = 0
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", unit="batch")
        with torch.no_grad():
            for images_val, masks_val in valid_pbar:
                images_val, masks_val = images_val.to(device), masks_val.to(device)
                outputs_val = model(pixel_values=images_val)
                upsampled_logits_val = F.interpolate(outputs_val.logits, size=masks_val.shape[-2:], mode='bilinear', align_corners=False)
                loss_val = nn.CrossEntropyLoss()(upsampled_logits_val, masks_val)
                val_loss += loss_val.item()
                batch_iou = compute_iou(upsampled_logits_val, masks_val, num_classes)
                total_iou += batch_iou.item()
                total_batches += 1
                valid_pbar.set_postfix(loss=f"{loss_val.item():.4f}", IoU=f"{batch_iou.item():.4f}")
        avg_val_loss = val_loss / len(valid_loader)
        avg_val_iou = total_iou / total_batches
        print(f"Validation Loss: {avg_val_loss:.4f}, mIoU: {avg_val_iou:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            patience = 0
            torch.save(model.state_dict(), "GoodTrain.pth")
            print("Model improved. Saving model.")
        else:
            patience += 1
            print(f"No improvement in mIoU. Patience: {patience}/{early_stop_patience}")
            if patience >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # Evaluate on test set.
    print("Evaluating on test set...")
    test_loss, test_iou = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device, num_classes)
    print(f"Test Loss: {test_loss:.4f}, Test mIoU: {test_iou:.4f}")

    # Show sample predictions.
    print("Showing sample predictions from test set:")
    for i in range(3):
        show_prediction(model, test_dataset, device, idx=i)
