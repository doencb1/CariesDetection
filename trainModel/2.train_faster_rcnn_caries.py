import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from collections import Counter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F_nn.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class DentalCariesDataset(Dataset):
    def __init__(self, image_dir, json_dir, transforms=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transforms = transforms
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith('.jpg') and os.path.exists(os.path.join(json_dir, f.replace('.jpg', '.json')))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        json_path = os.path.join(self.json_dir, img_filename.replace('.jpg', '.json'))

        image = np.array(Image.open(img_path).convert("RGB"))
        height, width, _ = image.shape

        boxes = []
        labels = []

        with open(json_path, 'r') as f:
            data = json.load(f)
            for shape in data['shapes']:
                label_str = shape['label']
                if label_str not in ['1', '2']:
                    continue
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]
                x_min, y_min = min(x1, x2), min(y1, y2)
                x_max, y_max = max(x1, x2), max(y1, y2)
                x_min = max(0, min(x_min, width - 1))
                x_max = max(0, min(x_max, width - 1))
                y_min = max(0, min(y_min, height - 1))
                y_max = max(0, min(y_max, height - 1))
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(label_str))

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_files))

        boxes = np.array(boxes)
        labels = np.array(labels)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = F.to_tensor(image)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target

def compute_sample_weights(dataset):
    class_counts = Counter()
    weights = []

    for i in range(len(dataset)):
        _, target = dataset[i]
        labels = target['labels'].tolist()
        if not labels:
            weights.append(0.0)
            continue
        main_label = max(set(labels), key=labels.count)
        class_counts[main_label] += 1

    class_weights = {1: 1.0, 2: 2.0}

    for i in range(len(dataset)):
        _, target = dataset[i]
        labels = target['labels'].tolist()
        if not labels:
            weights.append(0.0)
        else:
            main_label = max(set(labels), key=labels.count)
            weights.append(class_weights.get(main_label, 1.0))

    return weights

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.MotionBlur(p=0.2),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=1, min_visibility=0.3))

def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_model(num_classes=3, pretrained_path=None, device='cpu'):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("roi_heads.box_predictor")
        }
        model.load_state_dict(filtered_state_dict, strict=False)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def custom_loss_classifier(cls_score, labels):
        alpha = torch.tensor([1.0, 1.0, 4.0]).to(cls_score.device)
        loss = FocalLoss(alpha=alpha, gamma=2.0)
        return loss(cls_score, labels)

    model.roi_heads._loss_classifier = custom_loss_classifier

    return model


def train_model(model, train_loader, val_loader, device, num_epochs=60):
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.003, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()

        scheduler.step()

        model.train()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

    return model

if __name__ == "__main__":
    image_folder = "C:/thuctap_test/data_thay"
    json_folder = image_folder

    dataset_full = DentalCariesDataset(image_folder, json_folder, transforms=get_train_transform())
    val_percent = 0.3
    val_size = int(len(dataset_full) * val_percent)
    train_size = len(dataset_full) - val_size
    train_dataset, val_dataset = random_split(dataset_full, [train_size, val_size])
    val_dataset.dataset.transforms = get_val_transform()

    weights = compute_sample_weights(train_dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=2, sampler=sampler, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=3, pretrained_path="faster_rcnn_caries_label2only.pth", device=device)

    model = train_model(model, train_loader, val_loader, device)
    torch.save(model.state_dict(), "faster_rcnn_caries_focal_sampler_longer.pth")