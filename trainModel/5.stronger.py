import os
import json
import numpy as np
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DentalCariesDataset(Dataset):
    def __init__(self, image_dir, json_dir, transforms=None):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transforms = transforms
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith(".jpg") and os.path.exists(os.path.join(json_dir, f.replace(".jpg", ".json")))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        json_path = os.path.join(self.json_dir, img_filename.replace(".jpg", ".json"))

        image = np.array(Image.open(img_path).convert("RGB"))
        height, width, _ = image.shape

        boxes, labels = [], []
        with open(json_path) as f:
            data = json.load(f)
            for shape in data['shapes']:
                label = shape['label']
                if label not in ['1', '2']: continue
                x1, y1 = shape['points'][0]
                x2, y2 = shape['points'][1]
                x_min, y_min = min(x1,x2), min(y1,y2)
                x_max, y_max = max(x1,x2), max(y1,y2)
                x_min = max(0, min(x_min, width - 1))
                x_max = max(0, min(x_max, width - 1))
                y_min = max(0, min(y_min, height - 1))
                y_max = max(0, min(y_max, height - 1))
                if x_max <= x_min or y_max <= y_min: continue
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(label))

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.image_files))

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = F.to_tensor(image)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, target

def get_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=20, p=0.8),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))

def compute_sample_weights(dataset):
    class_weights = {1: 1.0, 2: 6.0}
    weights = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        labels = target['labels'].tolist()
        if not labels:
            weights.append(0.0)
        else:
            main_label = max(set(labels), key=labels.count)
            weights.append(class_weights.get(main_label, 1.0))
    return weights

def get_model(num_classes=3):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0','1','2','3'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

if __name__ == '__main__':
    image_folder = "C:/thuctap_test/data_thay"
    model_path = "faster_rcnn_caries_finetuned_label2_full_strong_sampler.pth"

    dataset = DentalCariesDataset(image_folder, image_folder, transforms=get_transform())
    weights = compute_sample_weights(dataset)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model.train()
    for epoch in range(15):
        total_loss = 0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()
        print(f"[Epoch {epoch+1}/15] Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "faster_rcnn_caries_final_boosted_label2.pth")
