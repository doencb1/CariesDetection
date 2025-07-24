import os
import json
import torch
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

def get_model(num_classes=3):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3', 'pool'], 
        output_size=7, 
        sampling_ratio=2
    )
    return FasterRCNN(
        backbone, 
        num_classes=num_classes, 
        rpn_anchor_generator=anchor_generator, 
        box_roi_pool=roi_pooler
    )

MODEL_PATH = "faster_rcnn_caries_final_boosted_label2.pth"
TEST_FOLDER = r"C:\63HTTT1_2151163669_TruongQuocBao\Source_Code\test"
SCORE_THRESHOLD = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Duyệt toàn bộ ảnh trong thư mục test
image_files = [f for f in os.listdir(TEST_FOLDER) if f.endswith(".jpg")]

TP, FP, FN, TN = 0, 0, 0, 0

for filename in tqdm(image_files, desc="🔍 Đang đánh giá toàn bộ ảnh"):
    image_path = os.path.join(TEST_FOLDER, filename)
    json_path = image_path.replace(".jpg", ".json")
    if not os.path.exists(json_path):
        TN += 1
        continue

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    gt_labels = [int(shape['label']) for shape in data['shapes'] if shape['label'] in ['1', '2']]
    gt_has_target = len(gt_labels) > 0

    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0]

    pred_labels = [
        int(label) for label, score in zip(output['labels'].cpu().numpy(), output['scores'].cpu().numpy())
        if score >= SCORE_THRESHOLD and label in [1, 2]
    ]
    pred_has_target = len(pred_labels) > 0

    if gt_has_target and pred_has_target:
        TP += 1
    elif not gt_has_target and pred_has_target:
        FP += 1
    elif gt_has_target and not pred_has_target:
        FN += 1
    elif not gt_has_target and not pred_has_target:
        TN += 1


accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else float('nan')
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')

# In kết quả
print("\n📊 Đánh giá mô hình trên toàn bộ folder:")
print(f"✅ Accuracy       : {accuracy:.2f}")
print(f"🎯 Sensitivity(SE): {sensitivity:.2f}")
print(f"🛡️ Specificity    : {specificity if not (specificity != specificity) else 'Không xác định'}")
