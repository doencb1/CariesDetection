
📖 Giới thiệu
**CariesDetection** là một dự án phát hiện sâu răng từ ảnh chụp miệng sử dụng mô hình học sâu **Faster R-CNN với ResNet50 + FPN**. Dữ liệu đầu vào là các ảnh gán nhãn từ công cụ LabelMe, với các nhãn cụ thể, trong đó nhãn **'2'** thường biểu hiện vùng sâu răng nghiêm trọng.  
Dự án trải qua nhiều giai đoạn huấn luyện và tinh chỉnh mô hình để cải thiện hiệu suất, đặc biệt tập trung vào nhãn ít xuất hiện (label 2).

## 🧠 Kiến trúc và luồng xử lý

### 🔹 Mô hình
- Sử dụng Faster R-CNN với backbone `resnet50` kết hợp FPN.
- Anchor generator và RoI Align đa mức độ.
- Tinh chỉnh đầu ra của mô hình để phù hợp với 3 lớp (label 0 là background).
- Áp dụng Focal Loss để xử lý mất cân bằng nhãn.
- Huấn luyện với WeightedRandomSampler để tăng trọng số ảnh chứa nhãn '2'.

### 🔹 Pipeline huấn luyện
| File | Vai trò |
|------|--------|
| `1.onlylabel2.py` | Huấn luyện ban đầu chỉ với label 2 |
| `2.train_faster_rcnn_caries.py` | Huấn luyện lại mô hình với Focal Loss + Sampler |
| `3.finetunelabel2.py` | Fine-tune phần đầu (head) mô hình với label 2 |
| `4.fineTuneAfterFineTuneLabel2.py` | Fine-tune toàn bộ mô hình thêm 20 epochs |
| `5.stronger.py` | Huấn luyện tăng cường thêm 15 epochs để đạt kết quả tốt hơn |

### 🔹 Web App (`app.py`)
- Giao diện Flask với upload nhiều ảnh.
- Tự động phát hiện, vẽ box lên ảnh và lưu ảnh kết quả.
- Hiển thị số lượng nhãn theo từng loại.
- Lưu lịch sử vào SQLite (`history.db`) kèm tên, giới tính, nơi ở, thời gian và ảnh.
- Trang `/history` để xem lại hoặc xoá kết quả cũ.

## 🚀 Cách sử dụng

### 1. Cài đặt
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng web
```bash
python app.py
```

Truy cập trình duyệt tại `http://localhost:5000`

## 📦 requirements.txt

```txt
torch==2.0.1
torchvision==0.15.2
albumentations==1.3.1
opencv-python==4.9.0.80
Pillow==10.3.0
Flask==2.2.5
numpy==1.24.4
```

## 🖼️ Dữ liệu

- Ảnh định dạng `.jpg`, nhãn từ LabelMe dạng `.json`.
- Các nhãn là `"1"`, `"2"` trong trường `"label"` của `"shapes"`.

## 📊 Kết quả đầu ra

- Box màu đỏ: sâu răng loại 1.
- Box màu vàng: sâu răng loại 2.
- Ghi chú từng box kèm độ tin cậy (`score`) được vẽ lên ảnh.
- Hiển thị tổng số lượng phát hiện cho từng loại.

## 🧪 Mở rộng & Cải tiến

- Có thể huấn luyện thêm với dữ liệu lớn hơn.
- Bổ sung các lớp khác như mảng bám, nứt, hoặc viêm nướu.
- Triển khai segmentation để phát hiện chính xác vùng tổn thương.

## 📁 Cấu trúc thư mục đề xuất

```
CariesDetection/
├── 1.onlylabel2.py
├── 2.train_faster_rcnn_caries.py
├── 3.finetunelabel2.py
├── 4.fineTuneAfterFineTuneLabel2.py
├── 5.stronger.py
├── app.py
├── static/uploads/               # Ảnh kết quả
├── templates/                    # index.html, result.html, history.html
├── history.db                    # SQLite lưu lịch sử
├── requirements.txt
└── README.md
```
