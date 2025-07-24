import os
import io
import sqlite3
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify
from PIL import Image, ImageDraw, ImageFont
import uuid
import torch
import time
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, nms

app = Flask(__name__)
app.secret_key = "SECRET_KEY"
UPLOAD_FOLDER = os.path.join("static", "uploads")
DB_PATH = "history.db"
MODEL_PATH = "faster_rcnn_caries_final_boosted_label2.pth"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def load_font(size=75):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnosis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            gender TEXT,
            location TEXT,
            result TEXT,
            timestamp TEXT,
            image TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_model(num_classes=3):
    backbone = resnet_fpn_backbone('resnet50', pretrained=False)
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', 'pool'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def apply_nms(boxes, scores, iou_threshold=0.3):
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep = nms(boxes_tensor, scores_tensor, iou_threshold)
    return keep.numpy()

def process_image(img_bytes, threshold=0.6):
    start_time = time.time()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    boxes = outputs['boxes'].cpu().numpy()

    keep_indices = apply_nms(boxes, scores, iou_threshold=0.3)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    count = {1: 0, 2: 0}
    notes = []
    draw = ImageDraw.Draw(image)
    font = load_font()
    m = 0
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score >= threshold and label in [1, 2]:
            count[label] += 1
            x1, y1, x2, y2 = box
            color = "red" if label == 1 else "yellow"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 3, y1 + 3), f"Box {m + 1}", fill="blue", font=font)
            notes.append((int(label), round(float(score), 2)))
            m += 1

    elapsed_time = round(time.time() - start_time, 2)
    filename = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path, format="JPEG")

    return filename, count, notes, elapsed_time

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        location = request.form['location']
        images = request.files.getlist('images')

        session['patient_info'] = {'name': name, 'gender': gender, 'location': location}
        saved_paths = []
        for img in images:
            if img and img.filename:
                filename = f"{uuid.uuid4().hex}.jpg"
                path = os.path.join(UPLOAD_FOLDER, filename)
                img.save(path)
                saved_paths.append(path)
        session['uploaded_paths'] = saved_paths
        return redirect(url_for('result'))

    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    paths = session.get('uploaded_paths', [])
    info = session.get('patient_info', {})
    session['results'] = []

    for path in paths:
        with open(path, "rb") as f:
            img_bytes = f.read()

        result_img, count, notes, elapsed_time = process_image(img_bytes)

        session['results'].append({
            'image': f'uploads/{result_img}',
            'count': {int(k): int(v) for k, v in count.items()},
            'notes': [(int(label), float(score)) for label, score in notes],
            'elapsed': float(elapsed_time)
        })

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO diagnosis (name, gender, location, result, timestamp, image)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            info.get('name'), info.get('gender'), info.get('location'),
            f"Sâu răng loại 2: {count[1]}, sâu răng loại 3: {count[2]}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result_img
        ))
        conn.commit()
        conn.close()

    return jsonify({'status': 'done'})

@app.route('/get_result_data')
def get_result_data():
    return jsonify(session.get('results', []))

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/session_info')
def session_info():
    return jsonify(session.get('patient_info', {}))


@app.route('/history')
def history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM diagnosis ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)

@app.route('/delete/<int:entry_id>')
def delete(entry_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT image FROM diagnosis WHERE id=?", (entry_id,))
    row = cursor.fetchone()
    if row:
        image_filename = row[0]
        image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        if os.path.exists(image_path):
            os.remove(image_path)

    cursor.execute("DELETE FROM diagnosis WHERE id=?", (entry_id,))
    conn.commit()
    conn.close()
    return redirect('/history')

if __name__ == '__main__':
    app.run(debug=True)
