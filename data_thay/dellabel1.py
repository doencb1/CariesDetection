import os
import json

def remove_json_without_labels(json_folder):
    for filename in os.listdir(json_folder):
        if not filename.endswith('.json'):
            continue
        path = os.path.join(json_folder, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        if not shapes:
            os.remove(path)
            print(f"Đã xóa file không có nhãn: {filename}")

if __name__ == "__main__":
    folder_path = r"C:\thuctap_test\data_thay"  # Thay đường dẫn thư mục chứa file json của bạn
    remove_json_without_labels(folder_path)
