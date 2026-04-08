import json
import os
from PIL import Image

class SlakeDataset:
    def __init__(self, json_path, img_dir):
        with open(json_path, 'r', encoding='utf-8') as f:
            # 영어 질문만 필터링
            self.data = [d for d in json.load(f) if d['q_lang'] == 'en']
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['img_name'])
        
        return {
            "img_id": item['img_id'],
            "image": Image.open(img_path).convert('RGB'),
            "question": item['question'],
            "answer": item['answer'],
            "q_type": item['q_type'],      # 'Location', 'Organ' 등 분석용
            "modality": item['modality']    # 'CT', 'MRI' 등 분석용
        }