import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

# Define dataset class
class CocoDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        self.images = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.annotations = {}
        self.cat_dog_classes = {17, 18}  # COCO category IDs for cat and dog
        for ann in self.coco_data['annotations']:
            if ann['category_id'] in self.cat_dog_classes:
                img_id = ann['image_id']
                if img_id not in self.annotations:
                    self.annotations[img_id] = {'boxes': [], 'labels': []}
                x, y, w, h = ann['bbox']
                if w > 0 and h > 0:  # Ensure bounding boxes have positive width and height
                    self.annotations[img_id]['boxes'].append([x, y, x + w, y + h])
                    self.annotations[img_id]['labels'].append(ann['category_id'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = list(self.images.keys())[idx]
        img_path = os.path.join(self.img_dir, self.images[img_id])
        image = Image.open(img_path).convert("RGB")
        target = self.annotations.get(img_id, {'boxes': [], 'labels': []})
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32).clone().detach()
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64).clone().detach()

        if self.transforms:
            image = self.transforms(image)

        return image, target, img_id
