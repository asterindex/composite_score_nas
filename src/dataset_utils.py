"""
Утиліти для роботи з VisDrone датасетом
Конвертація анотацій, візуалізація, статистика
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# VisDrone класи
VISDRONE_CLASSES = {
    0: 'ignored',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor'
}


def parse_annotation(ann_path: Path) -> List[Dict]:
    """
    Парсинг анотації VisDrone
    
    Формат: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    """
    annotations = []
    
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                x, y, w, h = map(int, parts[:4])
                score = int(parts[4])
                category = int(parts[5])
                truncation = int(parts[6])
                occlusion = int(parts[7])
                
                if score > 0 and 1 <= category <= 10:  # Валідні об'єкти
                    annotations.append({
                        'bbox': [x, y, x+w, y+h],
                        'category': category,
                        'class_name': VISDRONE_CLASSES[category],
                        'score': score,
                        'truncation': truncation,
                        'occlusion': occlusion
                    })
    
    return annotations


def visualize_sample(image_path: Path, ann_path: Path, output_path: Path = None):
    """Візуалізація зображення з bounding boxes"""
    
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    annotations = parse_annotation(ann_path)
    
    for ann in annotations:
        bbox = ann['bbox']
        label = ann['class_name']
        
        # Draw box
        draw.rectangle(bbox, outline='red', width=2)
        
        # Draw label
        draw.text((bbox[0], bbox[1]-10), label, fill='red')
    
    if output_path:
        image.save(output_path)
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()


def dataset_statistics(data_root: Path, split: str = 'train') -> Dict:
    """
    Статистика датасету
    
    Returns:
        Dict з статистикою: кількість зображень, об'єктів, розподіл класів тощо
    """
    
    images_dir = data_root / split / 'images'
    annotations_dir = data_root / split / 'annotations'
    
    image_files = list(images_dir.glob('*.jpg'))
    
    stats = {
        'split': split,
        'n_images': len(image_files),
        'class_distribution': Counter(),
        'objects_per_image': [],
        'image_sizes': []
    }
    
    for img_path in image_files:
        # Image size
        with Image.open(img_path) as img:
            stats['image_sizes'].append(img.size)
        
        # Annotations
        ann_path = annotations_dir / img_path.with_suffix('.txt').name
        if ann_path.exists():
            annotations = parse_annotation(ann_path)
            stats['objects_per_image'].append(len(annotations))
            
            for ann in annotations:
                stats['class_distribution'][ann['class_name']] += 1
    
    # Обчислення статистики
    stats['avg_objects_per_image'] = np.mean(stats['objects_per_image']) if stats['objects_per_image'] else 0
    stats['total_objects'] = sum(stats['class_distribution'].values())
    
    return stats


def print_statistics(stats: Dict):
    """Виведення статистики датасету"""
    
    print(f"\n{'='*60}")
    print(f"VisDrone Dataset Statistics: {stats['split']}")
    print(f"{'='*60}")
    print(f"📸 Зображень: {stats['n_images']}")
    print(f"📦 Об'єктів: {stats['total_objects']}")
    print(f"📊 Середня кількість об'єктів на зображення: {stats['avg_objects_per_image']:.2f}")
    
    print(f"\n🏷️  Розподіл класів:")
    for class_name, count in stats['class_distribution'].most_common():
        percentage = (count / stats['total_objects']) * 100
        print(f"   {class_name:20s}: {count:6d} ({percentage:5.2f}%)")


if __name__ == '__main__':
    # Приклад використання
    data_root = Path('data')
    
    # Train statistics
    train_stats = dataset_statistics(data_root, 'train')
    print_statistics(train_stats)
    
    # Val statistics
    val_stats = dataset_statistics(data_root, 'val')
    print_statistics(val_stats)
