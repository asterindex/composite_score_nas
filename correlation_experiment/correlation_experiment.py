#!/usr/bin/env python3
"""
Кореляційний експеримент:
- Генерація 30 випадкових архітектур
- Повне тренування кожної на 15 епох
- Збір всіх метрик на кожній епосі
- Кореляційний аналіз: які метрики на ранніх епохах предсказують фінальну якість
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import csv

# ============================================
# ПАРАМЕТРИ ЕКСПЕРИМЕНТУ
# ============================================

N_MODELS = 30               # Кількість моделей для тестування (30 для повного)
EPOCHS = 10                 # Епох тренування (15 для повного)
MAX_SAMPLES = 2000          # Samples для тренування (2000 для повного)
IMG_SIZE = 320
BATCH_SIZE = 32            # Фіксований batch size для всіх моделей
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
SEED = 42

# Шляхи до даних (як в synthesis_universal.py)
# SimpleDataset сам розбереться з відносними шляхами
TRAIN_IMAGES = "dataset/train/images"
TRAIN_LABELS = "dataset/train/annotations"
VAL_IMAGES = "dataset/val/images"
VAL_LABELS = "dataset/val/annotations"

# Output файли
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
METRICS_CSV = OUTPUT_DIR / "all_metrics_per_epoch.csv"
MODELS_JSON = OUTPUT_DIR / "models_config.json"

print("=" * 80)
print("КОРЕЛЯЦІЙНИЙ ЕКСПЕРИМЕНТ")
print("=" * 80)
print()
print(f"⚙️  ПАРАМЕТРИ:")
print(f"   Моделей: {N_MODELS}")
print(f"   Епох: {EPOCHS}")
print(f"   Dataset samples: {MAX_SAMPLES}")
print(f"   Image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Device: {DEVICE}")
print()
print(f"📁 ШЛЯХИ:")
print(f"   Train images: {TRAIN_IMAGES}")
print(f"   Train labels: {TRAIN_LABELS}")
print(f"   Val images: {VAL_IMAGES}")
print(f"   Val labels: {VAL_LABELS}")
print()

# ============================================
# DATASET
# ============================================

class SimpleDataset(Dataset):
    """Датасет для експерименту (копія з synthesis_universal.py)"""
    
    def __init__(self, image_dir, label_dir, img_size=320, max_samples=-1):
        self.img_size = img_size
        
        # Визначити шлях до датасету (як в synthesis_universal.py)
        if Path(image_dir).is_absolute():
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir)
        elif Path(image_dir).exists():
            # Датасет в поточній директорії (Colab/Jupyter)
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir)
        else:
            # Локальний запуск з correlation_experiment/
            try:
                script_dir = Path(__file__).parent
                self.image_dir = (script_dir / ".." / image_dir).resolve()
                self.label_dir = (script_dir / ".." / label_dir).resolve()
            except NameError:
                # Jupyter без __file__
                self.image_dir = Path(image_dir)
                self.label_dir = Path(label_dir)
        
        # Завантажити список зображень
        all_images = sorted(list(self.image_dir.glob("*.jpg")))
        self.images = all_images[:max_samples] if max_samples > 0 else all_images
        
        if len(self.images) == 0:
            raise ValueError(f"Не знайдено зображень в {self.image_dir}")
        
        print(f"   Dataset: {len(all_images)} знайдено, використовую {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Завантажити зображення
        img_path = self.images[idx]
        
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        
        # Завантажити анотації
        ann_path = self.label_dir / f"{img_path.stem}.txt"
        
        boxes = []
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                lines = [line.strip().split(',') for line in f if line.strip()]
            
            for line in lines:
                if len(line) >= 8:
                    x, y, w, h = map(int, line[:4])
                    cls = int(line[5])
                    if w > 0 and h > 0:
                        # Нормалізуємо координати до 0-1
                        x_norm = x / self.img_size
                        y_norm = y / self.img_size
                        w_norm = w / self.img_size
                        h_norm = h / self.img_size
                        boxes.append([x_norm, y_norm, x_norm + w_norm, y_norm + h_norm, cls / 10.0])  # cls також нормалізуємо
        
        if len(boxes) == 0:
            boxes = [[0.0, 0.0, 0.1, 0.1, 0.0]]  # Нормалізовані координати
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        return img, boxes

# ============================================
# MODEL
# ============================================

class DynamicDetector(nn.Module):
    def __init__(self, num_blocks, filters_list, kernel_sizes, fc_size, dropout, activation, num_classes=11):
        super().__init__()
        self.num_blocks = num_blocks
        
        # Activation
        act_dict = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'gelu': nn.GELU()
        }
        self.activation = act_dict.get(activation, nn.ReLU())
        
        # Conv blocks
        layers = []
        in_channels = 3
        for i in range(num_blocks):
            out_channels = filters_list[i]
            kernel = kernel_sizes[i]
            padding = kernel // 2
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(self.activation)
            layers.append(nn.MaxPool2d(2))
            
            in_channels = out_channels
        
        self.backbone = nn.Sequential(*layers)
        
        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, fc_size),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(fc_size, 5 * num_classes)  # [x, y, w, h, class_id] * num_classes
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.fc(x)
        return x.view(x.size(0), -1, 5)

# ============================================
# RANDOM ARCHITECTURE GENERATOR
# ============================================

def generate_random_architecture():
    """Генерує випадкову архітектуру"""
    config = {
        'num_blocks': random.randint(2, 5),
        'activation': random.choice(['relu', 'leaky_relu', 'gelu']),
        'fc_size': random.choice([64, 128, 256]),
        'dropout': random.choice([0.3, 0.5, 0.7]),
        'lr': random.choice([0.0001, 0.001, 0.01]),
        'optimizer': random.choice(['adam', 'adamw', 'sgd']),
        'weight_decay': random.choice([0, 1e-5, 1e-4, 1e-3])
    }
    
    # Generate filters and kernels
    config['filters_list'] = [random.choice([16, 32, 64, 128]) for _ in range(config['num_blocks'])]
    config['kernel_sizes'] = [random.choice([3, 5]) for _ in range(config['num_blocks'])]
    
    return config

# ============================================
# GRADIENT NORM COMPUTATION
# ============================================

def compute_grad_norm(model):
    """Обчислює L2 норму градієнтів"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

# ============================================
# TRAINING WITH METRICS COLLECTION
# ============================================

def train_model_with_metrics(model, config, train_loader, val_loader, model_idx):
    """
    Тренує модель та збирає всі метрики на кожній епосі
    
    Returns:
        list of dicts: метрики для кожної епохи
    """
    # Setup optimizer
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9)
    
    criterion = nn.MSELoss()
    
    all_epoch_metrics = []
    
    print(f"\n{'='*80}")
    print(f"МОДЕЛЬ #{model_idx + 1}/{N_MODELS}")
    print(f"{'='*80}")
    print(f"   Структура: {config['num_blocks']} блоків")
    print(f"   Фільтри: {config['filters_list']}")
    print(f"   Activation: {config['activation']}")
    print(f"   Optimizer: {config['optimizer']}, LR: {config['lr']}")
    print()
    
    for epoch in range(EPOCHS):
        # ==================== TRAINING ====================
        model.train()
        train_losses = []
        grad_norms = []
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(DEVICE)
            # targets - це list of tensors!
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            
            # Simple loss (MSE) - порівнюємо з першим box кожного зображення
            batch_size = outputs.size(0)
            target_tensor = torch.zeros(batch_size, 5).to(DEVICE)
            for i, target in enumerate(targets):
                if len(target) > 0:
                    target_tensor[i] = target[0, :5]  # Перший box
            
            loss = criterion(outputs.view(batch_size, -1)[:, :5], target_tensor)
            
            # Backward
            loss.backward()
            
            # Compute gradient norm BEFORE optimizer step
            grad_norm = compute_grad_norm(model)
            grad_norms.append(grad_norm)
            
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Training metrics for this epoch
        train_loss_mean = np.mean(train_losses)
        train_loss_std = np.std(train_losses)
        train_loss_cv = train_loss_std / (train_loss_mean + 1e-8)
        
        grad_norm_mean = np.mean(grad_norms)
        grad_norm_std = np.std(grad_norms)
        grad_norm_cv = grad_norm_std / (grad_norm_mean + 1e-8)
        
        # Learning progress (first vs last K batches)
        K = min(10, len(train_losses) // 2)
        train_loss_first_k = np.mean(train_losses[:K])
        train_loss_last_k = np.mean(train_losses[-K:])
        improvement = max(0, train_loss_first_k - train_loss_last_k)
        
        # ==================== VALIDATION ====================
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                
                outputs = model(images)
                batch_size = outputs.size(0)
                target_tensor = torch.zeros(batch_size, 5).to(DEVICE)
                for i, target in enumerate(targets):
                    if len(target) > 0:
                        target_tensor[i] = target[0, :5]
                
                loss = criterion(outputs.view(batch_size, -1)[:, :5], target_tensor)
                val_losses.append(loss.item())
        
        val_loss_mean = np.mean(val_losses)
        val_loss_std = np.std(val_losses)
        val_loss_cv = val_loss_std / (val_loss_mean + 1e-8)
        
        # Gap (overfitting indicator)
        gap = max(0, val_loss_mean - train_loss_mean)
        
        # ==================== STORE METRICS ====================
        epoch_metrics = {
            'model_idx': model_idx,
            'epoch': epoch + 1,
            
            # Loss metrics
            'train_loss': train_loss_mean,
            'train_loss_std': train_loss_std,
            'train_loss_cv': train_loss_cv,
            'val_loss': val_loss_mean,
            'val_loss_std': val_loss_std,
            'val_loss_cv': val_loss_cv,
            
            # Training dynamics
            'gap': gap,
            'improvement': improvement,
            'train_loss_first_k': train_loss_first_k,
            'train_loss_last_k': train_loss_last_k,
            
            # Gradient metrics
            'grad_norm_mean': grad_norm_mean,
            'grad_norm_std': grad_norm_std,
            'grad_norm_cv': grad_norm_cv,
            
            # Config
            'lr': config['lr'],
            'num_blocks': config['num_blocks'],
            'fc_size': config['fc_size'],
            'dropout': config['dropout'],
        }
        
        all_epoch_metrics.append(epoch_metrics)
        
        # Print progress (кожна епоха!)
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train: {train_loss_mean:.4f} | Val: {val_loss_mean:.4f} | "
              f"Gap: {gap:.4f} | Grad: {grad_norm_mean:.2f}")
    
    print(f"\n✅ Модель #{model_idx + 1} завершена! Best Val: {min(m['val_loss'] for m in all_epoch_metrics):.4f}")
    
    return all_epoch_metrics

# ============================================
# MAIN EXPERIMENT
# ============================================

def main():
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # ==================== PREPARE DATA ====================
    print("📦 Завантаження датасету...")
    train_dataset = SimpleDataset(TRAIN_IMAGES, TRAIN_LABELS, IMG_SIZE, MAX_SAMPLES)
    val_dataset = SimpleDataset(VAL_IMAGES, VAL_LABELS, IMG_SIZE, max_samples=200)  # Менше для швидкості
    
    # Collate function для різних розмірів boxes
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]  # List of tensors (не stack!)
        return images, targets
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    print()
    
    # ==================== GENERATE ARCHITECTURES ====================
    print("🏗️  Генерація архітектур...")
    architectures = []
    for i in range(N_MODELS):
        config = generate_random_architecture()
        architectures.append(config)
    
    # Save architectures
    with open(MODELS_JSON, 'w') as f:
        json.dump(architectures, f, indent=2)
    print(f"   Збережено конфігурації: {MODELS_JSON}")
    print()
    
    # ==================== TRAIN ALL MODELS ====================
    print(f"🚀 Тренування {N_MODELS} моделей по {EPOCHS} епох...")
    print()
    
    all_metrics = []
    
    for model_idx, config in enumerate(architectures):
        # Build model
        model = DynamicDetector(
            num_blocks=config['num_blocks'],
            filters_list=config['filters_list'],
            kernel_sizes=config['kernel_sizes'],
            fc_size=config['fc_size'],
            dropout=config['dropout'],
            activation=config['activation']
        ).to(DEVICE)
        
        # Train and collect metrics
        try:
            epoch_metrics = train_model_with_metrics(model, config, train_loader, val_loader, model_idx)
            all_metrics.extend(epoch_metrics)
        except Exception as e:
            print(f"❌ Модель #{model_idx + 1} failed: {e}")
            continue
    
    # ==================== SAVE METRICS ====================
    print()
    print("💾 Збереження метрик...")
    
    # Write to CSV
    if len(all_metrics) > 0:
        fieldnames = list(all_metrics[0].keys())
        with open(METRICS_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        
        print(f"   Збережено метрики: {METRICS_CSV}")
        print(f"   Всього записів: {len(all_metrics)}")
    
    print()
    print("=" * 80)
    print("✅ ЕКСПЕРИМЕНТ ЗАВЕРШЕНО!")
    print("=" * 80)
    print()
    print(f"📊 Наступний крок: запустіть аналіз кореляцій")
    print(f"   python analyze_correlations.py")
    print()

if __name__ == "__main__":
    main()
