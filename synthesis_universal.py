"""
Stability-Aware Proxy для бюджетного Bayesian Optimization
Detection Stability Score (DSS) для автоматичного синтезу компактних CNN-детекторів

Автор: Анатолій Кот
Дата: 2026-01-24
"""

import os
import sys
import json
import time
import random
import pickle
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import optuna
from optuna.samplers import TPESampler


# ============================================================================
# КОНФІГУРАЦІЯ
# ============================================================================

SEED = 42
N_TRIALS = 30
N_WARMUP = 10
EPOCHS_PER_TRIAL = 1
K_BATCHES = 10

MAX_SAMPLES = 700
VAL_SUBSET = 200
IMG_SIZE = 320

RESULTS_DIR = Path('results')
CHECKPOINT_FILE = RESULTS_DIR / 'optuna_study.pkl'
STATS_FILE = RESULTS_DIR / 'proxy_stats.json'
RESULTS_FILE = RESULTS_DIR / 'synthesis_results.json'

# Автоматичне визначення платформи
try:
    import google.colab
    IS_COLAB = True
    DATA_ROOT = Path('/content/data')
    DRIVE_ROOT = Path('/content/drive/MyDrive/Studying/Experiments/Composite_score_nas')
except ImportError:
    IS_COLAB = False
    DATA_ROOT = Path('data')
    DRIVE_ROOT = None

# Визначення device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    DEVICE_NAME = 'CUDA'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    DEVICE_NAME = 'MPS'
else:
    DEVICE = torch.device('cpu')
    DEVICE_NAME = 'CPU'


# ============================================================================
# LOGGING
# ============================================================================

LOG_FILE = None

def setup_logging():
    """Налаштування логування з UTC timestamps"""
    global LOG_FILE
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    LOG_FILE = RESULTS_DIR / f'experiment_{timestamp}.log'
    log_print(f"🚀 Запуск експерименту: {timestamp} UTC")
    log_print(f"📍 Platform: {DEVICE_NAME} | Colab: {IS_COLAB}")

def log_print(msg: str):
    """Вивід з UTC timestamp у консоль та файл"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    if LOG_FILE:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')


# ============================================================================
# DATASET
# ============================================================================

class VisDroneDataset(Dataset):
    """VisDrone2019-DET Dataset для детекції об'єктів"""
    
    def __init__(self, root: Path, split: str = 'train', transform=None):
        self.root = root / split
        self.images_dir = self.root / 'images'
        self.annotations_dir = self.root / 'annotations'
        self.transform = transform
        
        self.image_files = sorted(self.images_dir.glob('*.jpg'))
        log_print(f"📦 Dataset {split}: {len(self.image_files)} зображень")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Завантаження анотацій (формат: bbox_left, bbox_top, bbox_width, bbox_height, score, category, ...)
        ann_path = self.annotations_dir / img_path.with_suffix('.txt').name
        boxes = []
        labels = []
        
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        x, y, w, h, score, category = map(int, parts[:6])
                        if score > 0 and category in range(1, 11):  # 10 класів
                            boxes.append([x, y, x+w, y+h])
                            labels.append(category)
        
        if self.transform:
            image = self.transform(image)
        
        # Конвертація у тензори
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        
        return image, {'boxes': boxes, 'labels': labels}


def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Створення DataLoader'ів для train/val"""
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Train dataset (підмножина для швидкості)
    train_dataset = VisDroneDataset(DATA_ROOT, split='train', transform=transform)
    train_indices = list(range(min(MAX_SAMPLES, len(train_dataset))))
    train_subset = Subset(train_dataset, train_indices)
    
    # Val dataset (фіксована підмножина)
    val_dataset = VisDroneDataset(DATA_ROOT, split='val', transform=transform)
    val_indices = list(range(min(VAL_SUBSET, len(val_dataset))))
    val_subset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    
    return train_loader, val_loader


# ============================================================================
# MODEL
# ============================================================================

class DynamicDetector(nn.Module):
    """Динамічний CNN-детектор з параметризованою архітектурою"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Параметри
        n_blocks = config['n_blocks']
        filter_sizes = config['filter_sizes']
        kernel_sizes = config['kernel_sizes']
        fc_size = config['fc_size']
        dropout = config['dropout']
        activation = config['activation']
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        else:  # gelu
            act_fn = nn.GELU
        
        # Convolutional blocks
        layers = []
        in_channels = 3
        
        for i in range(n_blocks):
            out_channels = filter_sizes[i]
            kernel = kernel_sizes[i]
            padding = kernel // 2
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.BatchNorm2d(out_channels),
                act_fn(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Розрахунок розміру після conv-блоків
        feature_size = IMG_SIZE // (2 ** n_blocks)
        flat_size = in_channels * feature_size * feature_size
        
        # Detection head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_size),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 10)  # 10 класів VisDrone
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion,
                track_stability: bool = False) -> Dict[str, float]:
    """Тренування однієї епохи з опціональним tracking стабільності"""
    
    model.train()
    losses = []
    grad_norms = []
    
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(DEVICE)
        # Спрощена loss: CrossEntropy на першому bbox
        labels = targets['labels'][:, 0] if targets['labels'].size(1) > 0 else torch.zeros(images.size(0), dtype=torch.long)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient norm
        if track_stability:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norms.append(total_norm ** 0.5)
        
        optimizer.step()
        losses.append(loss.item())
        
        # Обмеження для K_BATCHES
        if track_stability and batch_idx >= K_BATCHES - 1:
            break
    
    metrics = {'loss_mean': np.mean(losses)}
    
    if track_stability and len(losses) > 1:
        metrics['loss_std'] = np.std(losses)
        metrics['loss_cv'] = metrics['loss_std'] / (metrics['loss_mean'] + 1e-8)
        metrics['grad_norm_mean'] = np.mean(grad_norms)
        metrics['grad_norm_std'] = np.std(grad_norms)
        metrics['grad_cv'] = metrics['grad_norm_std'] / (metrics['grad_norm_mean'] + 1e-8)
    
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, criterion) -> float:
    """Оцінка на валідації"""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            labels = targets['labels'][:, 0] if targets['labels'].size(1) > 0 else torch.zeros(images.size(0), dtype=torch.long)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    
    return np.mean(losses)


# ============================================================================
# PROXY STATISTICS
# ============================================================================

class ProxyStatistics:
    """Robust-статистика для z-нормалізації DSS компонентів"""
    
    def __init__(self):
        self.stats = {}
        self.warmup_data = {
            'impr': [],
            'L_val': [],
            'loss_cv': [],
            'grad_cv': [],
            'gap': [],
            'L_tr_last': [],
            'grad_norm_mean': []
        }
    
    def add_warmup_trial(self, metrics: Dict[str, float]):
        """Додати trial у warmup-фазу"""
        for key in self.warmup_data:
            if key in metrics:
                self.warmup_data[key].append(metrics[key])
    
    def calibrate(self):
        """Калібрувати robust-статистику (median + IQR)"""
        for key, values in self.warmup_data.items():
            if len(values) > 0:
                median = np.median(values)
                q25 = np.percentile(values, 25)
                q75 = np.percentile(values, 75)
                iqr = q75 - q25
                self.stats[key] = {
                    'median': float(median),
                    'iqr': float(iqr if iqr > 1e-8 else 1.0)
                }
        
        log_print(f"📊 Калібровано статистику на {len(self.warmup_data['L_val'])} warmup trials")
    
    def z_normalize(self, key: str, value: float) -> float:
        """Robust z-нормалізація"""
        if key not in self.stats:
            return 0.0
        median = self.stats[key]['median']
        iqr = self.stats[key]['iqr']
        return (value - median) / iqr
    
    def save(self, path: Path):
        """Зберегти статистику"""
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load(self, path: Path):
        """Завантажити статистику"""
        with open(path, 'r') as f:
            self.stats = json.load(f)


# ============================================================================
# DETECTION STABILITY SCORE (DSS)
# ============================================================================

def compute_dss(metrics: Dict[str, float], stats: ProxyStatistics) -> Tuple[float, str]:
    """
    Обчислення Detection Stability Score (DSS)
    
    Формула:
    DSS = 0.25·z(impr) + 0.20·z(L_val) + 0.15·z(loss_cv) + 0.15·z(grad_cv) + 
          0.15·z(gap) + 0.05·z(L_tr) + 0.05·z(grad_norm)
    
    Ключові принципи:
    - ВСІ метрики позитивно корелюють з final loss (чим менше - тим краще)
    - z-нормалізація для коректного масштабу
    - Ваги базуються на кореляційному аналізі
    """
    
    # Z-нормалізовані компоненти
    z_impr = stats.z_normalize('impr', metrics['impr'])
    z_L_val = stats.z_normalize('L_val', metrics['L_val'])
    z_loss_cv = stats.z_normalize('loss_cv', metrics['loss_cv'])
    z_grad_cv = stats.z_normalize('grad_cv', metrics['grad_cv'])
    z_gap = stats.z_normalize('gap', metrics['gap'])
    z_L_tr_last = stats.z_normalize('L_tr_last', metrics['L_tr_last'])
    z_grad_norm = stats.z_normalize('grad_norm_mean', metrics['grad_norm_mean'])
    
    # Композитний score
    dss = (
        0.25 * z_impr +
        0.20 * z_L_val +
        0.15 * z_loss_cv +
        0.15 * z_grad_cv +
        0.15 * z_gap +
        0.05 * z_L_tr_last +
        0.05 * z_grad_norm
    )
    
    return dss, 'DSS'


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

PROXY_STATS = ProxyStatistics()
TRIAL_COUNTER = 0

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function з DSS"""
    
    global TRIAL_COUNTER
    TRIAL_COUNTER += 1
    is_warmup = TRIAL_COUNTER <= N_WARMUP
    
    log_print(f"\n{'='*60}")
    log_print(f"🔍 Trial {TRIAL_COUNTER}/{N_TRIALS} {'(WARMUP)' if is_warmup else ''}")
    
    # Семплювання гіперпараметрів
    n_blocks = trial.suggest_int('n_blocks', 2, 5)
    filter_sizes = [trial.suggest_categorical(f'filter_{i}', [16, 32, 64, 128]) 
                    for i in range(n_blocks)]
    kernel_sizes = [trial.suggest_categorical(f'kernel_{i}', [3, 5]) 
                   for i in range(n_blocks)]
    fc_size = trial.suggest_categorical('fc_size', [32, 64, 128])
    dropout = trial.suggest_categorical('dropout', [0.3, 0.5, 0.7])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu'])
    
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01])
    weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    config = {
        'n_blocks': n_blocks,
        'filter_sizes': filter_sizes,
        'kernel_sizes': kernel_sizes,
        'fc_size': fc_size,
        'dropout': dropout,
        'activation': activation
    }
    
    log_print(f"🏗️  Architecture: {n_blocks} blocks, filters={filter_sizes}, kernels={kernel_sizes}")
    log_print(f"⚙️  Optimizer: {optimizer_name.upper()} (LR={lr}, WD={weight_decay}), BS={batch_size}")
    
    # Створення моделі
    model = DynamicDetector(config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Завантаження даних
    train_loader, val_loader = get_dataloaders(batch_size)
    
    # Початкова валідація
    initial_val_loss = evaluate(model, val_loader, criterion)
    
    # Тренування з tracking стабільності
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, track_stability=True)
    
    # Фінальна валідація
    final_val_loss = evaluate(model, val_loader, criterion)
    
    # Розрахунок метрик DSS
    metrics = {
        'impr': initial_val_loss - final_val_loss,  # Покращення (чим більше - тим краще навчання)
        'L_val': final_val_loss,
        'L_tr_last': train_metrics['loss_mean'],
        'gap': final_val_loss - train_metrics['loss_mean'],
        'loss_cv': train_metrics.get('loss_cv', 0.0),
        'grad_cv': train_metrics.get('grad_cv', 0.0),
        'grad_norm_mean': train_metrics.get('grad_norm_mean', 0.0)
    }
    
    log_print(f"📈 Metrics: L_val={metrics['L_val']:.4f}, impr={metrics['impr']:.4f}, gap={metrics['gap']:.4f}")
    
    # Warmup або DSS
    if is_warmup:
        PROXY_STATS.add_warmup_trial(metrics)
        proxy_value = final_val_loss
        proxy_name = 'val_loss'
        log_print(f"🔥 Warmup proxy: {proxy_value:.4f}")
        
        # Калібрувати після останнього warmup
        if TRIAL_COUNTER == N_WARMUP:
            PROXY_STATS.calibrate()
            PROXY_STATS.save(STATS_FILE)
    else:
        proxy_value, proxy_name = compute_dss(metrics, PROXY_STATS)
        log_print(f"✨ DSS: {proxy_value:.4f}")
    
    # Optuna мінімізує, тому для DSS (більше = краще) повертаємо -DSS
    return -proxy_value if proxy_name == 'DSS' else proxy_value


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Основний пайплайн синтезу"""
    
    # Setup
    setup_logging()
    set_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    log_print(f"🎯 Параметри експерименту:")
    log_print(f"   - N_TRIALS: {N_TRIALS}")
    log_print(f"   - N_WARMUP: {N_WARMUP}")
    log_print(f"   - EPOCHS_PER_TRIAL: {EPOCHS_PER_TRIAL}")
    log_print(f"   - MAX_SAMPLES: {MAX_SAMPLES}")
    log_print(f"   - VAL_SUBSET: {VAL_SUBSET}")
    log_print(f"   - SEED: {SEED}")
    log_print(f"   - DEVICE: {DEVICE_NAME}")
    
    # Optuna study
    sampler = TPESampler(seed=SEED, n_startup_trials=N_WARMUP)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    start_time = time.time()
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    except KeyboardInterrupt:
        log_print("\n⚠️  Експеримент перервано користувачем")
    
    elapsed = time.time() - start_time
    log_print(f"\n⏱️  Час синтезу: {elapsed/60:.2f} хвилин")
    
    # Збереження результатів
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(study, f)
    
    # Аналіз результатів
    analyze_results(study)
    
    log_print(f"\n✅ Експеримент завершено!")
    log_print(f"📊 Результати збережено у: {RESULTS_DIR}")
    
    if LOG_FILE:
        log_print(f"📝 Лог: {LOG_FILE}")


def analyze_results(study: optuna.Study):
    """Аналіз результатів синтезу"""
    
    log_print(f"\n{'='*60}")
    log_print("📊 АНАЛІЗ РЕЗУЛЬТАТІВ")
    log_print(f"{'='*60}")
    
    # Найкращий trial
    best_trial = study.best_trial
    log_print(f"\n🏆 Найкращий Trial #{best_trial.number}")
    log_print(f"   Proxy value: {best_trial.value:.4f}")
    log_print(f"   Параметри:")
    for key, value in best_trial.params.items():
        log_print(f"      {key}: {value}")
    
    # Топ-3
    log_print(f"\n🥇 Топ-3 архітектури:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value)
    for i, trial in enumerate(trials_sorted[:3], 1):
        log_print(f"\n   #{i} Trial {trial.number}: {trial.value:.4f}")
        n_blocks = trial.params['n_blocks']
        filters = [trial.params[f'filter_{j}'] for j in range(n_blocks)]
        kernels = [trial.params[f'kernel_{j}'] for j in range(n_blocks)]
        log_print(f"      Architecture: {n_blocks} blocks, filters={filters}, kernels={kernels}")
        log_print(f"      Optimizer: {trial.params['optimizer'].upper()} (LR={trial.params['lr']}, WD={trial.params['weight_decay']})")
        log_print(f"      Activation: {trial.params['activation']}, Dropout: {trial.params['dropout']}")
    
    # Збереження у JSON
    results = {
        'best_trial': {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params
        },
        'top3': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in trials_sorted[:3]
        ],
        'metadata': {
            'n_trials': len(study.trials),
            'device': DEVICE_NAME,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def set_seed(seed: int):
    """Встановлення seed для відтворюваності"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()
