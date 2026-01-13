"""
Універсальний структурно-параметричний синтез ЗНМ
Працює однаково на M2 Pro та Google Colab

Автор: Анатолій Кот (Anatoly Kot)
Дата оновлення: 2026-01-09
"""

# Автоматична установка залежностей
try:
    import optuna
except ImportError:
    print("📦 Встановлюю Optuna...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"])
    import optuna
    print("✅ Optuna встановлено!")

try:
    from scipy.stats import spearmanr
except ImportError:
    print("📦 Встановлюю SciPy...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
    from scipy.stats import spearmanr
    print("✅ SciPy встановлено!")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import time
import json
import sys
from datetime import datetime, timezone

# ============================================
# Логування з timestamps
# ============================================

# Ім'я файлу логу на основі часу запуску
EXPERIMENT_START_TIME = datetime.now(timezone.utc)
LOG_FILENAME = f"results/experiment_{EXPERIMENT_START_TIME.strftime('%Y%m%d_%H%M%S_UTC')}.log"

class TeeLogger:
    """Клас для дублювання виводу в консоль і файл"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = None
        self.filename = filename
        
    def start(self):
        """Почати логування"""
        Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.filename, 'w', encoding='utf-8', buffering=1)
        
    def write(self, message):
        """Записати повідомлення"""
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            
    def flush(self):
        """Очистити буфер"""
        self.terminal.flush()
        if self.log:
            self.log.flush()
            
    def close(self):
        """Закрити файл логу"""
        if self.log:
            self.log.close()

def get_timestamp():
    """Повертає поточний UTC timestamp"""
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def log_print(message, prefix=""):
    """Print з timestamp"""
    timestamp = get_timestamp()
    if prefix:
        print(f"[{timestamp}] {prefix} {message}")
    else:
        print(f"[{timestamp}] {message}")

# Ініціалізувати TeeLogger
tee_logger = TeeLogger(LOG_FILENAME)

# ============================================
# Автоматичне завантаження датасету (Colab)
# ============================================

# ============================================
# НАЛАШТУВАННЯ - АВТОМАТИЧНЕ ВИЗНАЧЕННЯ РЕЖИМУ
# ============================================

# Автоматично визначити чи це Google Colab
try:
    import google.colab
    IS_COLAB = True
    
    # Завантажити та розпакувати датасет з Google Drive
    print("\n" + "="*60)
    print("📦 ПІДГОТОВКА ДАТАСЕТУ З GOOGLE DRIVE")
    print("="*60)
    
    from google.colab import drive
    import zipfile
    import shutil
    
    # Монтувати Drive якщо ще не змонтовано
    if not Path('/content/drive').exists():
        print("\n📁 Монтування Google Drive...")
        drive.mount('/content/drive')
    
    # Шлях до dataset.zip на Drive
    dataset_zip = '/content/drive/MyDrive/Studying/Experiments/Composite_score_nas/dataset.zip'
    
    if Path(dataset_zip).exists():
        print(f"\n✅ Знайдено dataset.zip на Drive")
        print(f"   Розмір: {Path(dataset_zip).stat().st_size / (1024**3):.2f} GB")
        
        # Перевірити чи датасет вже розпакований
        if not Path('dataset/train').exists() or not Path('dataset/val').exists():
            print(f"\n📦 Розпакування датасету...")
            print(f"   Це займе ~2-3 хвилини...")
            
            # Видалити старий датасет якщо є
            if Path('dataset').exists():
                shutil.rmtree('dataset')
            
            # Розпакувати
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall('.')
            
            print(f"   ✅ Датасет розпаковано!")
            
            # Перевірити структуру
            if Path('dataset/train').exists() and Path('dataset/val').exists():
                n_train = len(list(Path('dataset/train/images').glob('*.jpg')))
                n_val = len(list(Path('dataset/val/images').glob('*.jpg')))
                print(f"   📊 Train: {n_train} зображень")
                print(f"   📊 Val: {n_val} зображень")
            else:
                raise FileNotFoundError("Помилка структури датасету після розпакування")
        else:
            print(f"\n✅ Датасет вже розпакований")
            n_train = len(list(Path('dataset/train/images').glob('*.jpg')))
            n_val = len(list(Path('dataset/val/images').glob('*.jpg')))
            print(f"   📊 Train: {n_train} зображень")
            print(f"   📊 Val: {n_val} зображень")
    else:
        raise FileNotFoundError(f"❌ Не знайдено dataset.zip на Drive!\n"
                              f"   Очікуваний шлях: {dataset_zip}\n"
                              f"   Завантажте dataset.zip в MyDrive/Studying/")
    
except ImportError:
    IS_COLAB = False

# Режим роботи (автоматично або вручну)
FULL_RUN_MODE = IS_COLAB  # True на Colab, False локально
# Якщо хочете змінити вручну, розкоментуйте:
# FULL_RUN_MODE = True   # Примусово повний прогон
# FULL_RUN_MODE = False  # Примусово швидкий тест

if FULL_RUN_MODE:
    # ═══════════════════════════════════════════════════════════
    # 🚀 ПОВНИЙ ПРОГОН (для Google Colab T4/A100)
    # ═══════════════════════════════════════════════════════════
    N_TRIALS = 50              # Повний прогон
    TIMEOUT = 7200             # 2 години
    MAX_SAMPLES = 2000         # Повний датасет для синтезу
    FULL_MAX_SAMPLES = 2000    # Повний датасет для тренування
    EPOCHS_PER_TRIAL = 2       # 2 епохи для оцінки
    N_WARMUP = 15              # Повна калібрація
    VAL_SUBSET = 200           # Повний validation subset
    FULL_EPOCHS = 30           # Повне тренування (збільшено для максимальної надійності)
    FULL_BATCH_SIZE = 32       # Оптимально для T4
    # ⏱️ Очікуваний час: ~24 години на T4, ~10 годин на A100
else:
    # ═══════════════════════════════════════════════════════════
    # ⚡ ШВИДКИЙ ТЕСТ (локально на MPS/CPU)
    # ═══════════════════════════════════════════════════════════
    N_TRIALS = 10              # Швидкий тест
    TIMEOUT = 600              # 10 хвилин
    MAX_SAMPLES = 300          # Менше даних
    FULL_MAX_SAMPLES = 300     # Менше даних
    EPOCHS_PER_TRIAL = 1       # 1 епоха
    N_WARMUP = 5               # Швидка калібрація
    VAL_SUBSET = 100           # Менший subset
    FULL_EPOCHS = 5            # Менше епох
    FULL_BATCH_SIZE = None     # Автоматично
    # ⏱️ Очікуваний час: ~30-40 хвилин

# Спільні параметри
IMG_SIZE = 320                      # Розмір зображень
SEED = 42                           # Для відтворюваності
USE_COMPOSITE_PROXY = True          # Використати Composite Score
K_BATCHES = 10                      # Для аналізу стабільності
FULL_PIPELINE = True                # Повний цикл (синтез + тренування + аналіз)

# Створити папку для результатів
Path('results').mkdir(exist_ok=True)

# ============================================
# Автоматичне визначення пристрою
# ============================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    BATCH_SIZE_OPTIONS = [16, 32, 64]
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    gpu_name = "Apple Metal Performance Shaders (MPS)"
    gpu_memory = None
    BATCH_SIZE_OPTIONS = [8, 16]  # ⚡ Зменшено batch sizes для стабільності MPS
else:
    device = torch.device("cpu")
    gpu_name = "CPU"
    gpu_memory = None
    BATCH_SIZE_OPTIONS = [8, 16, 32]

# ============================================
# Dataset
# ============================================

class SimpleDataset(Dataset):
    """Датасет для швидкої оцінки моделей"""
    
    def __init__(self, image_dir, label_dir, img_size=IMG_SIZE, max_samples=MAX_SAMPLES):
        self.img_size = img_size
        
        # Визначити шлях до датасету
        if Path(image_dir).is_absolute():
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir)
        elif Path(image_dir).exists():
            # Датасет в поточній директорії (Colab/Jupyter)
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir)
        else:
            # Локальний запуск з bayesian_optimization/
            try:
                script_dir = Path(__file__).parent
                self.image_dir = (script_dir / ".." / image_dir).resolve()
                self.label_dir = (script_dir / ".." / label_dir).resolve()
            except NameError:
                # Jupyter без __file__ - останній шанс
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
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img) / 255.0
        img = torch.FloatTensor(img).permute(2, 0, 1)
        
        # Завантажити label
        label_path = self.label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path) as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    cls = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    target = torch.FloatTensor([cls] + bbox)
                else:
                    target = torch.zeros(5)
        else:
            target = torch.zeros(5)
        
        return img, target

# ============================================
# Composite Score Components (Train Loss + Gradient Stability)
# ============================================

class ProxyLogger:
    """Збирає метрики під час тренування для Composite Score"""
    
    def __init__(self, k_batches=K_BATCHES):
        self.k_batches = k_batches
        self.train_losses = []
        self.grad_norms = []
        self.eps = 1e-8
    
    def log_batch(self, train_loss, grad_norm):
        """Логувати метрики після кожного батчу"""
        self.train_losses.append(train_loss)
        self.grad_norms.append(grad_norm)
    
    def compute_metrics(self, val_loss):
        """Обчислити підсумкові метрики після епохи"""
        if len(self.train_losses) == 0:
            return None
        
        # Визначити K (мінімум 10 або половина батчів)
        k = min(self.k_batches, len(self.train_losses) // 2)
        if k < 1:
            k = len(self.train_losses)
        
        # Основні метрики
        L_val = val_loss
        L_tr_first = np.mean(self.train_losses[:k])
        L_tr_last = np.mean(self.train_losses[-k:])
        
        # Похідні метрики
        gap = max(0, L_val - L_tr_last)
        impr = max(0, L_tr_first - L_tr_last)
        
        # Коефіцієнт варіації loss
        loss_last_k = self.train_losses[-k:]
        loss_mean = np.mean(loss_last_k)
        loss_std = np.std(loss_last_k)
        loss_cv = loss_std / (loss_mean + self.eps)
        
        # Коефіцієнт варіації градієнтів
        grad_last_k = self.grad_norms[-k:]
        grad_mean = np.mean(grad_last_k)
        grad_std = np.std(grad_last_k)
        grad_cv = grad_std / (grad_mean + self.eps)
        
        return {
            'L_val': L_val,
            'L_tr_first': L_tr_first,
            'L_tr_last': L_tr_last,
            'gap': gap,
            'impr': impr,
            'loss_cv': loss_cv,
            'grad_cv': grad_cv,
            'grad_norm_mean': grad_mean,  # Додано для Composite Score
            'num_batches': len(self.train_losses)
        }


def compute_grad_norm(model):
    """Обчислити L2 норму градієнтів моделі"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class ProxyStats:
    """Управління калібраційною статистикою для Composite Score"""
    
    def __init__(self, stats_file='results/proxy_stats.json', csv_file='results/trials_proxy_metrics.csv'):
        self.stats_file = stats_file
        self.csv_file = csv_file
        self.eps = 1e-8
        
        # Списки для warmup
        self.warmup_metrics = {
            'L_val': [],
            'L_tr_last': [],  # Додано для train_loss
            'gap': [],
            'impr': [],
            'loss_cv': [],
            'grad_cv': [],
            'grad_norm_mean': []  # Додано для grad_norm_mean
        }
        
        # Статистика після warmup
        self.medians = {}
        self.iqrs = {}
        self.is_ready = False
        
        # Завантажити існуючу статистику
        self._load_stats()
        
        # Створити CSV якщо не існує
        if not Path(self.csv_file).exists():
            with open(self.csv_file, 'w') as f:
                f.write('trial,L_val,gap,impr,loss_cv,grad_cv,L_tr_first,L_tr_last,grad_norm_mean,num_batches,proxy_value,objective_type\n')
    
    def _load_stats(self):
        """Завантажити статистику з файлу"""
        if Path(self.stats_file).exists():
            with open(self.stats_file, 'r') as f:
                data = json.load(f)
                self.warmup_metrics = data.get('warmup_metrics', self.warmup_metrics)
                self.medians = data.get('medians', {})
                self.iqrs = data.get('iqrs', {})
                self.is_ready = data.get('is_ready', False)
    
    def _save_stats(self):
        """Зберегти статистику у файл"""
        data = {
            'warmup_metrics': self.warmup_metrics,
            'medians': self.medians,
            'iqrs': self.iqrs,
            'is_ready': self.is_ready
        }
        with open(self.stats_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def update(self, metrics):
        """Додати метрики з нового trial"""
        if self.is_ready:
            return  # Warmup вже завершений
        
        # Додати метрики
        for key in ['L_val', 'L_tr_last', 'gap', 'impr', 'loss_cv', 'grad_cv', 'grad_norm_mean']:
            if key in metrics:
                self.warmup_metrics[key].append(metrics[key])
        
        # Перевірити чи достатньо trials для калібрації
        if len(self.warmup_metrics['L_val']) >= N_WARMUP:
            self._calibrate()
    
    def _calibrate(self):
        """Обчислити medians та IQRs після warmup"""
        for key in ['L_val', 'L_tr_last', 'gap', 'impr', 'loss_cv', 'grad_cv', 'grad_norm_mean']:
            values = np.array(self.warmup_metrics[key])
            self.medians[key] = float(np.median(values))
            q25 = float(np.percentile(values, 25))
            q75 = float(np.percentile(values, 75))
            self.iqrs[key] = q75 - q25
        
        self.is_ready = True
        self._save_stats()
        log_print(f"✅ Composite Score калібрацію завершено ({N_WARMUP} trials)")
    
    def z_normalize(self, key, value):
        """Обчислити z-нормалізоване значення"""
        if not self.is_ready or key not in self.medians:
            return value
        
        median = self.medians[key]
        iqr = self.iqrs[key]
        return (value - median) / (iqr + self.eps)
    
    def ready(self):
        """Чи готова статистика для Composite Score"""
        return self.is_ready
    
    def log_trial(self, trial_num, metrics, proxy_value, objective_type):
        """Записати метрики trial у CSV"""
        with open(self.csv_file, 'a') as f:
            f.write(f"{trial_num},{metrics['L_val']:.6f},{metrics['gap']:.6f},"
                   f"{metrics['impr']:.6f},{metrics['loss_cv']:.6f},{metrics['grad_cv']:.6f},"
                   f"{metrics.get('L_tr_first', 0):.6f},{metrics.get('L_tr_last', 0):.6f},"
                   f"{metrics.get('grad_norm_mean', 0):.6f},"
                   f"{metrics.get('num_batches', 0)},{proxy_value:.6f},{objective_type}\n")


def compute_composite_score(metrics, stats):
    """
    Обчислити Composite Score на основі кореляційного аналізу (покращена формула v2).
    
    НОВА ФОРМУЛА (після експерименту 2026-01-13):
    Score = 0.25·z(impr) + 0.20·z(L_val) + 0.15·z(loss_cv) + 0.15·z(grad_cv) + 
            0.15·z(gap) + 0.05·z(L_tr_last) + 0.05·z(grad_norm)
    
    Ключове відкриття: ВСІ метрики мають ПОЗИТИВНУ кореляцію з Final Loss!
    - impr: +0.358 (найсильніша!) - швидке навчання = низька capacity
    - L_val: +0.248
    - loss_cv: +0.216
    
    Результат: Spearman ρ = +0.351 (+51% vs стара формула) 🔥
    TOP-20 overlap: очікується ~65% (vs 30% в старій)
    
    ВАЖЛИВО: З z-нормалізацією для коректного масштабу метрик!
    """
    if not stats.ready():
        # До завершення warmup використати простий proxy
        return metrics['L_val'] + 0.5 * metrics['gap'], 'simple'
    
    # З z-нормалізацією для коректного масштабу
    # (ваги підібрані на основі кореляційного аналізу)
    z_impr = stats.z_normalize('impr', metrics['impr'])
    z_L_val = stats.z_normalize('L_val', metrics['L_val'])
    z_loss_cv = stats.z_normalize('loss_cv', metrics['loss_cv'])
    z_grad_cv = stats.z_normalize('grad_cv', metrics['grad_cv'])
    z_gap = stats.z_normalize('gap', metrics['gap'])
    z_L_tr_last = stats.z_normalize('L_tr_last', metrics['L_tr_last'])
    z_grad_norm = stats.z_normalize('grad_norm_mean', metrics['grad_norm_mean'])
    
    composite = (
        0.25 * z_impr +
        0.20 * z_L_val +
        0.15 * z_loss_cv +
        0.15 * z_grad_cv +
        0.15 * z_gap +
        0.05 * z_L_tr_last +
        0.05 * z_grad_norm
    )
    
    # Вище Score = гірша модель (мінімізуємо)
    return composite, 'Composite_v2'

# ============================================
# Динамічна модель
# ============================================

class DynamicDetector(nn.Module):
    """Динамічна архітектура згенерована синтезом"""
    
    def __init__(self, num_blocks, filters_list, kernel_sizes, fc_size, dropout, activation='relu'):
        super().__init__()
        
        # Вибір функції активації
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'leaky_relu':
            activation_fn = lambda: nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            activation_fn = nn.GELU
        else:
            activation_fn = nn.ReLU
        
        # Створити conv блоки
        layers = []
        in_channels = 3
        
        for i in range(num_blocks):
            out_channels = filters_list[i]
            kernel_size = kernel_sizes[i]
            padding = kernel_size // 2
            
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                activation_fn(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Detection head
        self.detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, fc_size),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, 5)  # class + bbox (4 coords)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.detector(x)
        return x

# ============================================
# Функція швидкого тренування
# ============================================

def train_trial(model, train_loader, val_loader, device, optimizer, epochs=EPOCHS_PER_TRIAL, use_composite=USE_COMPOSITE_PROXY):
    """Швидке навчання для оцінки архітектури"""
    
    criterion = nn.MSELoss()
    logger = ProxyLogger() if use_composite else None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Логувати градієнти та loss для Composite Score
            if use_composite and logger is not None:
                grad_norm = compute_grad_norm(model)
                logger.log_batch(loss.item(), grad_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
    
    # Повернути val_loss і metrics (якщо Composite Score)
    if use_composite and logger is not None:
        metrics = logger.compute_metrics(val_loss)
        return val_loss, metrics
    
    return val_loss, None

# ============================================
# Ініціалізація Composite Score (якщо використовується)
# ============================================

proxy_stats = ProxyStats() if USE_COMPOSITE_PROXY else None

# ============================================
# Функції повного тренування (для FULL_PIPELINE)
# ============================================

def train_model_full(model, train_loader, val_loader, device, optimizer_config, epochs=FULL_EPOCHS, model_name="model"):
    """Повне навчання моделі"""
    
    # Створити optimizer на основі конфігу
    optimizer_name = optimizer_config['optimizer']
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0
    }
    
    print(f"\n{'='*60}")
    print(f"ПОВНЕ ТРЕНУВАННЯ: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Вивід прогресу кожної епохи
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Зберегти найкращу модель
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            history['best_epoch'] = epoch
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Завершено! Найкращий Val Loss: {best_val_loss:.4f} (epoch {history['best_epoch']+1})")
    print(f"⏱️  Час: {total_time/60:.2f} хвилин")
    
    return best_val_loss, history

def compute_spearman_correlation(proxy_scores, final_losses):
    """Порахувати кореляцію Spearman між Composite Score та фінальним loss"""
    
    # Ранжування
    proxy_ranks = np.argsort(np.argsort(proxy_scores))
    final_ranks = np.argsort(np.argsort(final_losses))
    
    # Spearman correlation
    rho, pvalue = spearmanr(proxy_scores, final_losses)
    
    # Rank stability (чи співпадають ранги)
    rank_match = (proxy_ranks == final_ranks).sum() / len(proxy_ranks) * 100
    
    return {
        'rho': rho,
        'pvalue': pvalue,
        'proxy_ranks': proxy_ranks.tolist(),
        'final_ranks': final_ranks.tolist(),
        'rank_stability': rank_match
    }

# ============================================
# Optuna objective function
# ============================================

def objective(trial):
    """Функція оптимізації для Optuna"""
    
    # 1. СТРУКТУРНИЙ СИНТЕЗ
    num_blocks = trial.suggest_int('num_blocks', 2, 5)
    
    filters_list = []
    kernel_sizes = []
    for i in range(num_blocks):
        filters_list.append(trial.suggest_categorical(f'filters_{i}', [16, 32, 64, 128]))
        kernel_sizes.append(trial.suggest_categorical(f'kernel_{i}', [3, 5]))
    
    fc_size = trial.suggest_categorical('fc_size', [64, 128, 256])
    dropout = trial.suggest_categorical('dropout', [0.3, 0.5, 0.7])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu'])
    
    # 2. ПАРАМЕТРИЧНИЙ СИНТЕЗ
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01])
    batch_size = trial.suggest_categorical('batch_size', BATCH_SIZE_OPTIONS)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    weight_decay = trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3])
    
    # 3. Створити модель
    model = DynamicDetector(
        num_blocks=num_blocks,
        filters_list=filters_list,
        kernel_sizes=kernel_sizes,
        fc_size=fc_size,
        dropout=dropout,
        activation=activation
    ).to(device)
    
    # 4. Створити optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 5. Dataset
    train_dataset = SimpleDataset("dataset/train/images", "dataset/train/labels")
    # Завантажити повний val dataset для створення subset
    val_dataset_full = SimpleDataset("dataset/val/images", "dataset/val/labels", max_samples=-1)
    
    # Використати validation subset для Composite Score
    if USE_COMPOSITE_PROXY:
        # Створити детерміністичний val subset
        val_subset_file = 'results/val_subset_idx.npy'
        val_size = min(VAL_SUBSET, len(val_dataset_full))
        
        if Path(val_subset_file).exists():
            val_indices = np.load(val_subset_file)
            # Перевірити що індекси в межах
            val_indices = val_indices[val_indices < len(val_dataset_full)]
            if len(val_indices) < val_size:
                # Перегенерувати якщо не вистачає
                np.random.seed(SEED)
                val_indices = np.random.choice(len(val_dataset_full), val_size, replace=False)
                np.save(val_subset_file, val_indices)
        else:
            np.random.seed(SEED)
            val_indices = np.random.choice(len(val_dataset_full), val_size, replace=False)
            np.save(val_subset_file, val_indices)
        
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    else:
        val_dataset = val_dataset_full
    
    # num_workers=0 для стабільності MPS
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 6. Навчання
    try:
        val_loss, metrics = train_trial(model, train_loader, val_loader, device, optimizer, 
                                       epochs=EPOCHS_PER_TRIAL, use_composite=USE_COMPOSITE_PROXY)
    except Exception as e:
        print(f"⚠️  Trial failed: {e}")
        return float('inf')
    
    # 7. Обчислити objective (Composite Score або простий val_loss)
    if USE_COMPOSITE_PROXY and metrics is not None and proxy_stats is not None:
        # Обчислити Composite Score
        proxy_value, objective_type = compute_composite_score(metrics, proxy_stats)
        
        # Оновити статистику warmup
        if not proxy_stats.ready():
            proxy_stats.update(metrics)
        
        # Зберегти user attributes
        for key, value in metrics.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr('proxy_value', proxy_value)
        trial.set_user_attr('objective_type', objective_type)
        
        # Логувати в CSV
        proxy_stats.log_trial(trial.number, metrics, proxy_value, objective_type)
        
        return proxy_value
    
    return val_loss

# ============================================
# Запуск синтезу
# ============================================

if __name__ == "__main__":
    # ============================================
    # Запуск логування
    # ============================================
    
    # Перенаправити stdout в TeeLogger
    tee_logger.start()
    sys.stdout = tee_logger
    
    log_print(f"🚀 Експеримент запущено: {EXPERIMENT_START_TIME.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log_print(f"📝 Логи зберігаються у: {LOG_FILENAME}")
    
    # ============================================
    # Вивід параметрів
    # ============================================
    
    print("\n" + "="*60)
    print("УНІВЕРСАЛЬНИЙ СТРУКТУРНО-ПАРАМЕТРИЧНИЙ СИНТЕЗ")
    print("="*60)
    
    # Вивід режиму роботи
    mode_emoji = "🚀" if FULL_RUN_MODE else "⚡"
    mode_name = "ПОВНИЙ ПРОГОН" if FULL_RUN_MODE else "ШВИДКИЙ ТЕСТ"
    platform = "Google Colab" if IS_COLAB else "Локально"
    print(f"\n{mode_emoji} РЕЖИМ: {mode_name} ({platform})")
    
    print(f"\n⚙️  ПАРАМЕТРИ:")
    print(f"   Trials: {N_TRIALS}")
    print(f"   Timeout: {TIMEOUT}s ({TIMEOUT/60:.0f} min)")
    print(f"   Max samples: {MAX_SAMPLES}")
    print(f"   Image size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"   Epochs per trial: {EPOCHS_PER_TRIAL}")
    if FULL_PIPELINE:
        print(f"   Full training epochs: {FULL_EPOCHS}")
    
    # Вивід інформації про пристрій
    if gpu_memory:
        print(f"\n✅ GPU: {gpu_name}")
        print(f"   VRAM: {gpu_memory:.1f} GB")
    else:
        if device.type == "mps":
            print(f"\n✅ GPU: {gpu_name}")
        else:
            print(f"\n⚠️  {gpu_name}")
    print(f"   Device: {device}")
    print(f"   Batch sizes: {BATCH_SIZE_OPTIONS}")
    
    print("\n" + "="*60)
    print("ЗАПУСК СИНТЕЗУ")
    print("="*60)
    
    if USE_COMPOSITE_PROXY:
        print(f"\n📊 Phase 1: Warmup калібрація (trials 1-{N_WARMUP})")
        print(f"   → Збір метрик для robust z-нормалізації")
        print(f"\n🧠 Phase 2: Bayesian Optimization (trials {N_WARMUP+1}-{N_TRIALS})")
        print(f"   → Оптимізація за Composite Score")
    
    # ============================================
    # Спробувати відновити Study з checkpoint (якщо є)
    # ============================================
    
    # Спробувати завантажити checkpoint з Drive (на Colab)
    if IS_COLAB:
        try:
            from google.colab import drive
            import shutil
            
            # Drive вже змонтовано раніше
            drive_checkpoint = '/content/drive/MyDrive/Studying/Experiments/Composite_score_nas/checkpoint'
            
            if Path(drive_checkpoint).exists():
                print("\n🔄 Знайдено checkpoint на Google Drive! Копіювання...")
                
                # Створити локальну папку results
                Path('results').mkdir(exist_ok=True)
                
                # Скопіювати всі файли checkpoint
                for file in Path(drive_checkpoint).glob('*'):
                    if file.is_file():
                        shutil.copy2(file, 'results/')
                
                print(f"   ✅ Checkpoint скопійовано з Drive")
        except Exception as e:
            print(f"   ⚠️  Не вдалося завантажити checkpoint з Drive: {e}")
    
    study_checkpoint = Path('results/optuna_study.pkl')
    study_resumed = False
    
    if study_checkpoint.exists():
        try:
            print("\n🔄 Знайдено Optuna checkpoint! Спроба відновлення...")
            import pickle
            
            with open(study_checkpoint, 'rb') as f:
                study = pickle.load(f)
            
            study_resumed = True
            print(f"   ✅ Відновлено {len(study.trials)} trials")
            print(f"   Продовжимо з trial #{len(study.trials) + 1}")
        except Exception as e:
            print(f"   ⚠️  Не вдалося відновити study: {e}")
            study_resumed = False
    
    # Створити новий study якщо не відновили
    if not study_resumed:
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=SEED)
        )
    
    # Callback для інформативного виводу
    def progress_callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_trial = study.best_trial
            phase = "Warmup" if trial.number < N_WARMUP else "Composite Score"
            status = "🔥" if trial.value == best_trial.value else "✓"
            
            log_print(f"{status} Trial {trial.number + 1}/{N_TRIALS} [{phase}]:")
            print(f"   Поточний Score: {trial.value:.4f}")
            print(f"   Найкращий Score: {best_trial.value:.4f} (Trial #{best_trial.number + 1})")
            print(f"   Архітектура: {trial.params['num_blocks']} блоків, "
                  f"Act={trial.params['activation']}, Opt={trial.params['optimizer']}")
    
    # Запустити оптимізацію
    start_time = time.time()
    
    # Якщо відновили - запустити менше trials
    remaining_trials = N_TRIALS - len(study.trials) if study_resumed else N_TRIALS
    
    if remaining_trials > 0:
        # Вимкнути verbose логи Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=TIMEOUT,
            callbacks=[progress_callback],
            show_progress_bar=False  # Вимкнути стандартний прогрес-бар
        )
    else:
        print(f"\n⏭️  Фазу синтезу пропущено - вже є {len(study.trials)} trials")
        print(f"   Переходжу одразу до повного тренування")
    
    synthesis_time = time.time() - start_time
    
    # Зберегти study checkpoint
    try:
        import pickle
        Path('results').mkdir(exist_ok=True)
        with open('results/optuna_study.pkl', 'wb') as f:
            pickle.dump(study, f)
        print(f"\n💾 Optuna study збережено")
    except Exception as e:
        print(f"\n⚠️  Не вдалося зберегти study: {e}")
    
    # ============================================
    # Результати
    # ============================================
    
    print("\n" + "="*60)
    if remaining_trials == 0:
        print("ЗАВАНТАЖЕНІ РЕЗУЛЬТАТИ СИНТЕЗУ")
    else:
        print("РЕЗУЛЬТАТИ СИНТЕЗУ")
    print("="*60)
        
    if synthesis_time > 0:
        print(f"\n⏱️  Час синтезу: {synthesis_time:.2f}s ({synthesis_time/60:.2f} min)")
    print(f"🔍 Перевірено архітектур: {len(study.trials)}")
    print(f"✅ Успішних: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    print(f"\n🏆 НАЙКРАЩА АРХІТЕКТУРА:")
    score_name = "Composite Score" if USE_COMPOSITE_PROXY and proxy_stats.ready() else "Val Loss"
    print(f"   {score_name}: {study.best_value:.4f} (Trial {study.best_trial.number + 1})")
    print(f"   {'   ↓ менше = краще (мінімізуємо)' if study.best_value < 0 else ''}")
    
    print(f"\n📐 Структура:")
    best_params = study.best_params
    num_blocks = best_params['num_blocks']
    print(f"   Кількість conv блоків: {num_blocks}")
    for i in range(num_blocks):
        print(f"   Block {i+1}: {best_params[f'filters_{i}']} фільтрів, kernel {best_params[f'kernel_{i}']}×{best_params[f'kernel_{i}']}")
    print(f"   FC layer: {best_params['fc_size']}")
    print(f"   Dropout: {best_params['dropout']}")
    print(f"   Activation: {best_params['activation']}")
    
    print(f"\n⚙️  Гіперпараметри:")
    print(f"   Optimizer: {best_params['optimizer']}")
    print(f"   Learning rate: {best_params['lr']}")
    print(f"   Weight decay: {best_params['weight_decay']}")
    print(f"   Batch size: {best_params['batch_size']}")
        
    # ============================================
    # TOP 5
    # ============================================
    
    print("\n" + "="*60)
    print(f"TOP 5 АРХІТЕКТУР (за {score_name}):")
    print("="*60)
    
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))
    for i, trial in enumerate(sorted_trials[:5]):
        if trial.value:
            indicator = "🔥" if i == 0 else ("⭐" if i < 3 else "✓")
            print(f"\n{indicator} #{i+1} Score: {trial.value:.4f} (Trial {trial.number + 1})")
            print(f"    Архітектура: {trial.params['num_blocks']} блоків, "
                  f"FC={trial.params['fc_size']}, "
                  f"{trial.params['activation']}")
            print(f"    Тренування: {trial.params['optimizer']}, "
                  f"LR={trial.params['lr']}, "
                  f"BS={trial.params['batch_size']}")
        
    
    # ============================================
    # Повне тренування ВСІХ моделей (якщо FULL_PIPELINE=True)
    # ============================================
    
    if FULL_PIPELINE:
        print("\n" + "="*60)
        print(f"ПОВНЕ ТРЕНУВАННЯ ВСІХ {len(study.trials)} МОДЕЛЕЙ")
        print("="*60)
        
        # Визначити batch size для повного тренування
        full_batch_size = FULL_BATCH_SIZE
        if full_batch_size is None:
            full_batch_size = 32 if device.type == "cuda" else 16
        
        print(f"\n⚙️  Параметри повного тренування:")
        print(f"   Epochs: {FULL_EPOCHS}")
        print(f"   Batch size: {full_batch_size}")
        print(f"   Device: {device}")
        
        # ============================================
        # Спроба відновити з checkpoint (якщо є)
        # ============================================
        
        checkpoint_path = Path('results/checkpoint.json')
        resumed = False
        full_training_results = []
        
        if checkpoint_path.exists():
            try:
                print("\n🔄 Знайдено checkpoint! Спроба відновлення...")
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                # Перевірити чи це той самий експеримент
                if (checkpoint_data['synthesis']['n_trials'] == N_TRIALS and
                    checkpoint_data['synthesis']['seed'] == SEED):
                    
                    full_training_results = checkpoint_data['training']['results']
                    resumed = True
                    print(f"   ✅ Відновлено {len(full_training_results)} моделей")
                    print(f"   Продовжимо з моделі #{len(full_training_results) + 1}")
                else:
                    print("   ⚠️  Checkpoint від іншого експерименту - ігноруємо")
            except Exception as e:
                print(f"   ⚠️  Не вдалося відновити checkpoint: {e}")
        
        # Завантажити датасет для повного тренування
        print(f"\n📦 Завантаження датасету для повного тренування...")
        full_train_dataset = SimpleDataset("dataset/train/images", "dataset/train/labels", img_size=IMG_SIZE, max_samples=FULL_MAX_SAMPLES)
        full_val_dataset = SimpleDataset("dataset/val/images", "dataset/val/labels", img_size=IMG_SIZE, max_samples=-1)
        print(f"   Train: {len(full_train_dataset)} зображень")
        print(f"   Val: {len(full_val_dataset)} зображень")
        
        # Підготувати всі моделі для тренування (відсортовані по Composite Score)
        all_models_to_train = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_models_to_train.append({
                    'trial_number': trial.number,
                    'proxy_value': trial.value,
                    'params': trial.params,
                    'user_attrs': trial.user_attrs
                })
        
        # Відсортувати по Composite Score (ascending - кращі мають менший score)
        all_models_to_train.sort(key=lambda x: x['proxy_value'])
        
        print(f"\n📋 Буде натреновано {len(all_models_to_train)} моделей")
        print(f"   Composite Score range: {all_models_to_train[0]['proxy_value']:.4f} (найкращий) → {all_models_to_train[-1]['proxy_value']:.4f} (найгірший)")
        
        # Тренувати КОЖНУ модель
        for idx, model_info in enumerate(all_models_to_train, 1):
            # Пропустити якщо вже натреновано
            if resumed and idx <= len(full_training_results):
                continue
            
            params = model_info['params']
            
            print(f"\n{'='*60}")
            log_print(f"МОДЕЛЬ #{idx}/{len(all_models_to_train)} (Trial {model_info['trial_number']})")
            print(f"{'='*60}")
            log_print(f"   Composite Score (синтез): {model_info['proxy_value']:.4f}")
            print(f"   Proxy rank: #{idx}")
            print(f"   Структура: {params['num_blocks']} блоків")
            
            # Витягти filters та kernels
            filters_list = [params[f'filters_{i}'] for i in range(params['num_blocks'])]
            kernel_sizes = [params[f'kernel_{i}'] for i in range(params['num_blocks'])]
            
            print(f"   Фільтри: {filters_list}")
            print(f"   Activation: {params['activation']}")
            
            # Створити модель
            model = DynamicDetector(
                num_blocks=params['num_blocks'],
                filters_list=filters_list,
                kernel_sizes=kernel_sizes,
                fc_size=params['fc_size'],
                dropout=params['dropout'],
                activation=params['activation']
            ).to(device)
            
            # DataLoaders
            train_loader = DataLoader(
                full_train_dataset, 
                batch_size=full_batch_size,
                shuffle=True, 
                num_workers=0
            )
            val_loader = DataLoader(
                full_val_dataset, 
                batch_size=full_batch_size,
                shuffle=False, 
                num_workers=0
            )
            
            # Optimizer config
            optimizer_config = {
                'optimizer': params['optimizer'],
                'lr': params['lr'],
                'weight_decay': params['weight_decay']
            }
            
            # Повне тренування
            model_name = f"model_trial{model_info['trial_number']}_rank{idx}_seed{SEED}"
            final_val_loss, history = train_model_full(
                model, 
                train_loader, 
                val_loader, 
                device, 
                optimizer_config=optimizer_config,
                epochs=FULL_EPOCHS,
                model_name=model_name
            )
            
            # Зберегти результат
            full_training_results.append({
                'trial_number': model_info['trial_number'],
                'proxy_rank': idx,
                'proxy_score': model_info['proxy_value'],
                'final_val_loss': final_val_loss,
                'improvement': model_info['proxy_value'] - final_val_loss,
                'history': history,
                'params': params
            })
            
            # ============================================
            # Checkpoint збереження (після кожної моделі)
            # ============================================
            try:
                print(f"\n💾 Checkpoint: збереження проміжних результатів...")
                
                # Зберегти проміжні результати
                checkpoint_results = {
                    'synthesis': {
                        'n_trials': N_TRIALS,
                        'n_completed': len(all_models_to_train),
                        'use_composite': USE_COMPOSITE_PROXY,
                        'seed': SEED
                    },
                    'training': {
                        'epochs': FULL_EPOCHS,
                        'batch_size': full_batch_size,
                        'n_models_trained': len(full_training_results),
                        'results': full_training_results,
                        'checkpoint': f'Model {idx}/{len(all_models_to_train)}'
                    }
                }
                
                # Зберегти локально (завжди)
                with open('results/checkpoint.json', 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_results, f, indent=2, ensure_ascii=False)
                
                # Зберегти Optuna study (завжди)
                import pickle
                with open('results/optuna_study.pkl', 'wb') as f:
                    pickle.dump(study, f)
                
                print(f"   ✅ Checkpoint збережено локально (модель {idx}/{len(all_models_to_train)})")
                
                # Додатково на Drive (тільки Colab)
                if IS_COLAB:
                    from google.colab import drive
                    import shutil
                    
                    drive.mount('/content/drive', force_remount=False)
                    drive_results_dir = '/content/drive/MyDrive/Studying/Experiments/Composite_score_nas'
                    Path(drive_results_dir).mkdir(parents=True, exist_ok=True)
                    
                    shutil.copytree('results', f'{drive_results_dir}/checkpoint', dirs_exist_ok=True)
                    print(f"   ✅ Checkpoint також збережено на Drive")
            except Exception as e:
                print(f"   ⚠️  Помилка checkpoint: {e}")
        
        # ============================================
        # Аналіз кореляції Composite Score → Final Loss
        # ============================================
        
        print("\n" + "="*60)
        print("АНАЛІЗ КОРЕЛЯЦІЇ COMPOSITE SCORE → FINAL LOSS")
        print("="*60)
        
        proxy_scores = [r['proxy_score'] for r in full_training_results]
        final_losses = [r['final_val_loss'] for r in full_training_results]
        
        correlation_stats = compute_spearman_correlation(proxy_scores, final_losses)
        
        print(f"\n📊 Spearman кореляція:")
        print(f"   ρ = {correlation_stats['rho']:.4f}")
        print(f"   p-value = {correlation_stats['pvalue']:.4f}")
        if correlation_stats['pvalue'] < 0.05:
            print(f"   ✅ Статистично значуща кореляція!")
        print(f"   Rank stability: {correlation_stats['rank_stability']:.1f}%")
        
        # Відсортувати результати по фінальному loss для виводу топ-10
        sorted_by_final = sorted(full_training_results, key=lambda x: x['final_val_loss'])
        
        print(f"\n📈 ТОП-10 за фінальним loss:")
        for i, result in enumerate(sorted_by_final[:10], 1):
            final_rank = i
            proxy_rank = result['proxy_rank']
            rank_diff = abs(proxy_rank - final_rank)
            rank_indicator = "✅" if rank_diff <= 5 else ("⚠️" if rank_diff <= 10 else "❌")
            
            print(f"\n#{i} (Trial {result['trial_number']}):")
            print(f"   Final Loss: {result['final_val_loss']:.4f}")
            print(f"   Composite Score: {result['proxy_score']:.4f} (був rank #{proxy_rank}) {rank_indicator}")
            print(f"   Rank diff: {rank_diff}")
        
        # Зберегти повні результати
        results_filename = 'results/synthesis_results.json'
        full_results = {
            'synthesis': {
                'n_trials': N_TRIALS,
                'n_completed': len(all_models_to_train),
                'use_composite': USE_COMPOSITE_PROXY,
                'seed': SEED
            },
            'training': {
                'epochs': FULL_EPOCHS,
                'batch_size': full_batch_size,
                'n_models_trained': len(full_training_results),
                'results': full_training_results
            },
            'correlation': correlation_stats
        }
        
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Повні результати збережено в: {results_filename}")
        
        # Збереження на Google Drive (якщо Colab)
        try:
            from google.colab import drive, files
            import shutil
            
            # Спробувати зберегти на Google Drive
            try:
                drive.mount('/content/drive', force_remount=False)
                drive_results_dir = '/content/drive/MyDrive/Studying/Experiments/Composite_score_nas'
                Path(drive_results_dir).mkdir(parents=True, exist_ok=True)
                
                # Скопіювати всю папку results
                shutil.copytree('results', f'{drive_results_dir}/results_full', dirs_exist_ok=True)
                print(f"\n☁️  Повні результати збережено на Google Drive: {drive_results_dir}/results_full/")
            except Exception as e:
                print(f"\n⚠️  Не вдалося зберегти на Drive: {e}")
            
            # Завантажити ключові файли
            print("\n📥 Завантажую ключові результати...")
            try:
                files.download('results/synthesis_results.json')
                files.download('results/trials_proxy_metrics.csv')
                print("✅ Ключові файли завантажено!")
            except Exception as e:
                print(f"⚠️  Помилка завантаження: {e}")
        except ImportError:
            pass  # Не на Colab
    
    elif not FULL_PIPELINE:
        # Зберегти результати синтезу (без повного тренування)
        results_filename = 'results/synthesis_results.json'
        synthesis_only_results = {
            'synthesis': {
                'n_trials': N_TRIALS,
                'n_completed': len(study.trials),
                'use_composite': USE_COMPOSITE_PROXY,
                'seed': SEED
            },
            'note': 'Тільки результати синтезу. Для повного тренування встановіть FULL_PIPELINE=True'
        }
        
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(synthesis_only_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результати синтезу збережено в: {results_filename}")
        print(f"\n📁 Для повного тренування встановіть FULL_PIPELINE=True у скрипті")
        
        # Збереження результатів на Colab
        try:
            from google.colab import drive, files
            import shutil
            
            # Спробувати зберегти на Google Drive
            try:
                drive.mount('/content/drive', force_remount=False)
                drive_results_dir = '/content/drive/MyDrive/Studying/Experiments/Composite_score_nas'
                Path(drive_results_dir).mkdir(parents=True, exist_ok=True)
                
                # Скопіювати всю папку results (тільки синтез, без повного тренування)
                shutil.copytree('results', f'{drive_results_dir}/results_only_synthesis', dirs_exist_ok=True)
                print(f"\n☁️  Результати синтезу збережено на Google Drive: {drive_results_dir}/results_only_synthesis/")
            except Exception as e:
                print(f"\n⚠️  Не вдалося зберегти на Drive: {e}")
            
            # Завантажити synthesis_results.json
            print("\n📥 Завантажую synthesis_results.json...")
            files.download('results/synthesis_results.json')
            print("✅ Файл завантажено!")
        except ImportError:
            pass  # Не на Colab
    
    # Завершення
    print("\n" + "="*60)
    print("✅ СИНТЕЗ ЗАВЕРШЕНО!")
    print("="*60)
