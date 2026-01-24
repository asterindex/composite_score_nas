"""
Повне навчання топ-3 архітектур після синтезу
Автоматичне завантаження конфігурацій з Optuna study

Автор: Анатолій Кот
Дата: 2026-01-24
"""

import os
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Імпорт з synthesis_universal (той самий модуль src/)
from synthesis_universal import (
    DynamicDetector, get_dataloaders, evaluate, log_print,
    set_seed, DEVICE, DEVICE_NAME, RESULTS_DIR, CHECKPOINT_FILE,
    LOG_FILE, setup_logging
)


# ============================================================================
# КОНФІГУРАЦІЯ ПОВНОГО НАВЧАННЯ
# ============================================================================

FULL_EPOCHS = 25
TOP_K = 3
SEED = 42

MODELS_DIR = RESULTS_DIR / 'trained_models'
FINAL_RESULTS_FILE = RESULTS_DIR / 'final_results.json'


# ============================================================================
# ПОВНЕ НАВЧАННЯ
# ============================================================================

def train_full_model(config: Dict, trial_number: int, epochs: int = FULL_EPOCHS) -> Dict:
    """
    Повне навчання моделі з відстеженням всіх метрик
    
    Args:
        config: Конфігурація архітектури та гіперпараметрів
        trial_number: Номер trial з Optuna
        epochs: Кількість епох навчання
    
    Returns:
        Словник з метриками навчання
    """
    
    log_print(f"\n{'='*60}")
    log_print(f"🚀 Повне навчання Trial #{trial_number}")
    log_print(f"{'='*60}")
    
    # Налаштування
    set_seed(SEED)
    
    # Параметри архітектури
    n_blocks = config['n_blocks']
    model_config = {
        'n_blocks': n_blocks,
        'filter_sizes': [config[f'filter_{i}'] for i in range(n_blocks)],
        'kernel_sizes': [config[f'kernel_{i}'] for i in range(n_blocks)],
        'fc_size': config['fc_size'],
        'dropout': config['dropout'],
        'activation': config['activation']
    }
    
    # Гіперпараметри
    optimizer_name = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    
    log_print(f"🏗️  Architecture: {n_blocks} blocks")
    log_print(f"   Filters: {model_config['filter_sizes']}")
    log_print(f"   Kernels: {model_config['kernel_sizes']}")
    log_print(f"   FC: {model_config['fc_size']}, Dropout: {model_config['dropout']}")
    log_print(f"   Activation: {model_config['activation']}")
    log_print(f"⚙️  Optimizer: {optimizer_name.upper()} (LR={lr}, WD={weight_decay})")
    log_print(f"📦 Batch size: {batch_size}, Epochs: {epochs}")
    
    # Створення моделі
    model = DynamicDetector(model_config).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Scheduler (опціонально)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Завантаження даних
    train_loader, val_loader = get_dataloaders(batch_size)
    
    # Історія навчання
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    start_time = time.time()
    
    # Основний цикл навчання
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_losses = []
        
        for images, targets in train_loader:
            images = images.to(DEVICE)
            labels = targets['labels'][:, 0] if targets['labels'].size(1) > 0 else torch.zeros(images.size(0), dtype=torch.long)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        # Validation
        val_loss = evaluate(model, val_loader, criterion)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Оновлення історії
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            
            # Збереження найкращої моделі
            model_path = MODELS_DIR / f'trial_{trial_number}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, model_path)
        
        epoch_time = time.time() - epoch_start
        log_print(f"   Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} ({epoch_time:.1f}s)")
    
    total_time = time.time() - start_time
    log_print(f"\n✅ Навчання завершено за {total_time/60:.2f} хв")
    log_print(f"🏆 Найкращий результат: val_loss={history['best_val_loss']:.4f} (epoch {history['best_epoch']+1})")
    
    # Фінальні метрики
    final_metrics = {
        'trial_number': trial_number,
        'final_val_loss': history['val_loss'][-1],
        'best_val_loss': history['best_val_loss'],
        'best_epoch': history['best_epoch'],
        'training_time_minutes': total_time / 60,
        'history': history
    }
    
    return final_metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Основний пайплайн повного навчання топ-3"""
    
    # Setup
    setup_logging()
    log_print(f"🎯 Повне навчання топ-{TOP_K} архітектур")
    log_print(f"   FULL_EPOCHS: {FULL_EPOCHS}")
    log_print(f"   DEVICE: {DEVICE_NAME}")
    log_print(f"   SEED: {SEED}")
    
    # Створення директорії для моделей
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Завантаження Optuna study
    if not CHECKPOINT_FILE.exists():
        log_print(f"❌ Не знайдено файл study: {CHECKPOINT_FILE}")
        log_print(f"   Спочатку запустіть synthesis_universal.py")
        return
    
    log_print(f"📂 Завантаження Optuna study з {CHECKPOINT_FILE}")
    with open(CHECKPOINT_FILE, 'rb') as f:
        study = pickle.load(f)
    
    log_print(f"✅ Завантажено {len(study.trials)} trials")
    
    # Відбір топ-K
    trials_sorted = sorted(study.trials, key=lambda t: t.value)
    top_trials = trials_sorted[:TOP_K]
    
    log_print(f"\n📊 Топ-{TOP_K} архітектури для повного навчання:")
    for i, trial in enumerate(top_trials, 1):
        log_print(f"   #{i} Trial {trial.number}: proxy={trial.value:.4f}")
    
    # Повне навчання кожної моделі
    all_results = []
    
    for i, trial in enumerate(top_trials, 1):
        log_print(f"\n{'#'*60}")
        log_print(f"# МОДЕЛЬ {i}/{TOP_K}")
        log_print(f"{'#'*60}")
        
        try:
            metrics = train_full_model(trial.params, trial.number, epochs=FULL_EPOCHS)
            metrics['rank'] = i
            metrics['proxy_value'] = trial.value
            all_results.append(metrics)
        except Exception as e:
            log_print(f"❌ Помилка при навчанні Trial #{trial.number}: {e}")
            continue
    
    # Збереження фінальних результатів
    final_results = {
        'models': all_results,
        'metadata': {
            'top_k': TOP_K,
            'full_epochs': FULL_EPOCHS,
            'device': DEVICE_NAME,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    }
    
    with open(FINAL_RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Підсумок
    log_print(f"\n{'='*60}")
    log_print("🏁 ФІНАЛЬНІ РЕЗУЛЬТАТИ")
    log_print(f"{'='*60}")
    
    # Сортування за best_val_loss
    all_results_sorted = sorted(all_results, key=lambda r: r['best_val_loss'])
    
    for i, result in enumerate(all_results_sorted, 1):
        log_print(f"\n🥇 Місце #{i}: Trial {result['trial_number']}")
        log_print(f"   Proxy value: {result['proxy_value']:.4f}")
        log_print(f"   Best val loss: {result['best_val_loss']:.4f} (epoch {result['best_epoch']+1})")
        log_print(f"   Training time: {result['training_time_minutes']:.2f} хв")
    
    log_print(f"\n✅ Всі моделі збережено у: {MODELS_DIR}")
    log_print(f"📊 Результати: {FINAL_RESULTS_FILE}")
    
    if LOG_FILE:
        log_print(f"📝 Лог: {LOG_FILE}")


if __name__ == '__main__':
    main()
