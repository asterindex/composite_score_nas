# 📝 CHANGELOG

## [2026-01-09] Resume & Checkpoint System

### ✨ Додано

#### 🔄 Автоматичне відновлення з checkpoint
- **Optuna study resume**: Автоматично відновлює trials з `optuna_study.pkl`
- **Training resume**: Продовжує тренування з останньої збереженої моделі
- **ProxyStats resume**: Завантажує калібровану статистику з `proxy_stats.json`
- **Val subset resume**: Використовує той самий validation subset з `val_subset_idx.npy`

#### 💾 Система checkpoint'ів (тільки на Colab)
1. **Кожні 5 моделей (~30-60 хв)**:
   - Зберігає проміжні результати тренування
   - Локація: `MyDrive/Studying/composite_score_nas_results/checkpoint/`
   - Включає: `checkpoint.json`, `optuna_study.pkl`, всі результати

2. **В кінці (~12 год)**:
   - Повні результати експерименту
   - Локація: `MyDrive/Studying/composite_score_nas_results/results_full/`

#### 🧠 Розумна логіка відновлення
- Автоматично визначає чи є checkpoint
- Перевіряє `SEED` та `N_TRIALS` (ігнорує якщо не співпадають)
- Пропускає вже натреновані моделі
- Продовжує з останньої позиції

### 🛡️ Захист від втрати даних

| Сценарій | Втрата |
|----------|--------|
| Відключення на моделі #23 | 3 моделі (останній checkpoint на #20) |
| Відключення на моделі #49 | 4 моделі (останній checkpoint на #45) |
| Успішне завершення | 0 - все збережено ✅ |

**Максимальна втрата:** ~1 година роботи (5 моделей × ~12 хв)

### 📚 Документація

Додано файли:
- `RESUME_GUIDE.md` - детальна інструкція по resume
- Оновлено `COLAB_CHECKLIST.md` - додано інфо про resume
- Оновлено `README.md` - додано розділ про checkpoint'и

### 🔧 Технічні деталі

**Файли checkpoint:**
- `results/checkpoint.json` - JSON з проміжними результатами
- `results/optuna_study.pkl` - Pickled Optuna study object
- `results/synthesis_only.json` - Результати синтезу (топ-3)

**Перевірки при resume:**
```python
if checkpoint_data['synthesis']['n_trials'] == N_TRIALS and
   checkpoint_data['synthesis']['seed'] == SEED:
    # Resume OK ✅
```

**Логіка пропуску моделей:**
```python
for idx, model_info in enumerate(all_models_to_train, 1):
    if resumed and idx <= len(full_training_results):
        continue  # Пропустити вже натреновану
```

---

## [2026-01-09] Project Cleanup & Reorganization

### 🗂️ Структура проекту

#### Видалено
- `yolo11n_epoch*.pt` - старі чекпоінти YOLO
- `config_colab.py`, `config_local.py` - замінено на auto-detect
- `top3_models.json` - інтегровано в `synthesis_results.json`
- Мертвий код multi-start аналізу

#### Реорганізовано
```
composite_score_nas/
├── bayesian_optimization/
│   ├── results/              # Всі результати синтезу
│   │   ├── synthesis_results.json
│   │   ├── trials_proxy_metrics.csv
│   │   ├── proxy_stats.json
│   │   ├── val_subset_idx.npy
│   │   ├── checkpoint.json    # NEW!
│   │   └── optuna_study.pkl   # NEW!
│   └── synthesis_universal.py
├── correlation_experiment/
│   └── results/              # Всі результати кореляції
│       ├── all_metrics_per_epoch.csv
│       └── correlation_analysis.png
└── dataset/
    ├── train/
    └── val/
```

### ⚙️ Автоматичне визначення середовища

**Замість:**
```python
from config_local import *  # or config_colab
```

**Тепер:**
```python
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

FULL_RUN_MODE = IS_COLAB  # Auto-detect
```

**Вивід:**
```
🚀 РЕЖИМ: ПОВНИЙ ПРОГОН (Google Colab)
⚡ РЕЖИМ: ШВИДКИЙ ТЕСТ (Локально)
```

### 🔧 MPS Stability Fixes

**Проблема:** `Error: command buffer exited with error status`

**Рішення:**
- `num_workers=0` для всіх DataLoader (single-threaded)
- Менші batch sizes: `[8, 16]` замість `[16, 32, 64]`
- Менше samples: `300` замість `500`

### 📊 Покращений logging

**Було:** Тільки epoch 1 та кожні 5 епох

**Тепер:** Кожна епоха (для моніторингу прогресу)
```python
print(f"Epoch {epoch + 1:2d}/{epochs} | "
      f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
      f"Time: {epoch_time:.1f}s")
```

---

## Наступні кроки (можливі)

- [ ] Додати email notifications після завершення (якщо Colab)
- [ ] Візуалізація прогресу в Tensorboard
- [ ] Adaptive checkpoint frequency (частіше якщо модель довше тренується)
- [ ] Multi-device support (розподілене тренування)
- [ ] Compression checkpoint'ів для економії Drive space

---

**Версія:** 1.1.0  
**Дата:** 2026-01-09  
**Автор:** Анатолій Кот (Anatoly Kot)
