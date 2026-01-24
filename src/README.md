# Source Code - Composite Score NAS

Код для Detection Stability Score (DSS) експерименту.

## Модулі

### `synthesis_universal.py` (777 рядків)
Основний пайплайн синтезу архітектур через Bayesian Optimization.

**Ключові компоненти:**
- `DynamicDetector` - динамічна побудова CNN архітектури
- `ProxyStatistics` - robust z-нормалізація для DSS
- `compute_composite_score()` - обчислення DSS з 7 компонентів
- `objective()` - Optuna objective function з warmup
- `main()` - повний цикл синтезу (30 trials)

**Вхід:** Параметри через environment variables (встановлюються main.py)
**Вихід:** `output/optuna_study.pkl`, `proxy_stats.json`, `synthesis_results.json`

### `train_top3_models.py` (339 рядків)
Повне навчання топ-3 архітектур після синтезу.

**Ключові функції:**
- `load_top3_configs()` - завантаження конфігурацій з Optuna study
- `full_train()` - повне навчання моделі (30+ епох)
- `train_best_models()` - тренування всіх топ-3

**Вхід:** `output/optuna_study.pkl`
**Вихід:** `output/trained_models/trial_*_best.pth`, `final_results.json`

### `analyze_results.py` (180 рядків)
Аналіз та візуалізація результатів експерименту.

**Ключові функції:**
- `plot_convergence()` - графік DSS convergence
- `analyze_hyperparams()` - розподіл гіперпараметрів у топ-моделях
- `compute_correlation()` - кореляція DSS з фінальною якістю

**Вхід:** `output/optuna_study.pkl`
**Вихід:** `output/convergence.png`, `analysis_report.json`

### `dataset_utils.py` (120 рядків)
Утиліти для роботи з VisDrone2019-DET датасетом.

**Ключові класи:**
- `VisDroneDataset` - PyTorch Dataset для VisDrone
- `load_visdrone_data()` - завантаження та preprocessing

**Формат:** Annotations у форматі `<x,y,w,h,score,class,...>`

## Використання

### Через main.py (рекомендовано)
```bash
python main.py --mode synthesis   # Запускає synthesis_universal.py
python main.py --mode train-top3  # Запускає train_top3_models.py
python main.py --mode analyze     # Запускає analyze_results.py
```

### Прямий запуск (для debug)
```bash
cd src/
python synthesis_universal.py    # Потрібно встановити env vars
```

## Environment Variables

Параметри встановлюються автоматично через `main.py`:

```bash
NAS_N_TRIALS=30         # Кількість trials
NAS_N_WARMUP=10         # Warmup trials
NAS_EPOCHS_PER_TRIAL=1  # Епохи на trial
NAS_MAX_SAMPLES=700     # Train samples
NAS_VAL_SUBSET=200      # Val samples
NAS_SEED=42             # Random seed
NAS_OUTPUT_DIR=output   # Папка результатів
```

## Залежності між модулями

```
synthesis_universal.py
       ↓
   (використовує)
       ↓
dataset_utils.py

train_top3_models.py
       ↓
   (імпортує)
       ↓
synthesis_universal.py → dataset_utils.py

analyze_results.py
       ↓
   (читає)
       ↓
output/optuna_study.pkl (створений synthesis_universal.py)
```

## Конфігурація

Всі константи визначені на початку `synthesis_universal.py`:

```python
SEED = int(os.getenv('NAS_SEED', '42'))
N_TRIALS = int(os.getenv('NAS_N_TRIALS', '30'))
N_WARMUP = int(os.getenv('NAS_N_WARMUP', '10'))
EPOCHS_PER_TRIAL = int(os.getenv('NAS_EPOCHS_PER_TRIAL', '1'))
MAX_SAMPLES = int(os.getenv('NAS_MAX_SAMPLES', '700'))
VAL_SUBSET = int(os.getenv('NAS_VAL_SUBSET', '200'))
```

## Додавання нових модулів

1. Створіть новий файл у `src/`
2. Додайте імпорт у `src/__init__.py`
3. Додайте режим у `main.py` (якщо потрібно)

Приклад:
```python
# src/new_module.py
from synthesis_universal import RESULTS_DIR, DEVICE

def my_function():
    # Ваш код
    pass
```

```python
# src/__init__.py
from .new_module import *
```

## Logування

Всі модулі використовують `log_print()` з `synthesis_universal.py`:
- Автоматичне додавання UTC timestamps
- Запис у файл `output/experiment_YYYYMMDD_HHMMSS.log`
- Синхронний вивід у консоль та файл

## Device Support

Автоматичне визначення прискорювача:
1. CUDA (якщо доступна)
2. MPS (Apple Silicon)
3. CPU (fallback)

```python
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
```
