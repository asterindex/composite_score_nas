# Приклади використання main.py

## Базові команди

### 1. Інформація про проект
```bash
python main.py --mode info
```

### 2. Довідка по параметрах
```bash
python main.py --help
```

## Режими роботи

### Synthesis - Синтез архітектур

**Повний експеримент (30 trials):**
```bash
python main.py --mode synthesis
```

**Швидкий тест (5 trials):**
```bash
python main.py --mode synthesis --trials 5 --quick
```

**Налаштування параметрів:**
```bash
# 50 trials з більшим датасетом
python main.py --mode synthesis --trials 50 --samples 1000 --val-samples 300

# Зміна warmup фази
python main.py --mode synthesis --warmup 15

# Більше епох для кожного trial
python main.py --mode synthesis --epochs 2

# Інший seed
python main.py --mode synthesis --seed 123
```

**Продовження експерименту:**
```bash
python main.py --mode synthesis --resume
```

**Збереження в іншу папку:**
```bash
python main.py --mode synthesis --output-dir experiments/run1
```

### Train-top3 - Повне тренування

**Тренування топ-3 архітектур:**
```bash
python main.py --mode train-top3
```

**З альтернативної папки:**
```bash
python main.py --mode train-top3 --output-dir experiments/run1
```

### Analyze - Аналіз результатів

**Аналіз та візуалізація:**
```bash
python main.py --mode analyze
```

**З детальним виводом:**
```bash
python main.py --mode analyze --verbose
```

### Clean - Очищення результатів

**Інтерактивне видалення:**
```bash
python main.py --mode clean
```

**Автоматичне видалення (без підтвердження):**
```bash
python main.py --mode clean --confirm
```

**Очищення конкретної папки:**
```bash
python main.py --mode clean --output-dir experiments/run1 --confirm
```

## Типові сценарії

### Сценарій 1: Швидке тестування

```bash
# 1. Швидкий тест (3-5 хвилин)
python main.py --mode synthesis --trials 5 --quick

# 2. Перевірка результатів
python main.py --mode analyze

# 3. Очищення
python main.py --mode clean --confirm
```

### Сценарій 2: Повний експеримент

```bash
# 1. Повний синтез (15-18 хвилин)
python main.py --mode synthesis

# 2. Аналіз результатів
python main.py --mode analyze

# 3. Тренування топ-3 (50+ хвилин)
python main.py --mode train-top3
```

### Сценарій 3: Порівняння різних конфігурацій

```bash
# Експеримент 1: базовий
python main.py --mode synthesis --output-dir output/exp1

# Експеримент 2: більше trials
python main.py --mode synthesis --trials 50 --output-dir output/exp2

# Експеримент 3: більше даних
python main.py --mode synthesis --samples 1500 --output-dir output/exp3

# Аналіз кожного
python main.py --mode analyze --output-dir output/exp1
python main.py --mode analyze --output-dir output/exp2
python main.py --mode analyze --output-dir output/exp3
```

### Сценарій 4: Відновлення після переривання

```bash
# Якщо експеримент було перервано
python main.py --mode synthesis --resume
```

## Параметри за замовчуванням

```python
--mode          # Обов'язковий: synthesis/train-top3/analyze/clean/info
--trials 30     # Кількість trials
--warmup 10     # Warmup trials
--epochs 1      # Епох на trial
--samples 700   # Train samples
--val-samples 200  # Val samples
--seed 42       # Random seed
--output-dir output/  # Папка результатів
```

## Очікуваний час виконання

| Режим | Apple M2 Pro (MPS) | CPU |
|-------|-------------------|-----|
| synthesis (30 trials) | ~15-18 хв | ~45-50 хв |
| synthesis (5 trials, --quick) | ~3-5 хв | ~10-15 хв |
| train-top3 | ~50-60 хв | ~2-3 години |
| analyze | ~10-30 сек | ~10-30 сек |

## Структура output/

```
output/
├── optuna_study.pkl           # Optuna study (для --resume)
├── proxy_stats.json           # Калібровні статистики
├── synthesis_results.json     # Топ-3 архітектури
├── experiment_YYYYMMDD_HHMMSS.log  # Детальний лог
├── convergence.png            # Графік convergence (після analyze)
└── trained_models/            # Натреновані моделі (після train-top3)
    ├── trial_X_best.pth
    ├── trial_Y_best.pth
    └── trial_Z_best.pth
```

## Troubleshooting

### Помилка: "Датасет не знайдено"
```bash
# Завантажте VisDrone2019-DET:
# https://github.com/VisDrone/VisDrone-Dataset
# Розпакуйте у data/train/ та data/val/
```

### Помилка: "Знайдено попередній checkpoint"
```bash
# Варіант 1: Продовжити
python main.py --mode synthesis --resume

# Варіант 2: Почати заново
python main.py --mode clean --confirm
python main.py --mode synthesis
```

### Помилка: "Не знайдено optuna_study.pkl"
```bash
# Спочатку запустіть synthesis
python main.py --mode synthesis
# Потім інші режими
python main.py --mode train-top3
```
