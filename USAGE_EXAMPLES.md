# Приклади використання main.py

## Базові команди

### 1. Швидкий старт (за замовчуванням)
```bash
python3 main.py
```
Запустить швидкий тест (5 trials) без додаткових параметрів.

### 2. Інформація про проект
```bash
python3 main.py --mode info
```

### 3. Довідка по параметрах
```bash
python3 main.py --help
```

## Режими роботи

### Synthesis - Синтез архітектур

**Швидкий тест (5 trials):**
```bash
python3 main.py --mode fast
```

**Повний експеримент (30 trials):**
```bash
python3 main.py --mode full
```

**Користувацька конфігурація:**
```bash
# 50 trials з більшим датасетом
python3 main.py --mode synthesis --trials 50 --samples 1000 --val-samples 300

# Зміна warmup фази
python3 main.py --mode synthesis --warmup 15

# Більше епох для кожного trial
python3 main.py --mode synthesis --epochs 2

# Інший seed
python3 main.py --mode synthesis --seed 123
```

**Продовження експерименту:**
```bash
python3 main.py --mode synthesis --resume
```

**Збереження в іншу папку:**
```bash
python3 main.py --mode synthesis --output-dir experiments/run1
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
python3 analyze.py
# або:
python3 main.py --mode analyze
```

**З детальним виводом:**
```bash
python3 analyze.py --verbose
```

**З альтернативної папки:**
```bash
python3 analyze.py --output-dir experiments/run1
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
# 1. Швидкий тест (за замовчуванням, 3-5 хвилин)
python3 main.py

# 2. Перевірка результатів
python3 analyze.py

# 3. Очищення
python3 main.py --mode clean --confirm
```

### Сценарій 2: Повний експеримент

```bash
# 1. Повний синтез (15-18 хвилин)
python3 main.py --mode full

# 2. Аналіз результатів
python3 analyze.py

# 3. Тренування топ-3 (50+ хвилин)
python3 main.py --mode train-top3
```

### Сценарій 3: Порівняння різних конфігурацій

```bash
# Експеримент 1: швидкий
python3 main.py --mode fast --output-dir output/exp1

# Експеримент 2: повний
python3 main.py --mode full --output-dir output/exp2

# Експеримент 3: більше даних
python3 main.py --mode synthesis --samples 1500 --output-dir output/exp3

# Аналіз кожного
python3 analyze.py --output-dir output/exp1
python3 analyze.py --output-dir output/exp2
python3 analyze.py --output-dir output/exp3
```

### Сценарій 4: Відновлення після переривання

```bash
# Якщо експеримент було перервано
python3 main.py --mode synthesis --resume
```

## Параметри за замовчуванням

```python
# Режими
--mode fast         # 5 trials, 200 train, 50 val, 3 warmup
--mode full         # 30 trials, 700 train, 200 val, 10 warmup
--mode synthesis    # Користувацька конфігурація

# Параметри
--trials 30         # Кількість trials
--warmup 10         # Warmup trials
--epochs 1          # Епох на trial
--samples 700       # Train samples
--val-samples 200   # Val samples
--seed 42           # Random seed
--output-dir output/ # Папка результатів
```

## Очікуваний час виконання

| Режим | Apple M2 Pro (MPS) | CPU |
|-------|-------------------|-----|
| fast (5 trials) | ~3-5 хв | ~10-15 хв |
| full (30 trials) | ~15-18 хв | ~45-50 хв |
| synthesis (50 trials) | ~25-30 хв | ~75-85 хв |
| train-top3 | ~50-60 хв | ~2-3 години |
| analyze | ~10-30 сек | ~10-30 сек |

## Структура output/

```
output/
├── optuna_study.pkl           # Optuna study (для --resume)
├── proxy_stats.json           # Статистики калібрування
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
