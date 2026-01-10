# 🚀 Інструкції для Google Colab

## 📦 Крок 1: Завантажте датасет

### Завантаження VisDrone2019-DET:

**Офіційні джерела:**
- 🌐 GitHub: https://github.com/VisDrone/VisDrone-Dataset
- 📥 Прямий лінк: http://aiskyeye.com/download/object-detection-2/
- 📊 Розмір: ~1.5 GB

### Підготовка для Colab:

1. **Завантажте датасет на Google Drive:**
   - Завантажте `dataset.zip` з офіційного джерела
   - Помістіть у `MyDrive/Studying/Experiments/Composite_score_nas/dataset.zip`
   
2. **Структура на Drive:**
   ```
   MyDrive/
   └── Studying/
       └── Experiments/
           └── Composite_score_nas/
               └── dataset.zip  ← тут має бути файл
   ```

## Крок 2: Налаштування Colab

1. **Створіть новий Colab notebook або завантажте скрипт:**
   ```python
   # В першій клітинці Colab
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Запуск експерименту

### Варіант 1: Завантажити скрипт з Drive

```python
# Скопіювати скрипт з Drive
!cp /content/drive/MyDrive/Studying/composite_score_nas/synthesis_universal.py .

# Або клонувати з GitHub (якщо є репозиторій)
# !git clone https://github.com/your-repo/composite_score_nas.git
# %cd composite_score_nas/bayesian_optimization
```

### Варіант 2: Завантажити через інтерфейс

1. Натисніть 📁 Files в лівій панелі
2. Upload → виберіть `synthesis_universal.py`

### Запуск

```python
# Запустити експеримент
%run synthesis_universal.py
```

## Після завершення

Скрипт автоматично:

✅ **Збереже результати на Google Drive:**
- Шлях: `MyDrive/Studying/composite_score_nas_results/`
- Папка: `results_synthesis/` (після синтезу) або `results/` (після повного пайплайну)

✅ **Завантажить ключові файли в браузер:**
- `top3_models.json`
- `synthesis_results.json`
- `trials_proxy_metrics.csv`

## Робота з результатами

### На Colab

```python
# Переглянути топ-3 моделі
import json
with open('results/top3_models.json') as f:
    top3 = json.load(f)
    print(json.dumps(top3, indent=2))

# Переглянути метрики
import pandas as pd
df = pd.read_csv('results/trials_proxy_metrics.csv')
print(df.head())
```

### Локально (Mac/Linux)

Після завершення експерименту на Colab:

```bash
# Скопіювати результати з Google Drive на локальну машину
cp -r ~/Google\ Drive/My\ Drive/Studying/composite_score_nas_results/results/* \
      ~/Projects/composite_score_nas/bayesian_optimization/results/

# Або використати gdown
pip install gdown
# ... (детальніше в документації)
```

## Налаштування

Перед запуском можна змінити параметри в `synthesis_universal.py`:

```python
# Параметри синтезу (рядки 85-100)
N_TRIALS = 50              # Кількість спроб (зменшіть для тесту)
EPOCHS_PER_TRIAL = 2       # Епохи для оцінки
MAX_SAMPLES = 2000         # Зображень для синтезу
VAL_SUBSET = 200           # Розмір validation subset
FULL_PIPELINE = True       # True = повне тренування
FULL_EPOCHS = 15           # Епохи для повного тренування
```

### Швидкий тест (1-2 години)

```python
N_TRIALS = 10
EPOCHS_PER_TRIAL = 1
MAX_SAMPLES = 500
FULL_PIPELINE = False
```

### Повний експеримент (~12 годин на T4)

```python
N_TRIALS = 50
EPOCHS_PER_TRIAL = 2
MAX_SAMPLES = 2000
FULL_PIPELINE = True
FULL_EPOCHS = 15
```

## Troubleshooting

**Проблема:** "Dataset not found"
- Перевірте, що `dataset.zip` в `MyDrive/Studying/dataset.zip`
- Перемонтуйте Drive: `drive.mount('/content/drive', force_remount=True)`

**Проблема:** Out of memory
- Зменшіть `MAX_SAMPLES` або `BATCH_SIZE_OPTIONS`
- Використайте GPU з більшою пам'яттю (A100 замість T4)

**Проблема:** Результати не збереглися на Drive
- Перевірте, що Drive змонтовано
- Файли збережено локально в `/content/results/`
- Можна вручну скопіювати: `!cp -r results /content/drive/MyDrive/Studying/`

---

**Автор:** Анатолій Кот (Anatoly Kot)  
**Оновлено:** 2026-01-09
