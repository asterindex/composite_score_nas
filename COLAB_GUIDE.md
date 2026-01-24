# Запуск на Google Colab

Цей ноутбук демонструє запуск Detection Stability Score (DSS) для NAS на Google Colab з T4 GPU.

## 📋 Кроки

### 1. Клонування репозиторію

```python
!git clone https://github.com/asterindex/composite_score_nas.git
%cd composite_score_nas
```

### 2. Встановлення залежностей

```python
!pip install -q -r requirements.txt
```

### 3. Монтування Google Drive (опціонально)

Для збереження результатів на Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Завантаження датасету

```python
# Train dataset
!mkdir -p data
!gdown "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn" -O data/VisDrone2019-DET-train.zip
!unzip -q data/VisDrone2019-DET-train.zip -d data/
!mv data/VisDrone2019-DET-train data/train

# Val dataset
!gdown "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59" -O data/VisDrone2019-DET-val.zip
!unzip -q data/VisDrone2019-DET-val.zip -d data/
!mv data/VisDrone2019-DET-val data/val

print("✅ Датасет готовий")
```

### 5. Запуск синтезу

```python
!python synthesis_universal.py
```

**Очікуваний час на T4 GPU:** ~10-12 хвилин (30 trials)

### 6. Аналіз результатів

```python
!python analyze_results.py
```

### 7. Повне навчання топ-3

```python
!python train_top3_models.py
```

## 📊 Перегляд результатів

```python
import json
import pandas as pd

# Завантаження результатів
with open('results/synthesis_results.json', 'r') as f:
    results = json.load(f)

# Топ-3 архітектури
print("🏆 Топ-3 архітектури:")
for model in results['top3']:
    print(f"\nTrial #{model['number']}: DSS = {model['value']:.4f}")
    print(f"  n_blocks: {model['params']['n_blocks']}")
    print(f"  optimizer: {model['params']['optimizer']}")
    print(f"  lr: {model['params']['lr']}")
```

## 🖼️ Візуалізація

```python
from IPython.display import Image, display

# Графік конвергенції
display(Image('results/convergence.png'))
```

## 💾 Збереження на Drive

```python
# Копіювання результатів на Drive
!mkdir -p /content/drive/MyDrive/DSS_Experiment
!cp -r results/* /content/drive/MyDrive/DSS_Experiment/
print("✅ Результати збережено на Drive")
```

## 📝 Примітки

- **GPU:** Переконайтесь, що Runtime type = GPU (T4)
- **RAM:** 12.7 GB достатньо для експерименту
- **Час:** Повний цикл (синтез + навчання топ-3) ~40-50 хвилин
- **Відтворюваність:** SEED=42 гарантує ідентичні результати

## 🔗 Посилання

- GitHub: https://github.com/asterindex/composite_score_nas
- VisDrone Dataset: http://aiskyeye.com/
- Optuna: https://optuna.org/

---

**Автор:** Анатолій Кот  
**Дата:** 2026-01-24
