# 📦 Датасет VisDrone2019-DET

## Опис

Проект використовує **VisDrone2019-DET** - датасет для детекції об'єктів, зібраний з дронів.

### Характеристики:

- **Джерело:** VisDrone Dataset (http://aiskyeye.com/)
- **Тип задачі:** Object Detection
- **Формат анотацій:** YOLO format
- **Розмір:** ~1.5 GB (розпакований)
- **Класи:** 10 категорій об'єктів (pedestrian, car, van, truck, bus, motor, bicycle, etc.)

### Статистика:

| Набір | Зображень | Об'єктів | Розмір |
|-------|-----------|----------|--------|
| Train | 6,471     | ~540K    | ~1.2GB |
| Val   | 548       | ~47K     | ~300MB |

## 📥 Завантаження

### Офіційні джерела:

1. **GitHub репозиторій:**
   - https://github.com/VisDrone/VisDrone-Dataset

2. **Прямий лінк для завантаження:**
   - http://aiskyeye.com/download/object-detection-2/
   - Файл: `VisDrone2019-DET-train.zip` + `VisDrone2019-DET-val.zip`

3. **Альтернативні джерела:**
   - Kaggle: https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav
   - Roboflow: https://universe.roboflow.com/visdrone2019/visdrone2019-det

## 📂 Структура після розпакування

```
dataset/
├── train/
│   ├── images/           # 6471 зображення (.jpg)
│   ├── annotations/      # XML анотації (Pascal VOC format)
│   ├── labels/           # TXT анотації (YOLO format)
│   └── labels.cache      # Кеш для швидкого завантаження
│
└── val/
    ├── images/           # 548 зображень (.jpg)
    ├── annotations/      # XML анотації
    ├── labels/           # TXT анотації
    └── labels.cache      # Кеш
```

## 🛠️ Підготовка датасету

### Для локального використання:

```bash
# 1. Завантажте датасет з офіційного джерела
wget http://aiskyeye.com/download/object-detection-2/VisDrone2019-DET-train.zip
wget http://aiskyeye.com/download/object-detection-2/VisDrone2019-DET-val.zip

# 2. Розпакуйте в папку dataset/
unzip VisDrone2019-DET-train.zip -d dataset/train/
unzip VisDrone2019-DET-val.zip -d dataset/val/

# 3. Перевірте структуру
ls -lh dataset/train/images/ | wc -l  # має бути 6471
ls -lh dataset/val/images/ | wc -l    # має бути 548
```

### Для Google Colab:

**Варіант 1: Завантажити на Google Drive (рекомендовано)**

1. Завантажте `dataset.zip` локально
2. Завантажте на Google Drive:
   ```
   MyDrive/Studying/Experiments/Composite_score_nas/dataset.zip
   ```
3. Скрипт `synthesis_universal.py` автоматично:
   - Знайде файл на Drive
   - Розпакує в `/content/dataset/`
   - Використає для тренування

**Варіант 2: Завантажити безпосередньо в Colab**

```python
# В Colab notebook
!wget http://aiskyeye.com/download/object-detection-2/VisDrone2019-DET-train.zip
!wget http://aiskyeye.com/download/object-detection-2/VisDrone2019-DET-val.zip

!unzip VisDrone2019-DET-train.zip -d dataset/train/
!unzip VisDrone2019-DET-val.zip -d dataset/val/
```

**⚠️ Увага:** При варіанті 2 датасет буде втрачено після перезапуску Colab!

## 🔍 Формат анотацій

### YOLO format (використовується в проекті):

Кожен файл `.txt` містить рядки формату:
```
<class_id> <x_center> <y_center> <width> <height>
```

Всі координати нормалізовані (0-1).

**Приклад:**
```
0 0.5123 0.3456 0.0234 0.0456
1 0.7890 0.6543 0.0567 0.0890
```

### Класи об'єктів:

| ID | Клас       | Опис                    |
|----|------------|-------------------------|
| 0  | pedestrian | Пішохід                 |
| 1  | people     | Група людей             |
| 2  | bicycle    | Велосипед               |
| 3  | car        | Легковий автомобіль     |
| 4  | van        | Фургон                  |
| 5  | truck      | Вантажівка              |
| 6  | tricycle   | Триколісний велосипед   |
| 7  | awning-tricycle | Триколісний з тентом |
| 8  | bus        | Автобус                 |
| 9  | motor      | Мотоцикл                |

## 📊 Використання в проекті

Скрипт `synthesis_universal.py` автоматично:

1. **Завантажує датасет:**
   - Локально: з папки `dataset/`
   - Colab: з Google Drive (`MyDrive/Studying/Experiments/Composite_score_nas/dataset.zip`)

2. **Створює підмножини:**
   - Train: використовує `MAX_SAMPLES` зображень (за замовчуванням 2000)
   - Val: використовує всі 548 зображень

3. **Кешує дані:**
   - Створює `labels.cache` для швидкого завантаження
   - Зберігає індекси підмножин у `val_subset_idx.npy`

## 🔗 Посилання

- **Офіційний сайт:** http://aiskyeye.com/
- **GitHub:** https://github.com/VisDrone/VisDrone-Dataset
- **Paper:** "Vision Meets Drones: Past, Present and Future" (arXiv:2001.06303)
- **Benchmark:** http://aiskyeye.com/evaluate/results-format

## 📝 Цитування

Якщо використовуєте датасет у дослідженні, процитуйте:

```bibtex
@article{zhu2020vision,
  title={Vision Meets Drones: Past, Present and Future},
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={arXiv preprint arXiv:2001.06303},
  year={2020}
}
```

---

**Автор проекту:** Анатолій Кот (Anatoly Kot)  
**Email:** anatoly.kot@gmail.com  
**Дата оновлення:** 2026-01-10
