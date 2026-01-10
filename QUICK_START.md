# 🚀 QUICK START

## 📦 Крок 0: Завантажте датасет

**VisDrone2019-DET Dataset:**
- 🌐 Офіційний сайт: https://github.com/VisDrone/VisDrone-Dataset
- 📥 Прямий лінк: http://aiskyeye.com/download/object-detection-2/
- 📊 Розмір: ~1.5 GB

**Для локального використання:**
```bash
# Розпакуйте датасет у папку dataset/
unzip dataset.zip -d dataset/
```

**Для Google Colab:**
- Завантажте `dataset.zip` на Google Drive
- Помістіть у `MyDrive/Studying/Experiments/Composite_score_nas/dataset.zip`

---

## Локально (тест)

```bash
cd /Users/anatolykot/Projects/composite_score_nas
python3 bayesian_optimization/synthesis_universal.py
```

**Режим:** Швидкий тест (10 trials, ~30 хв)  
**Девайс:** MPS (Apple Silicon)  
**Результати:** `bayesian_optimization/results/`

---

## Google Colab (повний прогін)

### 1. Завантажте файли
- `synthesis_universal.py` → Colab
- `dataset.zip` → `MyDrive/Studying/Experiments/Composite_score_nas/dataset.zip`

### 2. Налаштуйте Runtime
- Runtime → Change runtime type → **T4 GPU**

### 3. Запустіть
```python
!python synthesis_universal.py
```

**Все автоматично:**
- Змонтує Drive
- Розпакує датасет (~2-3 хв)
- Почне експеримент

**Режим:** Повний прогін (50 trials, ~12 год)  
**Девайс:** CUDA (T4 GPU)  
**Результати:** `MyDrive/Studying/Experiments/Composite_score_nas/results_full/`

---

## 🔄 Якщо відключилося

**Просто перезапустіть той самий скрипт!**

Він автоматично:
- ✅ Знайде checkpoint
- ✅ Відновить прогрес
- ✅ Продовжить з місця зупинки

**Checkpoint'и кожні 5 моделей = максимальна втрата ~1 год**

---

## 📊 Що отримаєте

**Головний файл:** `synthesis_results.json`

```json
{
  "synthesis": {
    "top3_models": [...],  // Топ-3 архітектури
    "n_trials": 50
  },
  "training": {
    "results": [...],      // Повне тренування всіх моделей
    "epochs": 15
  },
  "analysis": {
    "spearman_rho": 0.743  // Кореляція proxy → final
  }
}
```

**Інші файли:**
- `trials_proxy_metrics.csv` - метрики всіх trials
- `proxy_stats.json` - калібрація Composite Score

---

## 🆘 Допомога

- **Детальна інструкція:** `COLAB_CHECKLIST.md`
- **Resume guide:** `RESUME_GUIDE.md`
- **Changelog:** `CHANGELOG.md`
- **Full README:** `README.md`

---

**Готові? ЛИШ ЗАПУСТІТЬ!** 🎯✨
