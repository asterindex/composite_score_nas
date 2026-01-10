#!/usr/bin/env python3
"""
Аналіз кореляцій між метриками на ранніх епохах та фінальною якістю
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# НАЛАШТУВАННЯ
# ============================================

METRICS_CSV = Path(__file__).parent / "results" / "all_metrics_per_epoch.csv"
OUTPUT_DIR = Path(__file__).parent / "results"

print("=" * 80)
print("АНАЛІЗ КОРЕЛЯЦІЙ")
print("=" * 80)
print()

# ============================================
# LOAD DATA
# ============================================

print("📊 Завантаження даних...")
df = pd.read_csv(METRICS_CSV)

print(f"   Завантажено: {len(df)} записів")
print(f"   Моделей: {df['model_idx'].nunique()}")
print(f"   Епох на модель: {df.groupby('model_idx')['epoch'].count().max()}")
print()

# ============================================
# PREPARE DATA
# ============================================

# Фінальна якість кожної моделі (best val_loss)
final_quality = df.groupby('model_idx')['val_loss'].min().reset_index()
final_quality.columns = ['model_idx', 'final_val_loss']

print(f"📈 Фінальна якість моделей:")
print(f"   Best: {final_quality['final_val_loss'].min():.4f}")
print(f"   Worst: {final_quality['final_val_loss'].max():.4f}")
print(f"   Mean: {final_quality['final_val_loss'].mean():.4f}")
print()

# ============================================
# CORRELATION ANALYSIS BY EPOCH
# ============================================

print("=" * 80)
print("1️⃣  КОРЕЛЯЦІЯ МЕТРИК З ФІНАЛЬНОЮ ЯКІСТЮ (по епохах)")
print("=" * 80)
print()

# Метрики для аналізу
metrics_to_analyze = [
    'train_loss',
    'val_loss',
    'gap',
    'improvement',
    'train_loss_cv',
    'val_loss_cv',
    'grad_norm_mean',
    'grad_norm_cv',
]

epochs_to_check = [1, 3, 5, 10, 15]

# DataFrame для результатів
correlation_results = []

for epoch in epochs_to_check:
    epoch_data = df[df['epoch'] == epoch].copy()
    
    # Merge з фінальною якістю
    epoch_data = epoch_data.merge(final_quality, on='model_idx')
    
    print(f"\n📍 EPOCH {epoch}:")
    print(f"   {'Метрика':<20} | Spearman ρ | Pearson r | p-value")
    print(f"   {'-'*20}-|------------|-----------|----------")
    
    for metric in metrics_to_analyze:
        if metric not in epoch_data.columns:
            continue
        
        # Remove NaN
        valid_data = epoch_data[[metric, 'final_val_loss']].dropna()
        
        if len(valid_data) < 3:
            continue
        
        # Spearman correlation
        rho, p_spearman = spearmanr(valid_data[metric], valid_data['final_val_loss'])
        
        # Pearson correlation
        r, p_pearson = pearsonr(valid_data[metric], valid_data['final_val_loss'])
        
        # Store results
        correlation_results.append({
            'epoch': epoch,
            'metric': metric,
            'spearman_rho': rho,
            'pearson_r': r,
            'p_value': p_spearman,
            'n_samples': len(valid_data)
        })
        
        # Визначити якість кореляції
        abs_rho = abs(rho)
        if abs_rho >= 0.7:
            quality = "🟢 Сильна"
        elif abs_rho >= 0.5:
            quality = "🟡 Помірна"
        elif abs_rho >= 0.3:
            quality = "🟠 Слабка"
        else:
            quality = "⚪ Дуже слабка"
        
        print(f"   {metric:<20} | {rho:10.4f} | {r:9.4f} | {p_spearman:.4f}  {quality}")

# Save correlation results
corr_df = pd.DataFrame(correlation_results)
corr_df.to_csv(OUTPUT_DIR / "correlation_analysis.csv", index=False)
print()
print(f"💾 Збережено результати: {OUTPUT_DIR / 'correlation_analysis.csv'}")
print()

# ============================================
# BEST PREDICTORS
# ============================================

print("=" * 80)
print("2️⃣  НАЙКРАЩІ ПРЕДИКТОРИ ФІНАЛЬНОЇ ЯКОСТІ")
print("=" * 80)
print()

# Знайти найкращі метрики для ранніх епох (1, 3, 5)
early_epochs = [1, 3, 5]

for epoch in early_epochs:
    print(f"\n🎯 EPOCH {epoch} (ранній етап):")
    
    epoch_corr = corr_df[corr_df['epoch'] == epoch].copy()
    epoch_corr['abs_rho'] = epoch_corr['spearman_rho'].abs()
    epoch_corr = epoch_corr.sort_values('abs_rho', ascending=False)
    
    print(f"   Top-3 предиктори:")
    for idx, (i, row) in enumerate(epoch_corr.head(3).iterrows(), 1):
        direction = "↑ більше = гірше" if row['spearman_rho'] > 0 else "↓ менше = краще"
        print(f"   #{idx}. {row['metric']:<20} | ρ = {row['spearman_rho']:+.3f}  {direction}")

print()

# ============================================
# COMPOSITE SCORE ANALYSIS
# ============================================

print("=" * 80)
print("3️⃣  КОМПОЗИТНИЙ SCORE (на основі кореляцій)")
print("=" * 80)
print()

# Побудуємо оптимальний composite score для epoch 5
epoch_5_data = df[df['epoch'] == 5].copy()
epoch_5_data = epoch_5_data.merge(final_quality, on='model_idx')

# Використаємо топ метрики
top_metrics_epoch5 = corr_df[corr_df['epoch'] == 5].copy()
top_metrics_epoch5['abs_rho'] = top_metrics_epoch5['spearman_rho'].abs()
top_metrics_epoch5 = top_metrics_epoch5.sort_values('abs_rho', ascending=False).head(5)

print("📐 Формула оптимального proxy (epoch 5):")
print()

# Нормалізація метрик
for _, row in top_metrics_epoch5.iterrows():
    metric = row['metric']
    weight = row['spearman_rho']
    
    if metric in epoch_5_data.columns:
        # Z-score normalization
        mean = epoch_5_data[metric].mean()
        std = epoch_5_data[metric].std()
        epoch_5_data[f'{metric}_z'] = (epoch_5_data[metric] - mean) / (std + 1e-8)

# Composite score (weighted sum)
composite_score = 0
weights_str = []

for _, row in top_metrics_epoch5.iterrows():
    metric = row['metric']
    weight = row['spearman_rho']
    
    if f'{metric}_z' in epoch_5_data.columns:
        composite_score += weight * epoch_5_data[f'{metric}_z']
        weights_str.append(f"{weight:+.3f} * z({metric})")

epoch_5_data['composite_score'] = composite_score

# Кореляція composite score з фінальною якістю
rho_composite, p_composite = spearmanr(epoch_5_data['composite_score'], epoch_5_data['final_val_loss'])

print(f"   Composite Score = {' + '.join(weights_str[:3])}")
print(f"                     {' + '.join(weights_str[3:])}" if len(weights_str) > 3 else "")
print()
print(f"   Spearman ρ (composite vs final) = {rho_composite:.4f}")
print(f"   p-value = {p_composite:.6f}")
print()

# Порівняння з простим val_loss
rho_simple, p_simple = spearmanr(epoch_5_data['val_loss'], epoch_5_data['final_val_loss'])
print(f"   Для порівняння:")
print(f"   Простий val_loss (epoch 5) ρ = {rho_simple:.4f}")
print()

if rho_composite > rho_simple:
    improvement = (rho_composite - rho_simple) / rho_simple * 100
    print(f"   ✅ Composite score на {improvement:.1f}% краще за простий val_loss!")
else:
    print(f"   ⚠️  Простий val_loss достатньо хороший!")

print()

# ============================================
# VISUALIZATION HINTS
# ============================================

print("=" * 80)
print("4️⃣  РЕКОМЕНДАЦІЇ ДЛЯ ВІЗУАЛІЗАЦІЇ")
print("=" * 80)
print()

print("   Графіки для статті:")
print("   1. Heatmap: кореляція метрик по епохах")
print("   2. Scatter: val_loss (epoch 5) vs final_val_loss")
print("   3. Bar chart: |ρ| для різних метрик на epoch 5")
print("   4. Line plot: кореляція val_loss по епохах (1-15)")
print()

# ============================================
# SUMMARY
# ============================================

print("=" * 80)
print("🎯 ВИСНОВОК")
print("=" * 80)
print()

# Знайти best epoch для val_loss
val_loss_corr = corr_df[corr_df['metric'] == 'val_loss'].copy()
best_epoch_row = val_loss_corr.loc[val_loss_corr['spearman_rho'].abs().idxmax()]

print(f"✅ Найкраща метрика: val_loss")
print(f"   Найкраща епоха: {int(best_epoch_row['epoch'])}")
print(f"   Spearman ρ = {best_epoch_row['spearman_rho']:.4f}")
print()

# Чи потрібні додаткові метрики?
if rho_composite > rho_simple + 0.05:
    print(f"✅ Композитний score покращує результат!")
    print(f"   Використовуй топ-5 метрик на epoch {int(best_epoch_row['epoch'])}")
else:
    print(f"💡 Простий val_loss достатньо!")
    print(f"   Немає потреби в складних композитних метриках")

print()
print("=" * 80)
