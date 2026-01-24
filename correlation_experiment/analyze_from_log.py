#!/usr/bin/env python3
"""
Аналіз L_val vs DSS з логу A100 запуску
Витягує дані з логу та порівнює кореляції
"""

import re
import numpy as np
from scipy.stats import spearmanr

print("=" * 80)
print("АНАЛІЗ: L_val vs DSS з логу A100 (50 trials)")
print("=" * 80)
print()

# ============================================
# ДАНІ З ЛОГУ
# ============================================

# DSS values з логу синтезу (Trial X finished with value: Y)
dss_values = [
    -0.8457,  # Trial 0
    -1.8540,  # Trial 1
    -1.3269,  # Trial 2
    -2.1910,  # Trial 3
    -0.5692,  # Trial 4
    -2.3040,  # Trial 5
    -0.8481,  # Trial 6
    -2.2434,  # Trial 7
    -1.6154,  # Trial 8
    -1.9210,  # Trial 9
    -2.8300,  # Trial 10
    -2.1129,  # Trial 11
    -2.4866,  # Trial 12
    -2.0826,  # Trial 13
    -1.9024,  # Trial 14
    -1.5566,  # Trial 15
    -2.0522,  # Trial 16
    -2.4542,  # Trial 17
    -1.6812,  # Trial 18
    -2.0874,  # Trial 19
    -0.9789,  # Trial 20
    -1.6915,  # Trial 21
    -2.9911,  # Trial 22
    -1.3214,  # Trial 23
    -2.0268,  # Trial 24
    -1.5300,  # Trial 25
    -1.7421,  # Trial 26
    -1.4981,  # Trial 27
    -1.3930,  # Trial 28
    -2.1938,  # Trial 29
    -1.4412,  # Trial 30
    -1.8567,  # Trial 31
    -2.0546,  # Trial 32
    -1.5690,  # Trial 33
    -1.7718,  # Trial 34
    -1.7263,  # Trial 35
    -1.8080,  # Trial 36
    -1.9754,  # Trial 37
    -2.5472,  # Trial 38
    -2.0537,  # Trial 39
    -2.0600,  # Trial 40
    -2.4231,  # Trial 41
    -1.7521,  # Trial 42
    -1.6721,  # Trial 43
    -2.9801,  # Trial 44
    -0.6956,  # Trial 45
    -2.5760,  # Trial 46
    -2.1697,  # Trial 47
    -2.2018,  # Trial 48
    -1.2372,  # Trial 49
]

# Final losses з логу full training (Найкращий Val Loss)
final_losses = [
    1.1008,  # Trial 0
    1.0765,  # Trial 1
    1.0879,  # Trial 2
    1.0964,  # Trial 3
    1.0976,  # Trial 4
    1.0693,  # Trial 5
    1.0979,  # Trial 6
    1.0750,  # Trial 7
    1.0968,  # Trial 8
    1.0814,  # Trial 9
    1.0832,  # Trial 10
    1.0773,  # Trial 11
    1.0807,  # Trial 12
    1.0751,  # Trial 13
    1.0767,  # Trial 14
    1.0809,  # Trial 15
    1.0924,  # Trial 16
    1.0760,  # Trial 17
    1.0660,  # Trial 18 ← BEST!
    1.0772,  # Trial 19
    1.1260,  # Trial 20
    1.0800,  # Trial 21
    1.0831,  # Trial 22
    1.0736,  # Trial 23
    1.0831,  # Trial 24
    1.0686,  # Trial 25
    1.0839,  # Trial 26
    1.0830,  # Trial 27
    1.0876,  # Trial 28
    1.0714,  # Trial 29
    1.0914,  # Trial 30
    1.0777,  # Trial 31
    1.0777,  # Trial 32
    1.0838,  # Trial 33
    1.0781,  # Trial 34
    1.0726,  # Trial 35
    1.0783,  # Trial 36
    1.0664,  # Trial 37
    1.0817,  # Trial 38
    1.0678,  # Trial 39
    1.0848,  # Trial 40
    1.0759,  # Trial 41
    1.0830,  # Trial 42
    1.0734,  # Trial 43
    1.0823,  # Trial 44
    1.1137,  # Trial 45
    1.0848,  # Trial 46
    1.0877,  # Trial 47
    1.0860,  # Trial 48
    1.0886,  # Trial 49
]

# Для L_val після 2 епох: треба зворотно розрахувати з DSS
# DSS = z(L_val) + 0.6*z(gap) + 0.4*z(loss_cv) + 0.2*z(grad_cv) - 0.4*z(impr)
# Припускаємо, що основний внесок — L_val (близько 40-50%)
# Оцінка: denormalize DSS component

# З warmup trials (0-9), де objective = L_val + 0.5*gap
# Можна оцінити L_val
# Але для простоти: використаємо кореляцію між early loss та DSS

# Альтернатива: припустимо L_val ≈ final * 1.4 + noise корельований з DSS
# Але точніше: використаємо DSS як проксі для "early ranking"

dss = np.array(dss_values)
final = np.array(final_losses)

# Оцінка L_val з DSS (грубо):
# Lower DSS objective = better = lower L_val
# Inverse scaling для імітації L_val
estimated_lval = -dss * 0.5 + 1.5  # Масштабування до розумного діапазону

print("📊 ДАНІ:")
print(f"   Trials: {len(dss)}")
print(f"   DSS range: {dss.min():.3f} to {dss.max():.3f}")
print(f"   Final range: {final.min():.4f} to {final.max():.4f}")
print(f"   Estimated L_val range: {estimated_lval.min():.3f} to {estimated_lval.max():.3f}")
print()

# ============================================
# 1. КОРЕЛЯЦІЯ
# ============================================
print("=" * 80)
print("1️⃣  КОРЕЛЯЦІЯ З ФІНАЛЬНИМ LOSS (Spearman ρ)")
print("=" * 80)
print()

# DSS кореляція
rho_dss, p_dss = spearmanr(-dss, final)  # Lower DSS = better

# Estimated L_val кореляція
rho_lval_est, p_lval_est = spearmanr(estimated_lval, final)

print(f"   DSS (2 epochs)    ↔  Final (15 epochs):")
print(f"      ρ = {rho_dss:.4f}")
print(f"      p = {p_dss:.4f}")
print(f"      {'✅ Значуща' if p_dss < 0.05 else '❌ Не значуща'}")
print()

print(f"   Estimated L_val   ↔  Final (15 epochs):")
print(f"      ρ = {rho_lval_est:.4f}")
print(f"      p = {p_lval_est:.4f}")
print(f"      {'✅ Значуща' if p_lval_est < 0.05 else '❌ Не значуща'}")
print()

print("   ⚠️  ВАЖЛИВО: L_val оцінка базується на DSS (грубе наближення)")
print("       Для точного порівняння потрібні реальні L_val після 2 епох")
print()

# ============================================
# 2. ТЕОРЕТИЧНИЙ АНАЛІЗ
# ============================================
print("=" * 80)
print("2️⃣  ТЕОРЕТИЧНИЙ АНАЛІЗ")
print("=" * 80)
print()

print("   DSS формула:")
print("   DSS = z(L_val) + 0.6*z(gap) + 0.4*z(loss_cv) + 0.2*z(grad_cv) - 0.4*z(impr)")
print()

print("   Ваги компонентів:")
print("      L_val:     1.0  (основний)")
print("      gap:       0.6  (overfitting)")
print("      loss_cv:   0.4  (stability)")
print("      grad_cv:   0.2  (optimization)")
print("      impr:     -0.4  (learning speed)")
print("      ─────────")
print("      Total:     2.6  (ефективна вага)")
print()

print("   Якщо L_val САМА має таку ж кореляцію, то:")
print("   → Додаткові 4 компоненти (gap, loss_cv, grad_cv, impr) НЕ ДОПОМАГАЮТЬ")
print("   → DSS додає ШУМ замість сигналу")
print("   → Висновок: використовуй просту L_val")
print()

# ============================================
# 3. TOP-K OVERLAP
# ============================================
print("=" * 80)
print("3️⃣  TOP-K OVERLAP")
print("=" * 80)
print()

dss_ranks = np.argsort(np.argsort(-dss))  # 0 = best
final_ranks = np.argsort(np.argsort(final))  # 0 = best

for k in [3, 5, 10]:
    topk_final = set(np.argsort(final)[:k])
    topk_dss = set(np.argsort(-dss)[:k])
    
    overlap_dss = len(topk_dss & topk_final)
    
    print(f"   TOP-{k}:")
    print(f"      DSS знайшла: {overlap_dss}/{k} моделей ({overlap_dss/k*100:.0f}%)")
print()

# ============================================
# 4. ДЕТАЛЬНИЙ АНАЛІЗ ТОП-10
# ============================================
print("=" * 80)
print("4️⃣  ТОП-10 МОДЕЛЕЙ (за final loss)")
print("=" * 80)
print()

true_top10 = np.argsort(final)[:10]

print("   Rank | Trial | Final   | DSS     | DSS Rank | Delta")
print("   -----|-------|---------|---------|----------|-------")

for rank, trial_idx in enumerate(true_top10, 1):
    trial = int(trial_idx)
    final_val = final[trial]
    dss_val = dss[trial]
    dss_rank = int(dss_ranks[trial] + 1)
    delta = dss_rank - rank
    
    status = "✅" if dss_rank <= 10 else "❌"
    
    print(f"   #{rank:2d}   | {trial:3d}   | {final_val:.4f} | {dss_val:7.4f} | #{dss_rank:3d}      | {delta:+3d} {status}")

print()

# ============================================
# 5. ФІНАЛЬНИЙ ВИСНОВОК
# ============================================
print("=" * 80)
print("🎯 ВИСНОВОК")
print("=" * 80)
print()

print("Базуючись на кореляції DSS з фінальним loss:")
print()

if rho_dss < 0.4:
    print("❌ DSS СЛАБКА (ρ = {:.3f})".format(rho_dss))
    print()
    print("   Проблеми:")
    print("   1. Додаткові метрики (gap, loss_cv, grad_cv) не покращують ранжування")
    print("   2. Можливо додають шум замість корисного сигналу")
    print("   3. 2 епохи недостатньо для стабільних метрик")
    print()
    print("   Рекомендація:")
    print("   → Спробувати просту L_val замість DSS")
    print("   → Або збільшити EPOCHS_PER_TRIAL до 5")
    print("   → DSS може бути корисна тільки для фільтрації bottom 50%")
elif rho_dss < 0.6:
    print("⚠️  DSS ПОМІРНА (ρ = {:.3f})".format(rho_dss))
    print()
    print("   DSS працює, але не ідеально.")
    print("   Можна покращити:")
    print("   → Збільшити epochs до 3-5")
    print("   → Або спробувати інші ваги в формулі DSS")
else:
    print("✅ DSS СИЛЬНА (ρ = {:.3f})".format(rho_dss))
    print()
    print("   DSS успішно використовує training dynamics!")

print()
print("=" * 80)
print()

# Статистика з логу
best_trial = 18
best_dss_trial = 22

print("📈 ЦІКАВІ ФАКТИ З ЛОГУ:")
print()
print(f"   Найкраща модель: Trial {best_trial} (Final = {final[best_trial]:.4f})")
print(f"      DSS rank: #{dss_ranks[best_trial] + 1} (DSS = {dss[best_trial]:.4f})")
print()
print(f"   Найкраща за DSS: Trial {best_dss_trial} (DSS = {dss[best_dss_trial]:.4f})")
print(f"      Final rank: #{final_ranks[best_dss_trial] + 1} (Final = {final[best_dss_trial]:.4f})")
print()
print(f"   DSS помилилась на: {abs(dss_ranks[best_trial] - final_ranks[best_trial])} позицій для best model")
print()
print("=" * 80)
