#!/usr/bin/env python3
"""
РЕАЛЬНИЙ аналіз: чи проста L_val краща за DSS для ранжування?
Використовує дані з trials_proxy_metrics.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Завантажити дані
df = pd.read_csv('bayesian_optimization/trials_proxy_metrics.csv')

# Final losses (з логу full training) — вручну з логу
final_losses = {
    0: 1.1008, 1: 1.0765, 2: 1.0879, 3: 1.0964, 4: 1.0976,
    5: 1.0693, 6: 1.0979, 7: 1.0750, 8: 1.0968, 9: 1.0814,
    10: 1.0832, 11: 1.0773, 12: 1.0807, 13: 1.0751, 14: 1.0767,
    15: 1.0809, 16: 1.0924, 17: 1.0760, 18: 1.0660, 19: 1.0772,
    20: 1.1260, 21: 1.0800, 22: 1.0831, 23: 1.0736, 24: 1.0831,
    25: 1.0686, 26: 1.0839, 27: 1.0830, 28: 1.0876, 29: 1.0714,
    30: 1.0914, 31: 1.0777, 32: 1.0777, 33: 1.0838, 34: 1.0781,
    35: 1.0726, 36: 1.0783, 37: 1.0664, 38: 1.0817, 39: 1.0678,
    40: 1.0848, 41: 1.0759, 42: 1.0830, 43: 1.0734, 44: 1.0823,
    45: 1.1137, 46: 1.0848, 47: 1.0877, 48: 1.0860, 49: 1.0886
}

# Додати final losses до dataframe
df['final_loss'] = df['trial'].map(final_losses)

# Витягти метрики
lval = df['L_val'].values
dss = -df['dss_value'].values  # Інвертувати: більший DSS = краще (як в Optuna minimize)
final = df['final_loss'].values

print("=" * 80)
print("ПОРІВНЯННЯ: ПРОСТА L_val vs DSS для ранжування архітектур")
print("=" * 80)
print()
print(f"📊 Дані: {len(df)} trials (2 epochs → 15 epochs)")
print()

# ============================================
# 1. КОРЕЛЯЦІЯ З ФІНАЛЬНИМ LOSS
# ============================================
print("=" * 80)
print("1️⃣  КОРЕЛЯЦІЯ З ФІНАЛЬНИМ LOSS (Spearman ρ)")
print("=" * 80)
print()

# L_val кореляція (МЕНШЕ КРАЩЕ)
rho_lval, p_lval = spearmanr(lval, final)

# DSS кореляція (БІЛЬШЕ КРАЩЕ, тому інвертуємо)
# В Optuna minimize, тому DSS = negative value
# Для правильної кореляції: lower DSS objective = better = lower final loss
rho_dss_orig, p_dss_orig = spearmanr(-df['dss_value'], final)

print(f"   L_val (2 epochs)  ↔  Final (15 epochs):")
print(f"      ρ = {rho_lval:.4f}")
print(f"      p-value = {p_lval:.4f}")
print(f"      {'✅ Значуща!' if p_lval < 0.05 else '❌ Не значуща'}")
print()

print(f"   DSS (2 epochs)    ↔  Final (15 epochs):")
print(f"      ρ = {rho_dss_orig:.4f}")
print(f"      p-value = {p_dss_orig:.4f}")
print(f"      {'✅ Значуща!' if p_dss_orig < 0.05 else '❌ Не значуща'}")
print()

delta_rho = rho_lval - rho_dss_orig
print(f"   📈 РІЗНИЦЯ: ρ(L_val) - ρ(DSS) = {delta_rho:+.4f}")
if abs(delta_rho) < 0.05:
    print(f"      ≈ Однакові")
elif delta_rho > 0:
    print(f"      ✅ L_val краща на {abs(delta_rho):.1%}")
else:
    print(f"      ❌ DSS краща на {abs(delta_rho):.1%}")
print()

# ============================================
# 2. RANK STABILITY
# ============================================
print("=" * 80)
print("2️⃣  RANK STABILITY (скільки моделей зберегли ранг)")
print("=" * 80)
print()

lval_ranks = np.argsort(np.argsort(lval))  # 0 = best
dss_ranks = np.argsort(np.argsort(-df['dss_value']))  # 0 = best (lower objective)
final_ranks = np.argsort(np.argsort(final))  # 0 = best

lval_stability = (lval_ranks == final_ranks).sum() / len(df) * 100
dss_stability = (dss_ranks == final_ranks).sum() / len(df) * 100

print(f"   L_val:  {lval_stability:.1f}% ({int(lval_stability/2)}/50 моделей)")
print(f"   DSS:    {dss_stability:.1f}% ({int(dss_stability/2)}/50 моделей)")
print()

if lval_stability > dss_stability:
    print(f"   ✅ L_val стабільніша на {lval_stability - dss_stability:.1f}%")
else:
    print(f"   ❌ DSS стабільніша на {dss_stability - lval_stability:.1f}%")
print()

# ============================================
# 3. TOP-K OVERLAP
# ============================================
print("=" * 80)
print("3️⃣  TOP-K OVERLAP (скільки топ моделей знайдено)")
print("=" * 80)
print()

for k in [3, 5, 10]:
    topk_final = set(np.argsort(final)[:k])
    topk_lval = set(np.argsort(lval)[:k])
    topk_dss = set(np.argsort(-df['dss_value'])[:k])
    
    overlap_lval = len(topk_lval & topk_final)
    overlap_dss = len(topk_dss & topk_final)
    
    print(f"   TOP-{k}:")
    print(f"      L_val:  {overlap_lval}/{k} моделей ({overlap_lval/k*100:.0f}%)")
    print(f"      DSS:    {overlap_dss}/{k} моделей ({overlap_dss/k*100:.0f}%)")
    
    if overlap_lval > overlap_dss:
        print(f"      ✅ L_val краща (+{overlap_lval - overlap_dss})")
    elif overlap_dss > overlap_lval:
        print(f"      ❌ DSS краща (+{overlap_dss - overlap_lval})")
    else:
        print(f"      ≈ Однакові")
    print()

# ============================================
# 4. ДЕТАЛІ ТОП-10
# ============================================
print("=" * 80)
print("4️⃣  ДЕТАЛІ ТОП-10 МОДЕЛЕЙ")
print("=" * 80)
print()

# Справжній топ-10
true_top10 = np.argsort(final)[:10]

print("   СПРАВЖНІЙ ТОП-10 (за final loss):")
print()
print("   Rank | Trial | Final   | L_val | L_val Rank | DSS Rank | Best?")
print("   -----|-------|---------|-------|------------|----------|-------")

for rank, trial_idx in enumerate(true_top10, 1):
    trial = int(trial_idx)
    final_val = final[trial]
    lval_val = lval[trial]
    lval_rank = int(lval_ranks[trial] + 1)
    dss_rank = int(dss_ranks[trial] + 1)
    
    if lval_rank <= 10 and dss_rank > 10:
        best = "L_val ✅"
    elif dss_rank <= 10 and lval_rank > 10:
        best = "DSS ✅"
    elif lval_rank <= 10 and dss_rank <= 10:
        best = "Both ✅"
    else:
        best = "None ❌"
    
    print(f"   #{rank:2d}   | {trial:3d}   | {final_val:.4f} | {lval_val:.3f} | #{lval_rank:3d}       | #{dss_rank:3d}      | {best}")

print()

# ============================================
# ФІНАЛЬНИЙ ВИСНОВОК
# ============================================
print("=" * 80)
print("🎯 ФІНАЛЬНИЙ ВИСНОВОК")
print("=" * 80)
print()

if abs(delta_rho) < 0.05:
    print("📊 Кореляція: L_val і DSS показують ОДНАКОВУ кореляцію з final loss")
elif delta_rho > 0.1:
    print("✅ L_val ЗНАЧНО КРАЩА за DSS!")
    print()
    print("   Рекомендація:")
    print("   → Використовуй просту validation loss замість DSS")
    print("   → DSS додає шум без покращення точності")
    print("   → Заощадь час обчислень (менше метрик)")
elif delta_rho > 0:
    print("✅ L_val трохи краща за DSS")
    print()
    print("   Рекомендація:")
    print("   → L_val простіша і дає схожий/кращий результат")
    print("   → DSS можна використати як додаткову фічу")
elif delta_rho < -0.1:
    print("❌ DSS ЗНАЧНО КРАЩА за просту L_val!")
    print()
    print("   Висновок:")
    print("   → DSS успішно використовує динаміку тренування")
    print("   → Додаткові метрики (gap, loss_cv, grad_cv, impr) корисні")
else:
    print("❌ DSS трохи краща за L_val")
    print()
    print("   Висновок:")
    print("   → DSS додає невелике покращення")
    print("   → Але складність може не виправдовуватись")

print()
print("=" * 80)
