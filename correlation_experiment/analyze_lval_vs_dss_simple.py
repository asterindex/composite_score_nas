#!/usr/bin/env python3
"""
РЕАЛЬНИЙ аналіз: чи проста L_val краща за DSS для ранжування?
Використовує дані з trials_proxy_metrics.csv (без pandas)
"""

import csv
import numpy as np
from scipy.stats import spearmanr

# Завантажити дані з CSV
lval_list = []
dss_list = []

with open('bayesian_optimization/trials_proxy_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        lval_list.append(float(row['L_val']))
        dss_list.append(float(row['dss_value']))

lval = np.array(lval_list)
dss = np.array(dss_list)

# Final losses (з логу full training)
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

final = np.array([final_losses[i] for i in range(50)])

print("=" * 80)
print("ПОРІВНЯННЯ: ПРОСТА L_val vs DSS для ранжування архітектур")
print("=" * 80)
print()
print(f"📊 Дані: {len(lval)} trials (2 epochs → 15 epochs)")
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

# DSS кореляція (В Optuna: minimize DSS, тому нижчий DSS = краще)
# DSS values негативні, тому нижчий (більш негативний) = краще
# Для кореляції з final (lower is better): inverse DSS
rho_dss, p_dss = spearmanr(-dss, final)

print(f"   L_val (2 epochs)  ↔  Final (15 epochs):")
print(f"      ρ = {rho_lval:.4f}")
print(f"      p-value = {p_lval:.4f}")
print(f"      {'✅ Значуща!' if p_lval < 0.05 else '❌ Не значуща'}")
print()

print(f"   DSS (2 epochs)    ↔  Final (15 epochs):")
print(f"      ρ = {rho_dss:.4f}")  
print(f"      p-value = {p_dss:.4f}")
print(f"      {'✅ Значуща!' if p_dss < 0.05 else '❌ Не значуща'}")
print()

delta_rho = rho_lval - rho_dss
print(f"   📈 РІЗНИЦЯ: ρ(L_val) - ρ(DSS) = {delta_rho:+.4f}")
if abs(delta_rho) < 0.05:
    print(f"      ≈ Практично однакові")
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
dss_ranks = np.argsort(np.argsort(-dss))  # 0 = best (lower DSS objective)
final_ranks = np.argsort(np.argsort(final))  # 0 = best

lval_stability = (lval_ranks == final_ranks).sum() / len(lval) * 100
dss_stability = (dss_ranks == final_ranks).sum() / len(dss) * 100

print(f"   L_val:  {lval_stability:.1f}% ({int(lval_stability/100*50)}/50 моделей)")
print(f"   DSS:    {dss_stability:.1f}% ({int(dss_stability/100*50)}/50 моделей)")
print()

if lval_stability > dss_stability:
    print(f"   ✅ L_val стабільніша на {lval_stability - dss_stability:.1f}%")
elif dss_stability > lval_stability:
    print(f"   ❌ DSS стабільніша на {dss_stability - lval_stability:.1f}%")
else:
    print(f"   ≈ Однакова стабільність")
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
    topk_dss = set(np.argsort(-dss)[:k])
    
    overlap_lval = len(topk_lval & topk_final)
    overlap_dss = len(topk_dss & topk_final)
    
    print(f"   TOP-{k}:")
    print(f"      L_val:  {overlap_lval}/{k} моделей ({overlap_lval/k*100:.0f}%)")
    print(f"      DSS:    {overlap_dss}/{k} моделей ({overlap_dss/k*100:.0f}%)")
    
    if overlap_lval > overlap_dss:
        print(f"      ✅ L_val краща (+{overlap_lval - overlap_dss} моделей)")
    elif overlap_dss > overlap_lval:
        print(f"      ❌ DSS краща (+{overlap_dss - overlap_lval} моделей)")
    else:
        print(f"      ≈ Однакові")
    print()

# ============================================
# 4. ДЕТАЛІ ТОП-10
# ============================================
print("=" * 80)
print("4️⃣  ДЕТАЛІ ТОП-10 МОДЕЛЕЙ (за final loss)")
print("=" * 80)
print()

# Справжній топ-10
true_top10 = np.argsort(final)[:10]

print("   Rank | Trial | Final   | L_val  | L_val R | DSS R  | Знайшли?")
print("   -----|-------|---------|--------|---------|--------|----------")

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
    
    print(f"   #{rank:2d}   | {trial:3d}   | {final_val:.4f} | {lval_val:.4f} | #{lval_rank:3d}    | #{dss_rank:3d}   | {best}")

print()

# ============================================
# ФІНАЛЬНИЙ ВИСНОВОК
# ============================================
print("=" * 80)
print("🎯 ФІНАЛЬНИЙ ВИСНОВОК")
print("=" * 80)
print()

if abs(delta_rho) < 0.05:
    print("📊 Кореляція: L_val і DSS показують ПРАКТИЧНО ОДНАКОВУ кореляцію")
    print()
    print("   Висновок:")
    print("   → DSS НЕ додає значного покращення")
    print("   → Проста L_val достатня і простіша")
    print("   → Рекомендація: використовуй L_val")
elif delta_rho > 0.1:
    print("✅ L_val ЗНАЧНО КРАЩА за DSS!")
    print()
    print("   Висновок:")
    print("   → DSS додає шум замість покращення")
    print("   → Додаткові метрики (gap, loss_cv, grad_cv) не допомагають")
    print("   → Рекомендація: використовуй просту L_val")
elif delta_rho > 0:
    print("✅ L_val КРАЩА за DSS")
    print()
    print("   Висновок:")
    print("   → L_val простіша і точніша")
    print("   → DSS не виправдовує додаткову складність")
    print("   → Рекомендація: використовуй L_val")
elif delta_rho < -0.1:
    print("❌ DSS ЗНАЧНО КРАЩА за просту L_val!")
    print()
    print("   Висновок:")
    print("   → DSS успішно використовує training dynamics")
    print("   → Додаткові метрики корисні")
    print("   → Залишай DSS!")
else:
    print("❌ DSS трохи краща за L_val")
    print()
    print("   Висновок:")
    print("   → DSS додає невелике покращення")
    print("   → Але складність може не виправдовуватись")
    print("   → Рішення залежить від пріоритетів")

print()
print("=" * 80)
print()

# Додаткова статистика
print("📈 ДОДАТКОВА СТАТИСТИКА:")
print()
print(f"   L_val діапазон:  {lval.min():.3f} - {lval.max():.3f}")
print(f"   DSS діапазон:    {dss.min():.3f} - {dss.max():.3f}")
print(f"   Final діапазон:  {final.min():.4f} - {final.max():.4f}")
print()
print(f"   L_val std:  {lval.std():.3f}")
print(f"   DSS std:    {dss.std():.3f}")
print(f"   Final std:  {final.std():.4f}")
print()
print("=" * 80)
