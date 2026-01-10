#!/usr/bin/env python3
"""
Аналіз стабілізації ранжування по епохах
Визначає на якій епосі порядок моделей відповідає фінальному
"""

import re
import numpy as np
from scipy.stats import spearmanr

print("=" * 80)
print("АНАЛІЗ СТАБІЛІЗАЦІЇ РАНЖУВАННЯ ПО ЕПОХАХ")
print("=" * 80)
print()

# ============================================
# ДАНІ З ЛОГУ FULL TRAINING (15 епох)
# ============================================

# Структура: [trial_idx][epoch] = val_loss
# Витягнуто з логу кожної моделі
# Epoch 1, 5, 10, 15 (з логу)

# Trial 0-49, фінальний результат
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

# Дані з логу: epoch 1, 5, 10, 15 (4 контрольні точки)
# Для спрощення візьмемо дані з логу напряму

# З логу витягнуто Val loss на кожній епосі для кожної моделі
# Формат: trial -> [epoch1, epoch5, epoch10, epoch15]

epoch_data = {
    0: [1.3088, 1.1544, 1.2227, 1.1008],    # Trial 0
    1: [1.2896, 1.1271, 1.1167, 1.0765],    # Trial 1
    2: [1.2969, 1.1357, 1.1345, 1.0879],    # Trial 2
    3: [1.4200, 1.1489, 1.1398, 1.0964],    # Trial 3
    4: [1.3330, 1.1423, 1.2147, 1.0976],    # Trial 4
    5: [1.3128, 1.1117, 1.1327, 1.0693],    # Trial 5
    6: [1.5024, 1.1571, 1.1607, 1.0979],    # Trial 6
    7: [1.2948, 1.1054, 1.1217, 1.0750],    # Trial 7
    8: [1.2955, 1.1349, 1.1349, 1.0968],    # Trial 8
    9: [1.3068, 1.1357, 1.1357, 1.0814],    # Trial 9
    10: [1.3043, 1.1178, 1.1178, 1.0832],   # Trial 10
    11: [1.2851, 1.1092, 1.1178, 1.0773],   # Trial 11
    12: [1.2835, 1.1097, 1.1289, 1.0807],   # Trial 12
    13: [1.2895, 1.1074, 1.1275, 1.0751],   # Trial 13
    14: [1.2821, 1.1178, 1.1226, 1.0767],   # Trial 14
    15: [1.3000, 1.1260, 1.1451, 1.0809],   # Trial 15
    16: [1.3102, 1.1303, 1.1303, 1.0924],   # Trial 16
    17: [1.2949, 1.1116, 1.1207, 1.0760],   # Trial 17
    18: [1.2888, 1.1003, 1.0960, 1.0660],   # Trial 18 ← BEST
    19: [1.2989, 1.1084, 1.1166, 1.0772],   # Trial 19
    20: [1.3459, 1.1907, 1.2080, 1.1260],   # Trial 20
    21: [1.2877, 1.1150, 1.1161, 1.0800],   # Trial 21
    22: [1.2925, 1.1068, 1.1068, 1.0831],   # Trial 22
    23: [1.2834, 1.1039, 1.1139, 1.0736],   # Trial 23
    24: [1.2935, 1.1087, 1.1222, 1.0831],   # Trial 24
    25: [1.2841, 1.1027, 1.1027, 1.0686],   # Trial 25
    26: [1.3046, 1.1159, 1.1246, 1.0839],   # Trial 26
    27: [1.2956, 1.1070, 1.1152, 1.0830],   # Trial 27
    28: [1.3007, 1.1229, 1.1297, 1.0876],   # Trial 28
    29: [1.2871, 1.1021, 1.1108, 1.0714],   # Trial 29
    30: [1.3093, 1.1265, 1.1363, 1.0914],   # Trial 30
    31: [1.2917, 1.1089, 1.1191, 1.0777],   # Trial 31
    32: [1.2905, 1.1088, 1.1191, 1.0777],   # Trial 32
    33: [1.3021, 1.1152, 1.1268, 1.0838],   # Trial 33
    34: [1.2898, 1.1061, 1.1163, 1.0781],   # Trial 34
    35: [1.2858, 1.1028, 1.1028, 1.0726],   # Trial 35
    36: [1.2945, 1.1080, 1.1179, 1.0783],   # Trial 36
    37: [1.2843, 1.0995, 1.0995, 1.0664],   # Trial 37
    38: [1.2938, 1.1076, 1.1184, 1.0817],   # Trial 38
    39: [1.2853, 1.1014, 1.1014, 1.0678],   # Trial 39
    40: [1.3021, 1.1135, 1.1259, 1.0848],   # Trial 40
    41: [1.2887, 1.1063, 1.1063, 1.0759],   # Trial 41
    42: [1.2968, 1.1096, 1.1192, 1.0830],   # Trial 42
    43: [1.2842, 1.1024, 1.1024, 1.0734],   # Trial 43
    44: [1.2919, 1.1073, 1.1176, 1.0823],   # Trial 44
    45: [1.3569, 1.1781, 1.1925, 1.1137],   # Trial 45
    46: [1.3020, 1.1142, 1.1258, 1.0848],   # Trial 46
    47: [1.3006, 1.1179, 1.1284, 1.0877],   # Trial 47
    48: [1.2981, 1.1119, 1.1228, 1.0860],   # Trial 48
    49: [1.3078, 1.1232, 1.1341, 1.0886],   # Trial 49
}

# Перевіримо розміри
assert len(epoch_data) == 50
assert all(len(v) == 4 for v in epoch_data.values())

# Конвертуємо в numpy array: [trial, epoch]
n_trials = 50
epochs_checkpoints = [1, 5, 10, 15]
n_checkpoints = len(epochs_checkpoints)

# Матриця: [checkpoint_idx, trial] = val_loss
losses_by_epoch = np.zeros((n_checkpoints, n_trials))
for trial_idx in range(n_trials):
    for ep_idx in range(n_checkpoints):
        losses_by_epoch[ep_idx, trial_idx] = epoch_data[trial_idx][ep_idx]

# Фінальні ранги (ground truth)
final_ranks = np.argsort(np.argsort(final_losses))  # 0 = best

print("📊 ДАНІ:")
print(f"   Trials: {n_trials}")
print(f"   Epochs checkpoints: {epochs_checkpoints}")
print(f"   Final best: Trial 18 (loss = {final_losses[18]:.4f})")
print()

# ============================================
# 1. КОРЕЛЯЦІЯ ПО ЕПОХАХ
# ============================================
print("=" * 80)
print("1️⃣  SPEARMAN КОРЕЛЯЦІЯ З ФІНАЛЬНИМ РАНЖУВАННЯМ")
print("=" * 80)
print()

print("   Epoch | Spearman ρ | p-value  | Якість")
print("   ------|------------|----------|--------")

correlations = []
for ep_idx, epoch_num in enumerate(epochs_checkpoints):
    epoch_losses = losses_by_epoch[ep_idx, :]
    rho, pval = spearmanr(epoch_losses, final_losses)
    correlations.append(rho)
    
    if rho >= 0.8:
        quality = "🟢 Відмінна"
    elif rho >= 0.6:
        quality = "🟡 Хороша"
    elif rho >= 0.4:
        quality = "🟠 Помірна"
    else:
        quality = "🔴 Слабка"
    
    print(f"   {epoch_num:5d} | {rho:10.4f} | {pval:.6f} | {quality}")

print()

# ============================================
# 2. TOP-K OVERLAP ПО ЕПОХАХ
# ============================================
print("=" * 80)
print("2️⃣  TOP-K OVERLAP З ФІНАЛЬНИМ ТОП-K")
print("=" * 80)
print()

true_top10 = set(np.argsort(final_losses)[:10])
true_top5 = set(np.argsort(final_losses)[:5])
true_top3 = set(np.argsort(final_losses)[:3])

print("   Epoch | TOP-3   | TOP-5   | TOP-10")
print("   ------|---------|---------|--------")

for ep_idx, epoch_num in enumerate(epochs_checkpoints):
    epoch_losses = losses_by_epoch[ep_idx, :]
    
    top3_epoch = set(np.argsort(epoch_losses)[:3])
    top5_epoch = set(np.argsort(epoch_losses)[:5])
    top10_epoch = set(np.argsort(epoch_losses)[:10])
    
    overlap3 = len(top3_epoch & true_top3)
    overlap5 = len(top5_epoch & true_top5)
    overlap10 = len(top10_epoch & true_top10)
    
    print(f"   {epoch_num:5d} | {overlap3}/3 ({overlap3/3*100:.0f}%) | {overlap5}/5 ({overlap5/5*100:.0f}%) | {overlap10}/10 ({overlap10/10*100:.0f}%)")

print()

# ============================================
# 3. НАЙКРАЩА МОДЕЛЬ ПО ЕПОХАХ
# ============================================
print("=" * 80)
print("3️⃣  ЯКА МОДЕЛЬ ЛІДИРУЄ НА КОЖНІЙ ЕПОСІ?")
print("=" * 80)
print()

print("   Epoch | Лідер (Trial) | Val Loss | Final Rank | Дельта")
print("   ------|---------------|----------|------------|--------")

for ep_idx, epoch_num in enumerate(epochs_checkpoints):
    epoch_losses = losses_by_epoch[ep_idx, :]
    leader_idx = int(np.argmin(epoch_losses))
    leader_loss = epoch_losses[leader_idx]
    leader_final_rank = int(final_ranks[leader_idx] + 1)
    delta = leader_final_rank - 1
    
    status = "✅" if leader_final_rank <= 5 else ("⚠️" if leader_final_rank <= 10 else "❌")
    
    print(f"   {epoch_num:5d} | Trial {leader_idx:3d}     | {leader_loss:.4f}   | #{leader_final_rank:3d}        | {delta:+3d} {status}")

print()

# Відстежимо Trial 18 (фінальний winner)
print("🏆 ВІДСТЕЖЕННЯ Trial 18 (фінальний переможець):")
print()
print("   Epoch | Val Loss | Rank на епосі | Дельта від #1")
print("   ------|----------|---------------|---------------")

trial_18_idx = 18
for ep_idx, epoch_num in enumerate(epochs_checkpoints):
    epoch_losses = losses_by_epoch[ep_idx, :]
    trial_18_loss = epoch_losses[trial_18_idx]
    trial_18_rank_at_epoch = int(np.argsort(np.argsort(epoch_losses))[trial_18_idx] + 1)
    delta_from_top = trial_18_rank_at_epoch - 1
    
    status = "🏆" if trial_18_rank_at_epoch == 1 else ("✅" if trial_18_rank_at_epoch <= 3 else "⚠️")
    
    print(f"   {epoch_num:5d} | {trial_18_loss:.4f}   | #{trial_18_rank_at_epoch:3d}           | {delta_from_top:+3d} {status}")

print()

# ============================================
# 4. RANK STABILITY (коли ранки стабілізуються)
# ============================================
print("=" * 80)
print("4️⃣  СТАБІЛЬНІСТЬ РАНГІВ")
print("=" * 80)
print()

print("   Середня абсолютна різниця рангів з фінальним:")
print()
print("   Epoch | Mean Δ Rank | Median Δ | Max Δ | Стабільність")
print("   ------|-------------|----------|-------|---------------")

for ep_idx, epoch_num in enumerate(epochs_checkpoints):
    epoch_losses = losses_by_epoch[ep_idx, :]
    epoch_ranks = np.argsort(np.argsort(epoch_losses))
    
    rank_diffs = np.abs(epoch_ranks - final_ranks)
    mean_diff = np.mean(rank_diffs)
    median_diff = np.median(rank_diffs)
    max_diff = np.max(rank_diffs)
    
    if mean_diff < 5:
        stability = "🟢 Висока"
    elif mean_diff < 10:
        stability = "🟡 Середня"
    elif mean_diff < 15:
        stability = "🟠 Низька"
    else:
        stability = "🔴 Дуже низька"
    
    print(f"   {epoch_num:5d} | {mean_diff:11.2f} | {median_diff:8.0f} | {max_diff:5.0f} | {stability}")

print()

# ============================================
# 5. КРИТИЧНА ЕПОХА
# ============================================
print("=" * 80)
print("🎯 ВИСНОВОК: КРИТИЧНА ЕПОХА")
print("=" * 80)
print()

# Знайти епоху де кореляція >= 0.7
critical_epoch = None
for ep_idx, epoch_num in enumerate(epochs_checkpoints):
    if correlations[ep_idx] >= 0.7:
        critical_epoch = epoch_num
        break

if critical_epoch:
    print(f"✅ РАНЖУВАННЯ СТАБІЛІЗУЄТЬСЯ НА ЕПОСІ {critical_epoch}")
    print()
    print(f"   Spearman ρ = {correlations[epochs_checkpoints.index(critical_epoch)]:.4f}")
    print()
    print("   Рекомендація:")
    print(f"   → Використовуй EPOCHS_PER_TRIAL = {critical_epoch} для DSS/proxy")
    print(f"   → Це дасть значно кращу кореляцію ніж 2 епохи")
else:
    print("❌ РАНЖУВАННЯ НЕ СТАБІЛІЗУЄТЬСЯ НАВІТЬ ПІСЛЯ 15 ЕПОХ!")
    print()
    print("   Це може означати:")
    print("   1. Датасет занадто малий (2000 samples)")
    print("   2. Архітектури занадто схожі за якістю")
    print("   3. Noise в даних домінує над signal")
    print()
    print("   Рекомендація:")
    print("   → Збільш FULL_MAX_SAMPLES до 5000+")
    print("   → Або збільш FULL_EPOCHS до 30")

print()

# Найкраща епоха за TOP-10 overlap
epoch_losses_ep10 = losses_by_epoch[2, :]  # epoch 10
top10_ep10 = set(np.argsort(epoch_losses_ep10)[:10])
overlap_ep10 = len(top10_ep10 & true_top10)

print("📊 ДОДАТКОВО:")
print()
print(f"   Epoch 1:  ρ = {correlations[0]:.3f}, TOP-10 overlap = {len(set(np.argsort(losses_by_epoch[0, :])[:10]) & true_top10)}/10")
print(f"   Epoch 5:  ρ = {correlations[1]:.3f}, TOP-10 overlap = {len(set(np.argsort(losses_by_epoch[1, :])[:10]) & true_top10)}/10")
print(f"   Epoch 10: ρ = {correlations[2]:.3f}, TOP-10 overlap = {overlap_ep10}/10")
print(f"   Epoch 15: ρ = {correlations[3]:.3f}, TOP-10 overlap = {len(set(np.argsort(losses_by_epoch[3, :])[:10]) & true_top10)}/10")
print()

print("=" * 80)
