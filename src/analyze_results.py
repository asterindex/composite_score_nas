"""
Аналіз результатів експерименту
Порівняння DSS vs baseline, візуалізація, статистика

Автор: Анатолій Кот
Дата: 2026-01-24
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from synthesis_universal import RESULTS_DIR, CHECKPOINT_FILE


def load_study():
    """Завантаження Optuna study"""
    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(f"Не знайдено файл study: {CHECKPOINT_FILE}")
    
    with open(CHECKPOINT_FILE, 'rb') as f:
        return pickle.load(f)


def analyze_convergence(study):
    """Аналіз конвергенції оптимізації"""
    
    trials = study.trials
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    
    # Best value so far
    best_values = []
    current_best = float('inf')
    for v in values:
        current_best = min(current_best, v)
        best_values.append(current_best)
    
    # Візуалізація
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Trial values
    axes[0].plot(trial_numbers, values, 'o-', alpha=0.6, label='Trial value')
    axes[0].plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best so far')
    axes[0].axvline(x=10, color='green', linestyle='--', label='Warmup end')
    axes[0].set_xlabel('Trial number')
    axes[0].set_ylabel('Objective value')
    axes[0].set_title('Convergence Plot')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Improvement over warmup
    if len(values) > 10:
        warmup_best = min(values[:10])
        improvements = [(warmup_best - v) / abs(warmup_best) * 100 for v in values[10:]]
        axes[1].plot(range(11, len(values)+1), improvements, 'o-')
        axes[1].axhline(y=0, color='red', linestyle='--')
        axes[1].set_xlabel('Trial number')
        axes[1].set_ylabel('Improvement over warmup (%)')
        axes[1].set_title('DSS Improvement')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'convergence.png', dpi=150)
    print(f"✅ Збережено: {RESULTS_DIR / 'convergence.png'}")


def analyze_hyperparameters(study):
    """Аналіз важливості гіперпараметрів"""
    
    # Топ-10 trials
    trials_sorted = sorted(study.trials, key=lambda t: t.value)
    top_trials = trials_sorted[:10]
    
    # Збір параметрів
    params_analysis = {}
    
    for trial in top_trials:
        for key, value in trial.params.items():
            if key not in params_analysis:
                params_analysis[key] = []
            params_analysis[key].append(value)
    
    # Візуалізація частоти
    print(f"\n{'='*60}")
    print("📊 Аналіз топ-10 архітектур")
    print(f"{'='*60}")
    
    for param, values in params_analysis.items():
        if 'filter_' in param or 'kernel_' in param:
            continue  # Skip individual layer params
        
        if isinstance(values[0], (int, float)):
            print(f"\n{param}:")
            print(f"   Mean: {np.mean(values):.3f}")
            print(f"   Std: {np.std(values):.3f}")
        else:
            from collections import Counter
            freq = Counter(values)
            print(f"\n{param}:")
            for val, count in freq.most_common():
                percentage = (count / len(values)) * 100
                print(f"   {val}: {count}/10 ({percentage:.0f}%)")


def analyze_architecture_patterns(study):
    """Аналіз архітектурних паттернів"""
    
    trials_sorted = sorted(study.trials, key=lambda t: t.value)
    top_trials = trials_sorted[:10]
    
    print(f"\n{'='*60}")
    print("🏗️  Архітектурні паттерни (топ-10)")
    print(f"{'='*60}")
    
    # Depth analysis
    depths = [t.params['n_blocks'] for t in top_trials]
    print(f"\n📐 Глибина (n_blocks):")
    from collections import Counter
    depth_freq = Counter(depths)
    for depth, count in sorted(depth_freq.items()):
        print(f"   {depth} blocks: {count}/10")
    
    # Filter patterns
    print(f"\n🔢 Паттерни фільтрів:")
    for i, trial in enumerate(top_trials[:5], 1):
        n_blocks = trial.params['n_blocks']
        filters = [trial.params[f'filter_{j}'] for j in range(n_blocks)]
        kernels = [trial.params[f'kernel_{j}'] for j in range(n_blocks)]
        print(f"   #{i}: filters={filters}, kernels={kernels}")


def compare_with_baseline(study):
    """Порівняння DSS з baseline (validation loss)"""
    
    # Warmup trials (baseline)
    warmup_trials = study.trials[:10]
    warmup_best = min(warmup_trials, key=lambda t: t.value)
    
    # DSS trials
    dss_trials = study.trials[10:]
    if len(dss_trials) > 0:
        dss_best = min(dss_trials, key=lambda t: t.value)
        
        print(f"\n{'='*60}")
        print("⚖️  Baseline vs DSS")
        print(f"{'='*60}")
        print(f"\n🔵 Baseline (validation loss):")
        print(f"   Best trial: #{warmup_best.number}")
        print(f"   Value: {warmup_best.value:.4f}")
        
        print(f"\n🟢 DSS (stability-aware):")
        print(f"   Best trial: #{dss_best.number}")
        print(f"   Value: {dss_best.value:.4f}")
        
        improvement = ((warmup_best.value - dss_best.value) / abs(warmup_best.value)) * 100
        print(f"\n📈 Покращення: {improvement:+.2f}%")


def save_analysis_report(study):
    """Збереження детального звіту"""
    
    trials_sorted = sorted(study.trials, key=lambda t: t.value)
    
    report = {
        'summary': {
            'total_trials': len(study.trials),
            'best_value': study.best_value,
            'best_trial': study.best_trial.number
        },
        'top_10': [
            {
                'rank': i,
                'trial_number': t.number,
                'value': t.value,
                'params': t.params
            }
            for i, t in enumerate(trials_sorted[:10], 1)
        ]
    }
    
    report_file = RESULTS_DIR / 'analysis_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Звіт збережено: {report_file}")


def main():
    """Основний пайплайн аналізу"""
    
    print("🔬 Аналіз результатів експерименту")
    print(f"{'='*60}\n")
    
    # Завантаження
    study = load_study()
    print(f"✅ Завантажено study з {len(study.trials)} trials")
    
    # Аналізи
    analyze_convergence(study)
    analyze_hyperparameters(study)
    analyze_architecture_patterns(study)
    compare_with_baseline(study)
    save_analysis_report(study)
    
    print(f"\n{'='*60}")
    print("✅ Аналіз завершено!")
    print(f"📊 Результати у: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
