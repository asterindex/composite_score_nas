#!/usr/bin/env python3
"""
Composite Score NAS - Main Entry Point

Запуск експериментів з різними конфігураціями через CLI.

Приклади використання:
    # Повний експеримент (30 trials)
    python main.py --mode synthesis --trials 30
    
    # Швидкий тест (5 trials)
    python main.py --mode synthesis --trials 5 --quick
    
    # Тренування топ-3 моделей
    python main.py --mode train-top3
    
    # Аналіз результатів
    python main.py --mode analyze
    
    # Очищення результатів
    python main.py --mode clean
"""

import argparse
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone


def setup_args():
    """Налаштування аргументів командного рядка"""
    parser = argparse.ArgumentParser(
        description='Composite Score NAS - Detection Stability Score для Bayesian Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади:
  %(prog)s --mode fast                         # Швидкий тест (5 trials)
  %(prog)s --mode full                         # Повний експеримент (30 trials)
  %(prog)s --mode synthesis --trials 10        # Користувацька кількість
  %(prog)s --mode train-top3                   # Тренування топ-3
  %(prog)s --mode analyze                      # Аналіз результатів
  %(prog)s --mode clean --confirm              # Очистити output/
        """
    )
    
    # Основні параметри
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['fast', 'full', 'synthesis', 'train-top3', 'analyze', 'clean', 'info'],
        help='Режим роботи'
    )
    
    # Параметри для synthesis
    parser.add_argument(
        '--trials',
        type=int,
        default=30,
        help='Кількість trials для Bayesian Optimization (default: 30)'
    )
    
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Кількість warmup trials для калібрування (default: 10)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Кількість епох для кожного trial (default: 1)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=700,
        help='Розмір тренувальної підмножини (default: 700)'
    )
    
    parser.add_argument(
        '--val-samples',
        type=int,
        default=200,
        help='Розмір валідаційної підмножини (default: 200)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed для відтворюваності (default: 42)'
    )
    
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Продовжити попередній експеримент (якщо є checkpoint)'
    )
    
    # Параметри для clean
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Підтвердити видалення без запиту'
    )
    
    # Загальні параметри
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Детальний вивід'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Директорія для результатів (default: output/)'
    )
    
    return parser.parse_args()


def print_header(title):
    """Друк заголовку"""
    width = 70
    print("\n" + "="*width)
    print(f"  {title}")
    print("="*width + "\n")


def print_config(args):
    """Друк конфігурації"""
    print("📋 Конфігурація:")
    print(f"   Режим:              {args.mode}")
    if hasattr(args, 'trials'):
        print(f"   Trials:             {args.trials}")
    if hasattr(args, 'warmup'):
        print(f"   Warmup trials:      {args.warmup}")
    if hasattr(args, 'epochs'):
        print(f"   Epochs per trial:   {args.epochs}")
    if hasattr(args, 'samples'):
        print(f"   Train samples:      {args.samples}")
    if hasattr(args, 'val_samples'):
        print(f"   Val samples:        {args.val_samples}")
    if hasattr(args, 'seed'):
        print(f"   Random seed:        {args.seed}")
    if hasattr(args, 'output_dir'):
        print(f"   Output dir:         {args.output_dir}")
    if hasattr(args, 'resume'):
        print(f"   Resume:             {'Так' if args.resume else 'Ні'}")
    print()


def mode_fast(args):
    """Режим швидкого тесту (5 trials)"""
    print_header("⚡ Швидкий тест (Fast Mode)")
    
    # Встановлюємо параметри для швидкого режиму
    args.trials = 5
    args.warmup = 3
    args.samples = 200
    args.val_samples = 50
    
    print("⚡ Швидкий режим активовано!")
    print(f"   Trials: {args.trials}")
    print(f"   Warmup: {args.warmup}")
    print(f"   Train samples: {args.samples}")
    print(f"   Val samples: {args.val_samples}")
    print(f"   Очікуваний час: ~3-5 хвилин\n")
    
    # Викликаємо synthesis
    mode_synthesis(args)


def mode_full(args):
    """Режим повного експерименту (30 trials)"""
    print_header("🔬 Повний експеримент (Full Mode)")
    
    # Встановлюємо параметри для повного режиму
    args.trials = 30
    args.warmup = 10
    args.samples = 700
    args.val_samples = 200
    
    print("🔬 Повний режим активовано!")
    print(f"   Trials: {args.trials}")
    print(f"   Warmup: {args.warmup}")
    print(f"   Train samples: {args.samples}")
    print(f"   Val samples: {args.val_samples}")
    print(f"   Очікуваний час: ~15-18 хвилин\n")
    
    # Викликаємо synthesis
    mode_synthesis(args)


def mode_synthesis(args):
    """Режим синтезу архітектур"""
    # Якщо викликається напряму (не через fast/full)
    if args.mode == 'synthesis':
        print_header("🔬 Detection Stability Score - Synthesis")
        print_config(args)
    
    # Перевірка dataset
    data_dir = Path('data')
    if not (data_dir / 'train').exists() or not (data_dir / 'val').exists():
        print("❌ Помилка: Датасет VisDrone2019-DET не знайдено!")
        print("   Завантажте датасет:")
        print("   1. https://github.com/VisDrone/VisDrone-Dataset")
        print("   2. Розпакуйте у папку data/")
        sys.exit(1)
    
    # Перевірка checkpoint
    output_dir = Path(args.output_dir)
    checkpoint_file = output_dir / 'optuna_study.pkl'
    
    if checkpoint_file.exists() and not args.resume:
        print("⚠️  Знайдено попередній checkpoint!")
        print(f"   {checkpoint_file}")
        print("\n   Опції:")
        print("   1. Видаліть output/ для чистого запуску")
        print("   2. Використайте --resume для продовження")
        sys.exit(1)
    
    # Імпорт та запуск
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"🚀 Запуск синтезу: {timestamp}\n")
    
    # Передача параметрів через environment variables
    os.environ['NAS_N_TRIALS'] = str(args.trials)
    os.environ['NAS_N_WARMUP'] = str(args.warmup)
    os.environ['NAS_EPOCHS_PER_TRIAL'] = str(args.epochs)
    os.environ['NAS_MAX_SAMPLES'] = str(args.samples)
    os.environ['NAS_VAL_SUBSET'] = str(args.val_samples)
    os.environ['NAS_SEED'] = str(args.seed)
    os.environ['NAS_OUTPUT_DIR'] = args.output_dir
    
    try:
        # Динамічний імпорт synthesis_universal з src/
        import sys
        sys.path.insert(0, 'src')
        import synthesis_universal
        print("\n✅ Синтез завершено!")
        print(f"   Результати збережено в: {args.output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Помилка під час синтезу: {e}")
        sys.exit(1)


def mode_train_top3(args):
    """Режим тренування топ-3 моделей"""
    print_header("🏋️  Повне тренування топ-3 архітектур")
    
    # Перевірка results
    output_dir = Path(args.output_dir)
    study_file = output_dir / 'optuna_study.pkl'
    
    if not study_file.exists():
        print("❌ Помилка: Не знайдено optuna_study.pkl")
        print("   Спочатку запустіть: python main.py --mode synthesis")
        sys.exit(1)
    
    print(f"📂 Завантаження study з: {study_file}\n")
    
    os.environ['NAS_OUTPUT_DIR'] = args.output_dir
    
    try:
        import sys
        sys.path.insert(0, 'src')
        import train_top3_models
        print("\n✅ Тренування завершено!")
        print(f"   Моделі збережено в: {args.output_dir}/trained_models/")
        
    except Exception as e:
        print(f"\n❌ Помилка під час тренування: {e}")
        sys.exit(1)


def mode_analyze(args):
    """Режим аналізу результатів"""
    print_header("📊 Аналіз результатів експерименту")
    
    output_dir = Path(args.output_dir)
    study_file = output_dir / 'optuna_study.pkl'
    
    if not study_file.exists():
        print("❌ Помилка: Не знайдено optuna_study.pkl")
        print("   Спочатку запустіть: python main.py --mode synthesis")
        sys.exit(1)
    
    print(f"📂 Аналіз study з: {study_file}\n")
    
    os.environ['NAS_OUTPUT_DIR'] = args.output_dir
    
    try:
        import sys
        sys.path.insert(0, 'src')
        import analyze_results
        print("\n✅ Аналіз завершено!")
        print(f"   Графіки збережено в: {args.output_dir}/")
        
    except Exception as e:
        print(f"\n❌ Помилка під час аналізу: {e}")
        sys.exit(1)


def mode_clean(args):
    """Режим очищення результатів"""
    print_header("🧹 Очищення результатів")
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"✅ Папка {args.output_dir}/ вже чиста (не існує)")
        return
    
    # Підрахунок файлів
    files = list(output_dir.rglob('*'))
    file_count = len([f for f in files if f.is_file()])
    
    print(f"📂 Знайдено файлів: {file_count}")
    print(f"   Шлях: {output_dir.absolute()}\n")
    
    if file_count > 0:
        print("   Буде видалено:")
        for f in sorted(output_dir.rglob('*'))[:10]:  # Показати перші 10
            if f.is_file():
                print(f"   - {f.relative_to(output_dir)}")
        if file_count > 10:
            print(f"   ... та ще {file_count - 10} файлів")
        print()
    
    if not args.confirm:
        response = input("⚠️  Підтвердіть видалення (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Скасовано")
            return
    
    try:
        shutil.rmtree(output_dir)
        print(f"✅ Папку {args.output_dir}/ видалено")
        
    except Exception as e:
        print(f"❌ Помилка при видаленні: {e}")
        sys.exit(1)


def mode_info(args):
    """Режим виводу інформації"""
    print_header("ℹ️  Composite Score NAS - Інформація")
    
    print("📦 Структура проекту:")
    print("""
    composite_score_nas/
    ├── main.py                  # 🆕 Головний скрипт запуску
    ├── src/                     # Код експерименту
    │   ├── synthesis_universal.py   # Пайплайн синтезу з DSS
    │   ├── train_top3_models.py     # Повне тренування топ-3
    │   ├── analyze_results.py       # Аналіз convergence
    │   └── dataset_utils.py         # Утиліти для VisDrone
    ├── requirements.txt         # Залежності
    ├── data/                    # VisDrone2019-DET
    │   ├── train/
    │   └── val/
    └── output/                  # Результати експериментів
        ├── optuna_study.pkl
        ├── proxy_stats.json
        ├── synthesis_results.json
        └── experiment_*.log
    """)
    
    print("\n🚀 Швидкий старт:")
    print("   1. Завантажте датасет:")
    print("      https://github.com/VisDrone/VisDrone-Dataset")
    print()
    print("   2. Швидкий тест (5 trials, ~3-5 хв):")
    print("      python3 main.py --mode fast")
    print()
    print("   3. Повний експеримент (30 trials, ~15-18 хв):")
    print("      python3 main.py --mode full")
    print()
    print("   4. Тренуйте топ-3:")
    print("      python3 main.py --mode train-top3")
    print()
    print("   5. Аналізуйте результати:")
    print("      python3 main.py --mode analyze")
    print()
    
    print("\n📊 Detection Stability Score (DSS):")
    print("   DSS = 0.25·z(impr) + 0.20·z(L_val) + 0.15·z(loss_cv) +")
    print("         0.15·z(grad_cv) + 0.15·z(gap) + 0.05·z(L_tr) + 0.05·z(grad_norm)")
    print()
    print("   Компоненти:")
    print("   - impr:      покращення loss за епоху")
    print("   - L_val:     validation loss")
    print("   - loss_cv:   коефіцієнт варіації train loss")
    print("   - grad_cv:   коефіцієнт варіації градієнта")
    print("   - gap:       різниця val-train loss")
    print("   - L_tr:      фінальний train loss")
    print("   - grad_norm: середня норма градієнта")
    print()
    
    print("\n🔗 Посилання:")
    print("   GitHub:   https://github.com/asterindex/composite_score_nas")
    print("   VisDrone: https://github.com/VisDrone/VisDrone-Dataset")
    print("   Optuna:   https://optuna.org/")
    print()


def main():
    """Головна функція"""
    args = setup_args()
    
    # Маршрутизація по режимах
    modes = {
        'fast': mode_fast,
        'full': mode_full,
        'synthesis': mode_synthesis,
        'train-top3': mode_train_top3,
        'analyze': mode_analyze,
        'clean': mode_clean,
        'info': mode_info,
    }
    
    try:
        modes[args.mode](args)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Перервано користувачем")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n❌ Критична помилка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
