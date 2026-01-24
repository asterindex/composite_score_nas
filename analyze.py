#!/usr/bin/env python3
"""
Швидкий запуск аналізу результатів
Обгортка для main.py --mode analyze

Використання:
    python3 analyze.py
    python3 analyze.py --output-dir experiments/run1
"""

import sys
import subprocess

def main():
    """Запустити аналіз результатів"""
    # Передаємо всі аргументи до main.py
    args = ['python3', 'main.py', '--mode', 'analyze'] + sys.argv[1:]
    
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Перервано користувачем")
        sys.exit(130)

if __name__ == '__main__':
    main()
