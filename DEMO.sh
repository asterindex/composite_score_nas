#!/bin/bash
# Демо-скрипт для Composite Score NAS

echo "======================================"
echo "  Composite Score NAS - Демо"
echo "======================================"
echo ""

# Кольори для виводу
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}1. Інформація про проект${NC}"
python3 main.py --mode info
echo ""
read -p "Натисніть Enter для продовження..."
echo ""

echo -e "${BLUE}2. Швидкий тест (5 trials)${NC}"
echo "Запуск: python3 main.py --mode fast"
python3 main.py --mode fast
echo ""

echo -e "${GREEN}✅ Тест завершено!${NC}"
echo ""
echo "Результати збережено в: output/"
echo ""

echo -e "${BLUE}3. Аналіз результатів${NC}"
python3 analyze.py
echo ""

echo -e "${GREEN}✅ Демо завершено!${NC}"
echo ""
echo "Для повного експерименту (30 trials):"
echo "  python3 main.py --mode full"
echo ""
echo "Для очищення результатів:"
echo "  python3 main.py --mode clean --confirm"
echo ""
