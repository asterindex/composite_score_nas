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
python main.py --mode info
echo ""
read -p "Натисніть Enter для продовження..."
echo ""

echo -e "${BLUE}2. Швидкий тест (5 trials)${NC}"
echo "Запуск: python main.py --mode synthesis --trials 5 --quick"
python main.py --mode synthesis --trials 5 --quick
echo ""

echo -e "${GREEN}✅ Тест завершено!${NC}"
echo ""
echo "Результати збережено в: output/"
echo ""

echo -e "${BLUE}3. Аналіз результатів${NC}"
python main.py --mode analyze
echo ""

echo -e "${GREEN}✅ Демо завершено!${NC}"
echo ""
echo "Для повного експерименту (30 trials):"
echo "  python main.py --mode synthesis"
echo ""
echo "Для очищення результатів:"
echo "  python main.py --mode clean --confirm"
echo ""
