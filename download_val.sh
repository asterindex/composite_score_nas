#!/bin/bash

# Скрипт для завантаження валідаційного датасету VisDrone2019-DET
# Автор: Анатолій Кот

echo "📥 Завантаження VisDrone2019-DET-val..."

cd data

# Валідаційний датасет
VAL_URL="https://drive.usercontent.google.com/download?id=1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59&export=download&authuser=0&confirm=t"

if [ ! -f "VisDrone2019-DET-val.zip" ]; then
    echo "⬇️  Завантаження валідаційного датасету..."
    curl -L -C - "$VAL_URL" -o VisDrone2019-DET-val.zip
    
    if [ $? -eq 0 ]; then
        echo "✅ Валідаційний датасет завантажено"
    else
        echo "❌ Помилка при завантаженні"
        exit 1
    fi
else
    echo "✓ Валідаційний датасет вже завантажено"
fi

# Розпакування
if [ ! -d "val" ]; then
    echo "📦 Розпакування..."
    unzip -q VisDrone2019-DET-val.zip
    
    # Реорганізація
    mkdir -p val
    mv VisDrone2019-DET-val/images val/
    mv VisDrone2019-DET-val/annotations val/
    rm -rf VisDrone2019-DET-val
    
    echo "✅ Датасет готовий до використання"
else
    echo "✓ Валідаційний датасет вже розпаковано"
fi

cd ..

echo ""
echo "📊 Статистика датасету:"
echo "   Train: $(ls data/train/images | wc -l) зображень"
echo "   Val: $(ls data/val/images | wc -l) зображень"
echo ""
echo "✅ Готово! Можете запускати synthesis_universal.py"
