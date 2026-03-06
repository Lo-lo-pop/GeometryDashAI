#!/usr/bin/env python3
import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_menu():
    print("""
    🎮 GEOMETRY DASH AI 🧠
    =======================
    1. 🚀 Начать обучение (с нуля)
    2. ▶️  Продолжить обучение (загрузить чекпоинт)
    3. 🎮 Запустить обученную модель
    4. 📊 Просмотр статистики
    5. ❌ Выход
    """)

def main():
    from config.settings import config
    from ai.trainer import Trainer
    
    print(f"📁 Проект настроен. Данные сохраняются в: {config.DATA_DIR}")
    
    while True:
        print_menu()
        choice = input("Выберите действие (1-5): ").strip()
        
        if choice == '1':
            print("🚀 Запуск обучения с нуля...")
            trainer = Trainer(resume=False)
            trainer.train()
            
        elif choice == '2':
            print("▶️  Возобновление обучения...")
            trainer = Trainer(resume=True)
            trainer.train()
            
        elif choice == '3':
            print("🎮 Запуск игры...")
            trainer = Trainer(resume=True)
            episodes = int(input("Сколько эпизодов запустить? (по умолчанию 10): ") or 10)
            trainer.evaluate(episodes=episodes)
            
        elif choice == '4':
            print("📊 Статистика обучения:")
            checkpoints = os.listdir(config.CHECKPOINT_DIR) if os.path.exists(config.CHECKPOINT_DIR) else []
            print(f"   Сохраненные модели: {len(checkpoints)}")
            for cp in checkpoints:
                print(f"   - {cp}")
                
        elif choice == '5':
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор!")

if __name__ == "__main__":
    main()