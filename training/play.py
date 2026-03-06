import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.trainer import Trainer

def main():
    print("🎮 Запуск обученного агента...")
    trainer = Trainer(resume=True)
    trainer.evaluate(episodes=100)

if __name__ == "__main__":
    main()