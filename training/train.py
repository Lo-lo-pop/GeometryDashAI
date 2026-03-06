import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.trainer import Trainer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Geometry Dash AI')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    args = parser.parse_args()
    
    trainer = Trainer(resume=args.resume)
    
    if args.episodes:
        from config.settings import config
        config.EPISODES = args.episodes
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⏹️ Обучение прервано пользователем")
        trainer.save_checkpoint('interrupted.pth')
        print("💾 Прогресс сохранен!")

if __name__ == "__main__":
    main()