import os
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    # Пути
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    CHECKPOINT_DIR: str = os.path.join(DATA_DIR, "checkpoints")
    LOG_DIR: str = os.path.join(DATA_DIR, "logs")
    
    # Размеры экрана игры (настрой под свое разрешение!)
    SCREEN_REGION: Tuple[int, int, int, int] = (4, 6, 800, 600)  # x, y, w, h
    
    # Параметры зрения
    FRAME_SIZE: Tuple[int, int] = (84, 84)  # Размер для нейросети (уменьшаем для скорости)
    FRAME_STACK: int = 4  # Количество кадров для временной зависимости
    
    # Параметры нейросети
    LEARNING_RATE: float = 0.0003
    GAMMA: float = 0.99  # Дисконтирование наград
    EPSILON_START: float = 1.0  # Для epsilon-greedy
    EPSILON_END: float = 0.01
    EPSILON_DECAY: float = 0.995
    BATCH_SIZE: int = 32
    MEMORY_SIZE: int = 100000  # Размер памяти опыта
    
    # Обучение
    EPISODES: int = 10000
    MAX_STEPS_PER_EPISODE: int = 5000
    SAVE_INTERVAL: int = 50  # Сохранять каждые N эпизодов
    TARGET_UPDATE: int = 1000  # Обновление target network
    
    # Награды
    REWARD_PROGRESS: float = 1.0  # За прогресс
    REWARD_DEATH: float = -10.0   # За смерть
    REWARD_ALIVE: float = 0.1     # За выживание (живи дольше = лучше)
    
    # Игровые действия
    ACTION_SPACE: int = 2  # [не прыгать, прыгнуть]
    
    def __post_init__(self):
        # Создание папок
        for path in [self.CHECKPOINT_DIR, self.LOG_DIR, os.path.join(self.DATA_DIR, "episodes")]:
            os.makedirs(path, exist_ok=True)

config = Config()