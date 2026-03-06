import time
import torch
import numpy as np
from typing import Tuple, Dict
from core.vision import VisionSystem
from utils.input_controller import InputController
from config.settings import config

class GeometryDashEnv:
    def __init__(self, async_vision: bool = True, vision_fps: int = 180):
        self.vision = VisionSystem(async_mode=async_vision, target_fps=vision_fps)
        self.controller = InputController()
        
        self.episode_start_time = None
        self.max_progress = 0
        self.steps_since_progress = 0
        self.prev_distance = 999.0
        
    def reset(self) -> torch.Tensor:
        self.controller.reset()
        time.sleep(0.5)
        self.controller.restart_level()
        
        self.vision.reset()
        self.episode_start_time = time.time()
        self.max_progress = 0
        self.steps_since_progress = 0
        self.prev_distance = 999.0
        
        state = self.vision.get_state()
        time.sleep(0.1)
        return state
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        self.controller.perform_action(action)
        time.sleep(1/60)
        
        next_state = self.vision.get_state()
        is_alive = self.vision.is_player_alive()
        
        reward = 0.0
        info = {}
        
        if not is_alive:
            reward = config.REWARD_DEATH
            done = True
            info['cause'] = 'death'
        else:
            # Базовая награда за выживание
            reward += config.REWARD_ALIVE
            
            # УЛУЧШЕННАЯ награда за прогресс (движение вправо)
            if self.vision.detections and self.vision.detections.get('player'):
                px, py, pw, ph = self.vision.detections['player']
                # Награда за X-координату (прогресс вправо)
                progress = px / 1000.0  # Нормализация
                reward += progress * 0.5
                
                # Награда за избежание столкновений
                dist = self.vision.get_distance_to_obstacle()
                if dist < self.prev_distance and dist < 100:
                    reward += 0.2  # Приближаемся к препятствию (готовимся прыгнуть)
                if dist < 50 and action == 1:  # Прыгнули вовремя!
                    reward += 1.0
                
                self.prev_distance = dist
            
            # Проверка застревания
            self.steps_since_progress += 1
            if self.vision.get_progress_reward() > 0.1:
                self.steps_since_progress = 0
            
            if self.steps_since_progress > 300:
                reward = config.REWARD_DEATH
                done = True
                info['cause'] = 'stuck'
            else:
                done = False
        
        info['time'] = time.time() - self.episode_start_time
        info['distance'] = self.vision.get_distance_to_obstacle() if self.vision.detections else 999.0
        
        return next_state, reward, done, info
    
    def close(self):
        self.vision.release()
        self.controller.reset()