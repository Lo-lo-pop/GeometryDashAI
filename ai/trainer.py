import time
import torch
import os
import json
from datetime import datetime
from ai.agent import DQNAgent
from core.game_env import GeometryDashEnv
from config.settings import config

class Trainer:
    def __init__(self, resume: bool = False):
        self.env = GeometryDashEnv(async_vision=True, vision_fps=180)
        self.agent = DQNAgent(use_auxiliary=True)
        
        self.log_file = os.path.join(config.LOG_DIR, "training_log.txt")
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'best_reward': float('-inf'),
            'total_steps': 0,
            'start_time': datetime.now().isoformat()
        }
        
        # Загрузка статистики если есть
        stats_path = os.path.join(config.LOG_DIR, "stats.json")
        if resume and os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            self.log("📂 Статистика загружена")
        
        if resume:
            self.load_checkpoint()
    
    def log(self, message: str, level: str = "INFO"):
        """Логирование с цветами и уровнями"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Цвета для консоли
        colors = {
            "INFO": "\033[94m",      # Синий
            "SUCCESS": "\033[92m",   # Зеленый
            "WARNING": "\033[93m",   # Желтый
            "ERROR": "\033[91m",     # Красный
            "BEST": "\033[95m",      # Пурпурный
            "RESET": "\033[0m"
        }
        
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        
        log_msg = f"[{timestamp}] {color}[{level}]{reset} {message}"
        print(log_msg)
        
        # В файл без цветов
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
    
    def format_time(self, seconds: float) -> str:
        """Форматирование времени"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def train(self):
        start_time = time.time()
        
        self.log("=" * 60, "INFO")
        self.log("🚀 УСЛОЖНЕННОЕ ОБУЧЕНИЕ ЗАПУЩЕНО", "SUCCESS")
        self.log(f"🎯 Асинхронное зрение: 180 FPS", "INFO")
        self.log(f"🧠 Архитектура: ResNet + Auxiliary Learning", "INFO")
        self.log(f"💾 Чекпоинты: {config.CHECKPOINT_DIR}", "INFO")
        self.log(f"📝 Логи: {self.log_file}", "INFO")
        self.log("=" * 60, "INFO")
        self.log("⚡ Нажми Ctrl+C для остановки (прогресс сохранится)", "WARNING")
        self.log("=" * 60, "INFO")
        
        try:
            for episode in range(config.EPISODES):
                ep_start = time.time()
                state = self.env.reset()
                self.agent.reset_hidden()
                
                episode_reward = 0
                episode_q_loss = 0
                episode_aux_loss = 0
                steps = 0
                done = False
                
                while not done and steps < config.MAX_STEPS_PER_EPISODE:
                    action = self.agent.select_action(state, training=True)
                    next_state, reward, done, info = self.env.step(action)
                    
                    self.agent.store_transition(state, action, reward, next_state, float(done))
                    
                    if steps % 4 == 0:
                        q_loss, aux_loss = self.agent.learn_with_auxiliary()
                        episode_q_loss += q_loss
                        episode_aux_loss += aux_loss
                    
                    if self.agent.steps_done % config.TARGET_UPDATE == 0:
                        self.agent.update_target_network()
                    
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    self.stats['total_steps'] += 1
                
                # Статистика эпизода
                self.agent.decay_epsilon()
                self.stats['episode_rewards'].append(episode_reward)
                self.stats['episode_lengths'].append(steps)
                
                ep_time = time.time() - ep_start
                total_time = time.time() - start_time
                avg_reward = sum(self.stats['episode_rewards'][-100:]) / min(len(self.stats['episode_rewards']), 100)
                
                # Прогресс-бар
                progress = (episode + 1) / config.EPISODES * 100
                bar_len = 30
                filled = int(bar_len * (episode + 1) / config.EPISODES)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                # Логирование
                self.log(f"EP {episode:5d} |{bar}| {progress:5.1f}%", "INFO")
                self.log(f"  🎮 Reward: {episode_reward:8.2f} | Avg100: {avg_reward:8.2f}", "INFO")
                self.log(f"  📊 Steps: {steps:5d} | Time: {self.format_time(ep_time)}", "INFO")
                self.log(f"  🧠 Q-Loss: {episode_q_loss/max(steps,1):.4f} | Aux: {episode_aux_loss/max(steps,1):.4f}", "INFO")
                self.log(f"  🎲 Epsilon: {self.agent.epsilon:.3f} | Total steps: {self.stats['total_steps']}", "INFO")
                
                # Проверка лучшего результата
                is_best = episode_reward > self.stats['best_reward']
                if is_best:
                    self.stats['best_reward'] = episode_reward
                    self.save_checkpoint('best_model.pth')
                    self.log(f"  🏆 НОВЫЙ РЕКОРД! Reward: {episode_reward:.2f}", "BEST")
                
                # Автосохранение
                if episode % config.SAVE_INTERVAL == 0:
                    self.save_checkpoint(f'checkpoint_ep{episode}.pth')
                    self.save_checkpoint('latest_model.pth')
                    self.save_stats()
                    self.log(f"  💾 Автосохранение (эпизод {episode})", "SUCCESS")
                
                # Итоговое время
                self.log(f"  ⏱️  Общее время: {self.format_time(total_time)}", "INFO")
                self.log("-" * 60, "INFO")
                
        except KeyboardInterrupt:
            self.log("\n" + "=" * 60, "WARNING")
            self.log("⏹️ ОБУЧЕНИЕ ПРЕРВАНО ПОЛЬЗОВАТЕЛЕМ", "WARNING")
            self.log("💾 Сохраняю прогресс...", "WARNING")
            
            # Экстренное сохранение
            self.save_checkpoint('interrupted.pth')
            self.save_stats()
            
            self.log("✅ Прогресс сохранен!", "SUCCESS")
            self.log(f"📂 Чекпоинт: interrupted.pth", "SUCCESS")
            self.log(f"📊 Лучший результат: {self.stats['best_reward']:.2f}", "BEST")
            self.log("🚀 Для продолжения запусти с опцией 'Продолжить обучение'", "INFO")
            self.log("=" * 60, "WARNING")
            
        finally:
            self.env.close()
            self.log("✅ Обучение завершено!", "SUCCESS")
    
    def save_checkpoint(self, filename: str):
        path = os.path.join(config.CHECKPOINT_DIR, filename)
        self.agent.save(path)
    
    def save_stats(self):
        path = os.path.join(config.LOG_DIR, "stats.json")
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_checkpoint(self, filename: str = 'latest_model.pth'):
        path = os.path.join(config.CHECKPOINT_DIR, filename)
        if os.path.exists(path):
            self.agent.load(path)
            self.log(f"📂 Загружен чекпоинт: {filename}", "SUCCESS")
        else:
            self.log(f"⚠️ Чекпоинт не найден: {filename}", "WARNING")
    
    def evaluate(self, episodes: int = 10):
        self.log("🎮 ЗАПУСК ОЦЕНКИ", "INFO")
        self.agent.epsilon = 0
        
        for ep in range(episodes):
            state = self.env.reset()
            self.agent.reset_hidden()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 1000:
                action = self.agent.select_action(state, training=False)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                steps += 1
            
            self.log(f"🎯 Эпизод {ep+1}/{episodes}: Reward = {total_reward:.2f}, Steps = {steps}", "INFO")