import random
from collections import deque, namedtuple
import torch
import numpy as np

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    """Хранение и сэмплирование игрового опыта"""
    def __init__(self, capacity: int = 100000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Сохранение перехода"""
        # Конвертация в CPU для экономии памяти GPU
        if torch.is_tensor(state):
            state = state.cpu()
        if torch.is_tensor(next_state):
            next_state = next_state.cpu()
            
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> tuple:
        """Случайная выборка для обучения (breaks correlation)"""
        batch = random.sample(self.memory, batch_size)
        
        states = torch.cat([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.cat([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)
    
    def save(self, path: str):
        """Сохранение памяти на диск"""
        torch.save(list(self.memory), path)
    
    def load(self, path: str):
        """Загрузка памяти"""
        self.memory = deque(torch.load(path), maxlen=self.memory.maxlen)


class PrioritizedReplayMemory(ReplayMemory):
    """Приоритизированная память (важные переходы чаще)"""
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
    
    def push(self, state, action, reward, next_state, done):
        # Максимальный приоритет для новых переходов
        max_prio = max(self.priorities) if self.priorities else 1.0
        
        super().push(state, action, reward, next_state, done)
        self.priorities.append(max_prio)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Сэмплирование с учетом приоритетов"""
        if len(self.memory) == 0:
            return None
        
        # Вычисление вероятностей
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Сэмплирование
        indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        # Веса для коррекции смещения
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        # Распаковка
        states = torch.cat([s.state for s in samples])
        actions = torch.LongTensor([s.action for s in samples])
        rewards = torch.FloatTensor([s.reward for s in samples])
        next_states = torch.cat([s.next_state for s in samples])
        dones = torch.FloatTensor([s.done for s in samples])
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices, priorities):
        """Обновление приоритетов после обучения"""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio + 1e-6  # epsilon для стабильности