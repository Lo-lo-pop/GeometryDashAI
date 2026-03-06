import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Tuple
from ai.model import GeometryDashNet, SimpleDQN  # <-- ИСПРАВЛЕНО: model, не model_advanced
from ai.memory import ReplayMemory, PrioritizedReplayMemory
from config.settings import config

class DQNAgent:
    def __init__(self, use_lstm: bool = False, use_per: bool = True, use_auxiliary: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = config.ACTION_SPACE
        self.use_auxiliary = use_auxiliary
        
        # ВСЕГДА используем усложненную сеть
        self.policy_net = GeometryDashNet(self.action_space).to(self.device)
        self.target_net = GeometryDashNet(self.action_space).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.memory = PrioritizedReplayMemory(config.MEMORY_SIZE) if use_per else ReplayMemory(config.MEMORY_SIZE)
        
        self.epsilon = config.EPSILON_START
        self.steps_done = 0
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        self.steps_done += 1
        
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_space)
        
        with torch.no_grad():
            state = state.to(self.device)
            output = self.policy_net(state)
            if isinstance(output, tuple):
                q_values, _ = output
            else:
                q_values = output
            return q_values.argmax().item()
    
    def decay_epsilon(self):
        self.epsilon = max(config.EPSILON_END, self.epsilon * config.EPSILON_DECAY)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self) -> float:
        """Обычное обучение (для совместимости)"""
        if len(self.memory) < config.BATCH_SIZE:
            return 0.0
        
        if isinstance(self.memory, PrioritizedReplayMemory):
            batch = self.memory.sample(config.BATCH_SIZE)
            if batch is None:
                return 0.0
            states, actions, rewards, next_states, dones, weights, indices = batch
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(config.BATCH_SIZE)
            weights = torch.ones(config.BATCH_SIZE).to(self.device)
            indices = None
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Current Q
        policy_out = self.policy_net(states)
        if isinstance(policy_out, tuple):
            q_values, _ = policy_out
        else:
            q_values = policy_out
        
        current_q = q_values.gather(1, actions.unsqueeze(1))
        
        # Target Q
        with torch.no_grad():
            next_policy_out = self.policy_net(next_states)
            if isinstance(next_policy_out, tuple):
                next_q_policy, _ = next_policy_out
            else:
                next_q_policy = next_policy_out
            
            next_actions = next_q_policy.argmax(1)
            
            next_target_out = self.target_net(next_states)
            if isinstance(next_target_out, tuple):
                next_q_target, _ = next_target_out
            else:
                next_q_target = next_target_out
            
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * config.GAMMA * next_q
        
        # Loss
        loss = (weights * nn.SmoothL1Loss(reduction='none')(current_q.squeeze(), target_q)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()
        
        if indices is not None:
            with torch.no_grad():
                td_errors = torch.abs(current_q.squeeze() - target_q).cpu().numpy()
            self.memory.update_priorities(indices, td_errors)
        
        return loss.item()
    
    def learn_with_auxiliary(self) -> Tuple[float, float]:
        """Обучение с auxiliary task"""
        if len(self.memory) < config.BATCH_SIZE:
            return 0.0, 0.0
        
        batch = self.memory.sample(config.BATCH_SIZE)
        if batch is None:
            return 0.0, 0.0
        
        # Всегда 7 значений (weights и indices могут быть None для обычной памяти)
        if len(batch) == 7:
            states, actions, rewards, next_states, dones, weights, indices = batch
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = batch
            weights = torch.ones(config.BATCH_SIZE).to(self.device)
            indices = None
        
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # Forward с auxiliary
        q_values, aux_pred = self.policy_net(states, return_auxiliary=True)
        current_q = q_values.gather(1, actions.unsqueeze(1))
        
        # Target для Q
        with torch.no_grad():
            next_q, _ = self.target_net(next_states, return_auxiliary=True)
            next_q = next_q.max(1)[0]
            target_q = rewards + (1 - dones) * config.GAMMA * next_q
        
        # Q loss
        q_loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)
        
        # Auxiliary loss
        next_states_flat = next_states.view(states.size(0), -1)
        aux_loss = nn.MSELoss()(aux_pred, next_states_flat)
        
        # Combined loss
        total_loss = q_loss + 0.1 * aux_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()
        
        return q_loss.item(), aux_loss.item()
    
    def update_target_network(self):
        """Soft update target network"""
        tau = 0.001
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def reset_hidden(self):
        """Для совместимости"""
        pass
    
    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']