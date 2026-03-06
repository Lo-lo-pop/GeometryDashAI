import cv2
import numpy as np
import torch
import torch.nn as nn
import threading
import time
from typing import List, Tuple, Dict
from collections import deque

class LearnableVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        ).to(self.device)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1),
        ).to(self.device)
        
        self.learning_rate = 0.001
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        print(f"🎮 LearnableVision on: {self.device}")
        
    def forward(self, x):
        x = x.to(self.device)
        features = self.encoder(x)
        segmentation = self.decoder(features)
        return segmentation, features
    
    def learn(self, frame, prev_frame=None):
        self.train()
        
        # GPU: конвертация и нормализация
        frame_t = torch.FloatTensor(frame).permute(2, 0, 1).unsqueeze(0) / 255.0
        frame_t = frame_t.to(self.device)
        
        seg, features = self.forward(frame_t)
        
        loss = torch.mean((seg - 0.5) ** 2)
        
        if prev_frame is not None:
            with torch.no_grad():
                prev_t = torch.FloatTensor(prev_frame).permute(2, 0, 1).unsqueeze(0) / 255.0
                prev_t = prev_t.to(self.device)
                _, prev_feat = self.forward(prev_t)
            
            temporal_loss = torch.mean((features - prev_feat) ** 2) * 0.1
            loss = loss + temporal_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Возвращаем на CPU для OpenCV
        return seg.detach().cpu().squeeze().permute(1, 2, 0).numpy()

class VisionSystem:
    def __init__(self, async_mode: bool = True, target_fps: int = 180, viz_fps: int = 30):
        self.async_mode = async_mode
        self.target_fps = target_fps
        self.viz_fps = viz_fps
        self.frame_time = 1.0 / target_fps
        self.viz_time = 1.0 / viz_fps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 VisionSystem using: {self.device}")
        
        self.frame_buffer = deque(maxlen=4)
        for _ in range(4):
            self.frame_buffer.append(np.zeros((84, 84), dtype=np.float32))
        
        self.prev_frame = None
        self.latest_state = None
        self.running = False
        self.capture_thread = None
        self.viz_thread = None
        self.show_viz = True
        
        self.learnable = LearnableVision()
        self.use_learned = True
        self.learning_steps = 0
        
        self.latest_color_frame = None
        self.segmentation_map = None
        self.detections = {}
        
        # GPU буферы для быстрого resize
        self.gpu_pool = torch.nn.AdaptiveAvgPool2d((84, 84))
        
        if self.async_mode:
            self.start_async()
    
    def start_async(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        if self.show_viz:
            self.viz_thread = threading.Thread(target=self._viz_loop, daemon=True)
            self.viz_thread.start()
        
        print(f"🚀 GPU-ускоренное зрение: {self.target_fps} FPS")
    
    def _capture_loop(self):
        import mss
        from config.settings import config
        
        sct = mss.mss()
        monitor = {
            "left": config.SCREEN_REGION[0],
            "top": config.SCREEN_REGION[1],
            "width": config.SCREEN_REGION[2],
            "height": config.SCREEN_REGION[3]
        }
        
        frame_count = 0
        prev_color = None
        
        while self.running:
            start = time.perf_counter()
            
            # Захват (CPU)
            screenshot = sct.grab(monitor)
            color = np.array(screenshot)[:, :, :3]
            
            # GPU: улучшение контраста через CUDA если доступно
            if self.device.type == 'cuda':
                color_gpu = torch.cuda.FloatTensor(color).div_(255.0)
                # Простое CLAHE на CPU пока, потом перенесем на GPU
                color_enhanced = self._enhance_contrast_gpu(color)
            else:
                color_enhanced = self._enhance_contrast(color)
            
            # Обучение (GPU)
            if self.use_learned and frame_count % 3 == 0:
                seg_map = self.learnable.learn(color_enhanced, prev_color)
                self.segmentation_map = seg_map
                self.detections = self._extract_from_segmentation(seg_map)
                prev_color = color_enhanced.copy()
                self.learning_steps += 1
            else:
                gray = cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
                self.detections = self._classic_detect(color_enhanced, gray)
            
            # GPU: создание state
            if self.segmentation_map is not None:
                # Быстрый resize на GPU
                seg_gpu = torch.FloatTensor(self.segmentation_map).permute(2, 0, 1).unsqueeze(0)
                seg_gpu = seg_gpu.to(self.device)
                
                player_gpu = seg_gpu[0, 1]
                obstacle_gpu = seg_gpu[0, 2]
                combined_gpu = torch.maximum(player_gpu, obstacle_gpu)
                
                # Resize bilinear на GPU
                combined_resized = torch.nn.functional.interpolate(
                    combined_gpu.unsqueeze(0).unsqueeze(0), 
                    size=(84, 84), 
                    mode='bilinear'
                ).squeeze()
                
                frame_processed = combined_resized.cpu().numpy()
            else:
                gray = cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
                frame_processed = cv2.resize(gray, (84, 84)).astype(np.float32) / 255.0
            
            self.frame_buffer.append(frame_processed)
            
            if frame_count % 6 == 0:
                self.latest_color_frame = color_enhanced
            
            # GPU: создание stacked state
            stacked = np.stack(self.frame_buffer, axis=0)
            self.latest_state = torch.FloatTensor(stacked).unsqueeze(0).to(self.device)
            
            frame_count += 1
            
            # Поддержание FPS
            elapsed = time.perf_counter() - start
            sleep_time = max(0, self.frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        sct.close()
    
    def _enhance_contrast_gpu(self, img: np.ndarray) -> np.ndarray:
        """GPU-ускоренное улучшение контраста"""
        # Пока на CPU, можно добавить CUDA kernels
        return self._enhance_contrast(img)
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _extract_from_segmentation(self, seg_map: np.ndarray) -> Dict:
        detections = {'player': None, 'spikes': [], 'platforms': [], 'orbs': []}
        
        player_mask = (seg_map[:, :, 1] > 0.5).astype(np.uint8) * 255
        spike_mask = (seg_map[:, :, 2] > 0.5).astype(np.uint8) * 255
        plat_mask = (seg_map[:, :, 3] > 0.5).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                detections['player'] = (x, y, w, h)
                break
        
        contours, _ = cv2.findContours(spike_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 30:
                x, y, w, h = cv2.boundingRect(cnt)
                detections['spikes'].append((x, y, w, h))
        
        contours, _ = cv2.findContours(plat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                detections['platforms'].append((x, y, w, h))
        
        return detections
    
    def _classic_detect(self, color: np.ndarray, gray: np.ndarray) -> Dict:
        detections = {'player': None, 'spikes': [], 'platforms': [], 'orbs': []}
        
        _, bright_mask = cv2.threshold(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 80 < area < 1500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                if 0.6 < aspect < 1.4:
                    roi = gray[y:y+h, x:x+w]
                    if np.mean(roi) < 200:
                        detections['player'] = (x, y, w, h)
                        break
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > w and h < 100:
                    detections['spikes'].append((x, y, w, h))
        
        return detections
    
    def _viz_loop(self):
        cv2.namedWindow("Geometry Dash AI Vision", cv2.WINDOW_NORMAL)
        
        while self.running and self.show_viz:
            start = time.perf_counter()
            
            if self.latest_color_frame is not None:
                display = self.latest_color_frame.copy()
                d = self.detections
                
                # Упрощенная визуализация (без наложения сегментации)
                if d.get('player'):
                    x, y, w, h = d['player']
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 3)
                    cv2.putText(display, "PLAYER", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                for (x, y, w, h) in d.get('spikes', [])[:5]:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                for (x, y, w, h) in d.get('platforms', [])[:3]:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.putText(display, f"GPU: {self.device} | FPS: {self.target_fps}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Player: {'YES' if d.get('player') else 'NO'} | Learned: {self.learning_steps}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Geometry Dash AI Vision", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Простая задержка без сложной логики
            elapsed = time.perf_counter() - start
            sleep_time = max(0, 1.0/30.0 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cv2.destroyAllWindows()
    
    def get_state(self) -> torch.Tensor:
        if self.async_mode and self.latest_state is not None:
            # Уже на GPU!
            return self.latest_state.clone()
        return torch.zeros(1, 4, 84, 84).to(self.device)
    
    def get_progress_reward(self) -> float:
        if len(self.frame_buffer) < 2:
            return 0.0
        
        current = self.frame_buffer[-1]
        prev = self.frame_buffer[-2]
        diff = np.abs(current - prev)
        movement = np.mean(diff)
        
        return movement * 5.0 if movement > 0.001 else 0.0
    
    def is_player_alive(self) -> bool:
        return self.detections.get('player') is not None
    
    def get_distance_to_obstacle(self) -> float:
        if not self.detections.get('player'):
            return 999.0
        
        px = self.detections['player'][0]
        min_dist = 999.0
        
        for (sx, sy, sw, sh) in self.detections.get('spikes', []):
            dist = sx - px
            if 0 < dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def reset(self):
        self.frame_buffer.clear()
        for _ in range(4):
            self.frame_buffer.append(np.zeros((84, 84), dtype=np.float32))
        self.prev_frame = None
        self.latest_color_frame = None
        self.segmentation_map = None
        self.detections = {}
        if not self.async_mode:
            self.latest_state = None
    
    def release(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.viz_thread:
            self.viz_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        # Очистка GPU
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()