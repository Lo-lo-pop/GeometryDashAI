import cv2
import numpy as np
import mss
from config.settings import config

class ScreenCapture:  # <-- Убедись, что класс назван ТАК
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = {
            "left": config.SCREEN_REGION[0],
            "top": config.SCREEN_REGION[1],
            "width": config.SCREEN_REGION[2],
            "height": config.SCREEN_REGION[3]
        }
    
    def capture(self) -> np.ndarray:
        """Захват экрана и конвертация в grayscale"""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        resized = cv2.resize(gray, config.FRAME_SIZE)
        return resized
    
    def capture_color(self) -> np.ndarray:
        """Цветной захват для визуализации"""
        screenshot = self.sct.grab(self.monitor)
        return np.array(screenshot)[:, :, :3]
    
    def detect_death(self, frame: np.ndarray) -> bool:
        """Простая детекция смерти"""
        mean_val = np.mean(frame)
        return mean_val < 10 or mean_val > 250
    
    def detect_progress(self, current: np.ndarray, previous: np.ndarray) -> float:
        """Определение прогресса"""
        current = current.astype(np.float32)
        previous = previous.astype(np.float32)
        diff = cv2.absdiff(current, previous)
        movement = np.sum(diff) / diff.size
        return movement

    def release(self):
        self.sct.close()