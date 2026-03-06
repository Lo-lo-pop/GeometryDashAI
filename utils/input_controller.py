import keyboard
import time
import threading
from typing import Optional

class InputController:
    def __init__(self):
        self.is_jumping = False
        self.jump_lock = threading.Lock()
    
    def jump(self, duration: float = 0.1):
        """Нажатие пробела для прыжка"""
        with self.jump_lock:
            if not self.is_jumping:
                self.is_jumping = True
                keyboard.press('space')
                time.sleep(duration)
                keyboard.release('space')
                self.is_jumping = False
    
    def hold_jump(self):
        """Удержание прыжка (для длинных прыжков)"""
        keyboard.press('space')
    
    def release_jump(self):
        """Отпускание прыжка"""
        keyboard.release('space')
    
    def perform_action(self, action: int):
        """
        action: 0 = ничего не делать, 1 = прыгнуть
        """
        if action == 1:
            # Запускаем прыжок в отдельном потоке, чтобы не блокировать
            threading.Thread(target=self.jump, daemon=True).start()
    
    def reset(self):
        """Сброс всех клавиш"""
        keyboard.release('space')
        self.is_jumping = False

    def restart_level(self):
        """Рестарт уровня (обычно R в Geometry Dash)"""
        keyboard.press('r')
        time.sleep(0.1)
        keyboard.release('r')
        time.sleep(1.0)  # Ждем загрузки