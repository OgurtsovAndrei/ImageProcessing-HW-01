import pyautogui
import time
import sys

# Интервал в секундах (600 секунд = 10 минут)
INTERVAL = 60

print(f"Скрипт запущен. Компьютер не уснет. Нажмите Ctrl+C для остановки.")
print(f"Имитация движения мыши каждые {INTERVAL} секунд.")

try:
    while True:
        # Получаем текущую позицию мыши
        x, y = pyautogui.position()

        # Сдвигаем курсор на 1 пиксель вниз
        pyautogui.moveRel(0, 1)

        # Возвращаем курсор на 1 пиксель вверх
        pyautogui.moveRel(0, -1)

        # Печатаем текущее время и позицию, чтобы видеть, что скрипт работает
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{current_time}] Мышь сдвинута на ({x}, {y})...", end='\r')

        # Ждем заданный интервал
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nСкрипт остановлен пользователем.")
    sys.exit(0)