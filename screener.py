import mss
import mss.tools
import time
from datetime import datetime
import os

save_dir = "screenshots"
os.makedirs(save_dir, exist_ok=True)

print("Начинаю делать скриншоты каждые 5 секунд…")
print("Нажмите Ctrl+C для остановки\n")

count = 0

try:
    with mss.mss() as sct:
        # Основной монитор. Для всех мониторов можно использовать sct.monitors[0]
        monitor = sct.monitors[1]

        while True:
            screenshot = sct.grab(monitor)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            filename = os.path.join(save_dir, f"screenshot_{timestamp}.png")

            mss.tools.to_png(screenshot.rgb, screenshot.size, output=filename)

            count += 1
            print(f"[{count}] Сохранено: {filename}")

            time.sleep(5)

except KeyboardInterrupt:
    print(f"\nОстановлено. Всего сделано скриншотов: {count}")