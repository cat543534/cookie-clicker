import time

import cv2
import mss
import numpy as np
import onnxruntime as ort
import pyautogui
from pynput import keyboard as pynput_keyboard

# ---------------------------------------------------------------- настройки

# Путь к экспортированной ONNX-модели
MODEL_PATH = "runs/detect/cookie_detector/weights/best.onnx"

# ВАЖНО: порядок классов должен совпадать с data.yaml, по которому учили модель!
CLASS_NAMES = {
    0: "game",
    1: "upgrade",
    2: "skip"  # третий класс — кнопка пропуска рекламы
}

# Количество кликов (для "upgrade" — количество точек по вертикали блока)
CLICKS_BY_CLASS = {
    "upgrade": 10,
    "game": 10,
    "skip": 1,  # одного клика по центру кнопки пропуска достаточно
}

# Низкий общий порог: сначала находим всё подряд,
# а фильтруем по классам ниже (CONFIDENCE_BY_CLASS)
CONFIDENCE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.45
INPUT_SIZE = 640  # размер входа модели (как при экспорте)

# Порог уверенности для клика по каждому классу.
# Для "skip" он снижен: кнопка пропуска часто детектируется слабее,
# чем игровые блоки (мало примеров в обучающей выборке)
CONFIDENCE_BY_CLASS = {
    "game": 0.7,
    "upgrade": 0.7,
    "skip": 0.35,
}

# Печать всех детекций в консоль (для отладки).
# После починки можно поставить False
DEBUG_DETECTIONS = True

# Смещение точки клика по горизонтали внутри блока:
# 0.0 — у самого левого края, 0.5 — центр
LEFT_OFFSET_RATIO = 0.15

# Во сколько раз увеличиваем расстояние между точками клика
STEP_SPACING = 1.7

# Пауза между кликами по разным точкам
PAUSE_BETWEEN_POINTS = 0.25

# Флаг паузы: True — бот стоит, False — работает
paused = False
# Время последнего переключения (защита от повторов при зажатом пробеле)
_last_toggle_time = 0.0


# ---------------------------------------------------------------- детектор

class YOLOOnnx:
    """YOLO-детектор на чистом onnxruntime (без ultralytics/torch).

    Совместим со стандартным экспортом YOLOv8/YOLO11:
    выход модели имеет форму (1, 4+nc, N) — боксы cx, cy, w, h и
    оценки классов для N кандидатов.
    """

    def __init__(self, model_path, conf_threshold, iou_threshold,
                 input_size=640, class_names=None):
        # GPU-провайдеры, если есть, иначе CPU
        available = ort.get_available_providers()
        providers = [
            p for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                        "CoreMLExecutionProvider")
            if p in available
        ]
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print("ONNX Runtime провайдеры:", self.session.get_providers())

        # Статический вход модели задаёт размер, иначе берём INPUT_SIZE
        in_shape = self.session.get_inputs()[0].shape
        self.input_h = in_shape[2] if isinstance(in_shape[2], int) else input_size
        self.input_w = in_shape[3] if isinstance(in_shape[3], int) else input_size

        # Проверка: сколько классов реально знает загруженная модель.
        # Выход (1, 4+nc, N): если вторая размерность 6 — классов два,
        # если 7 — три. Сразу видно, переобучена ли модель под "skip"
        out_shape = self.session.get_outputs()[0].shape
        print("Форма выхода модели:", out_shape)
        if len(out_shape) == 3 and isinstance(out_shape[1], int):
            n_model_classes = out_shape[1] - 4
            print(f"Классов в модели: {n_model_classes}")
            if class_names and n_model_classes != len(class_names):
                print(
                    "ВНИМАНИЕ! Модель обучена на другом числе классов, "
                    f"чем указано в CLASS_NAMES ({len(class_names)}). "
                    "Нужно переобучить модель или исправить CLASS_NAMES."
                )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or {}

    def detect(self, image_bgr):
        """Скриншот BGR -> список детекций:
        [{"box": [x1, y1, x2, y2], "confidence": float,
          "class_id": int, "class_name": str}, ...]
        """
        blob, ratio, pad_x, pad_y = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs[0], ratio, pad_x, pad_y, image_bgr.shape)

    def _preprocess(self, image_bgr):
        """Letterbox до размера входа + нормализация -> NCHW float32."""
        h, w = image_bgr.shape[:2]
        ratio = min(self.input_w / w, self.input_h / h)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

        resized = cv2.resize(image_bgr, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_x = (self.input_w - new_w) // 2
        pad_y = (self.input_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        blob = img.transpose(2, 0, 1)[None]  # HWC -> CHW -> NCHW
        return np.ascontiguousarray(blob), ratio, pad_x, pad_y

    def _postprocess(self, output, ratio, pad_x, pad_y, orig_shape):
        pred = output[0]                    # (4+nc, N)
        if pred.shape[0] < pred.shape[1]:   # приводим к (N, 4+nc)
            pred = pred.T

        boxes_cxcywh = pred[:, :4]          # cx, cy, w, h в пикселях входа
        class_scores = pred[:, 4:]

        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        keep = confidences > self.conf_threshold
        boxes_cxcywh, class_ids, confidences = (
            boxes_cxcywh[keep], class_ids[keep], confidences[keep])

        if len(boxes_cxcywh) == 0:
            return []

        # NMS (cv2.dnn ждёт боксы в формате x, y, w, h)
        nms_boxes = np.stack([
            boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2,
            boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2,
            boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 3],
        ], axis=1)
        indices = cv2.dnn.NMSBoxes(
            nms_boxes.tolist(), confidences.tolist(),
            self.conf_threshold, self.iou_threshold)
        indices = np.array(indices).reshape(-1)

        orig_h, orig_w = orig_shape[:2]
        detections = []
        for i in indices:
            cx, cy, bw, bh = boxes_cxcywh[i]

            # Обратно из координат входа в координаты скриншота
            x1 = (cx - bw / 2 - pad_x) / ratio
            y1 = (cy - bh / 2 - pad_y) / ratio
            x2 = (cx + bw / 2 - pad_x) / ratio
            y2 = (cy + bh / 2 - pad_y) / ratio
            x1, x2 = np.clip([x1, x2], 0, orig_w)
            y1, y2 = np.clip([y1, y2], 0, orig_h)

            class_id = int(class_ids[i])
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(confidences[i]),
                "class_id": class_id,
                "class_name": self.class_names.get(class_id, str(class_id)),
            })
        return detections


model = YOLOOnnx(
    MODEL_PATH,
    conf_threshold=CONFIDENCE_THRESHOLD,
    iou_threshold=NMS_IOU_THRESHOLD,
    input_size=INPUT_SIZE,
    class_names=CLASS_NAMES,
)
print("Соответствие классов:", model.class_names)


# ---------------------------------------------------------------- пауза

def toggle_pause():
    """Переключает паузу по нажатию пробела."""
    global paused, _last_toggle_time

    now = time.time()
    if now - _last_toggle_time < 0.3:  # игнорируем автоповтор клавиши
        return
    _last_toggle_time = now

    paused = not paused
    print(f"[{'ПАУЗА' if paused else 'РАБОТА'}] — пробел")


def _on_press(key):
    """Колбэк слушателя pynput: реагируем только на пробел."""
    if key == pynput_keyboard.Key.space:
        toggle_pause()


def start_key_listener():
    """Запускает фоновый слушатель клавиатуры (без прав администратора)."""
    listener = pynput_keyboard.Listener(on_press=_on_press)
    listener.daemon = True
    listener.start()
    return listener


def wait_if_paused():
    """Блокирует выполнение, пока стоит пауза."""
    while paused:
        time.sleep(0.1)


def interruptible_sleep(duration):
    """Обычный sleep, но мгновенно прерывается при постановке на паузу."""
    end = time.time() + duration
    while time.time() < end:
        if paused:
            return
        time.sleep(0.05)


key_listener = start_key_listener()
print("Бот запущен. Пробел — пауза/продолжение, Ctrl+C — выход.")


# ---------------------------------------------------------------- клики

def get_click_points(x1, y1, x2, y2, n_points):
    """Точки в левой части блока, распределённые снизу вверх."""
    width = x2 - x1
    height = y2 - y1

    click_x = x1 + width * LEFT_OFFSET_RATIO

    step = height / n_points * STEP_SPACING
    center_y = (y1 + y2) / 2

    ys = [
        center_y + (i - (n_points - 1) / 2) * step
        for i in range(n_points - 1, -1, -1)
    ]

    ys = [min(max(y, y1), y2) for y in ys]

    return [(int(click_x), int(y)) for y in ys]


def click_upgrade_block(monitor, box, n_points):
    """Кликает по левой части блока улучшений снизу вверх."""
    x1, y1, x2, y2 = box

    points = get_click_points(x1, y1, x2, y2, n_points)

    for px, py in points:
        wait_if_paused()

        screen_x = monitor["left"] + px
        screen_y = monitor["top"] + py

        pyautogui.click(x=screen_x, y=screen_y)
        interruptible_sleep(PAUSE_BETWEEN_POINTS)
        print(f"upgrade, координаты: ({screen_x}, {screen_y})")


def click_block_center(monitor, box, n_clicks, pause=0.1):
    """Кликает n_clicks раз в центр блока. Возвращает координаты клика."""
    x1, y1, x2, y2 = box
    center_x = monitor["left"] + int((x1 + x2) / 2)
    center_y = monitor["top"] + int((y1 + y2) / 2)

    for _ in range(n_clicks):
        wait_if_paused()
        pyautogui.click(x=center_x, y=center_y)
        interruptible_sleep(pause)

    return center_x, center_y


def find_and_click(sct, monitor):
    # Скриншот монитора в формате BGRA → BGR
    screenshot = np.array(sct.grab(monitor))
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Детекция объектов
    detections = model.detect(screenshot)

    # Отладка: показываем всё, что нашла модель, до фильтрации по порогам
    if DEBUG_DETECTIONS and detections:
        print("Детекции:")
        for det in detections:
            print(f"  {det['class_name']}: conf={det['confidence']:.2f}, "
                  f"box={det['box']}")

    clicked = False

    targets = []
    for det in detections:
        if det["class_name"] not in CLICKS_BY_CLASS:
            continue
        # Фильтр по пер-классовому порогу уверенности
        min_conf = CONFIDENCE_BY_CLASS.get(
            det["class_name"], CONFIDENCE_THRESHOLD)
        if det["confidence"] < min_conf:
            continue
        # «skip» обрабатываем первым (сначала закрываем рекламу),
        # остальные блоки сортируем по вертикали, снизу вверх
        priority = 0 if det["class_name"] == "skip" else 1
        targets.append((priority, det["box"][1], det))

    for _, _, det in sorted(targets, key=lambda t: (t[0], -t[1])):
        class_name = det["class_name"]

        if class_name == "upgrade":
            # Кликаем по левой части блока улучшений снизу вверх
            click_upgrade_block(monitor, det["box"], CLICKS_BY_CLASS[class_name])
        else:
            # «skip» и остальные классы — клик в центр блока
            center_x, center_y = click_block_center(
                monitor, det["box"], CLICKS_BY_CLASS[class_name])
            print(
                f"Класс: {class_name}, уверенность: {det['confidence']:.2f}, "
                f"координаты: ({center_x}, {center_y})"
            )

        clicked = True

    return clicked


with mss.mss() as sct:
    monitor = sct.monitors[1]  # Первый монитор

    try:
        while True:
            wait_if_paused()

            if find_and_click(sct, monitor):
                interruptible_sleep(1)
            else:
                interruptible_sleep(0.5)
    except KeyboardInterrupt:
        print("Остановка по Ctrl+C.")
    finally:
        key_listener.stop()