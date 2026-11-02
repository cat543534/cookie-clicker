from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Загрузка предобученной модели
    model = YOLO('yolov8n.pt')

    # 2. Запуск обучения
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=10,
        patience=50,  # остановка, если нет улучшений 50 эпох подряд
        workers=0     # отключаем многопоточность (частые проблемы с этой опцией)
    )
