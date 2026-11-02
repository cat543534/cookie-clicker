from ultralytics import YOLO

model = YOLO("runs/detect/cookie_detector/weights/best.pt")
model.export(format="onnx", opset=12, simplify=True)