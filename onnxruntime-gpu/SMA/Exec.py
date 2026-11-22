from ultralytics import YOLO

YOLO("yolo11n.pt").export(format="onnx", dynamic=True, simplify=True)
