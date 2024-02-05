from ultralytics import YOLO

model = YOLO ('./4_webapp/models/yolov8s-pose.pt')

results = model(source=0, show=True, conf=0.3)