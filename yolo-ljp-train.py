from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data='yolo-ljp.yaml', workers=0, epochs=50, batch=8)