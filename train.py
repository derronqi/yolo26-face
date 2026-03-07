from ultralytics import YOLO
# Load a model
model = YOLO('yolo26m-pose.pt')
# Train the model
model.train(data='widerface.yaml', epochs=300, imgsz=640, batch=16, patience=0, device=2)
