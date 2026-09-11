import os
import sys

sys.path.insert(0, os.getcwd())


from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n-face.yaml')
# Train the model
model.train(
        data='widerface.yaml', 
        epochs=500, 
        imgsz=640, 
        batch=16, 
        patience=0, 
        device=2, 
        end2end=False, 
        optimizer='AdamW', 
        lr0=0.001,
        cos_lr=True,
        single_cls=True)
