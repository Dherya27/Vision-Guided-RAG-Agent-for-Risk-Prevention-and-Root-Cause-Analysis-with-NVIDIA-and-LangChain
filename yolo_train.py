import torch
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

results = model.train(
   data='data.yaml',
   epochs=15,
   batch=8,
   name='yolov8n_plant_2')