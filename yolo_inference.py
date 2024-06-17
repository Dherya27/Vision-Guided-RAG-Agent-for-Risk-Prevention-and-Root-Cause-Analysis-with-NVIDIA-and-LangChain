
from ultralytics import YOLO
from PIL import Image
import tempfile

model = YOLO("runs/detect/yolov8n_plant_2/weights/best.pt")

def detect_disease(uploaded_image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        img = Image.open(uploaded_image)
        img.save(tmp.name, format="JPEG")
        results = model(tmp.name)
        print(results)
        
    unique_labels = set()

    for result in results:
        for box in result.boxes:
            unique_labels.add(model.names[int(box.cls)])
    print(unique_labels)

    return list(unique_labels)[0]

# disease = detect_disease("test/8aa78fd5c6c0cec2.jpg")
# print(disease)

