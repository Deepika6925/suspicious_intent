from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")  # nano version for speed

def detect_people(frame):
    results = yolo(frame)
    boxes = []
    for r in results:
        for b in r.boxes.xyxy:
            boxes.append(tuple(map(int, b.tolist())))
    return boxes