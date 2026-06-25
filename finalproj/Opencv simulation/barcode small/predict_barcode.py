from ultralytics import YOLO

model = YOLO(r"runs/detect/hub_det/weights/best.pt")

model.predict(
    source=r"check_pic",
    conf=0.25,
    save=True
)