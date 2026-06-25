from pathlib import Path
from ultralytics import YOLO
import yaml

BASE = Path(__file__).resolve().parent
DATA_YAML = BASE / "dataset" / "data.yaml"

# בדיקות מהירות לפני אימון
print("BASE:", BASE)
print("DATA_YAML:", DATA_YAML)
assert DATA_YAML.exists(), f"Missing: {DATA_YAML}"

cfg = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
ds_path = BASE / cfg["path"]
train_dir = ds_path / cfg["train"]
val_dir = ds_path / cfg["val"]

print("Train images dir:", train_dir)
print("Val images dir:", val_dir)
assert train_dir.exists(), f"Missing train dir: {train_dir}"
assert val_dir.exists(), f"Missing val dir: {val_dir}"

# אימון
model = YOLO("yolov8n.pt")
model.train(
    data=str(DATA_YAML),
    imgsz=640,
    epochs=80,
    batch=4,
    device="cpu",
    name="hub_det"
)

