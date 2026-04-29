from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Baseline detection
model.predict(
    "https://ultralytics.com/images/bus.jpg",
    save=True,
    imgsz=320,
    conf=0.25,
    project="runs/detect",
    name="baseline",
    exist_ok=True
)

# Detection + heatmap overlay
model.predict(
    "https://ultralytics.com/images/bus.jpg",
    save=True,
    imgsz=320,
    conf=0.25,
    HeatMap=True,
    HMO=0.8,  # stronger for visibility
    project="runs/detect",
    name="heatmap",
    exist_ok=True
)

print("Saved:")
print(" - runs/detect/baseline/bus.jpg")
print(" - runs/detect/heatmap/bus.jpg")