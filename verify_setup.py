from ultralytics import YOLO

# Load the YOLOv8 nano model
# First time only: this will auto-download yolov8n.pt (6MB)
model = YOLO("yolov8n.pt")

# Print confirmation
print("Model loaded successfully!")
print(f"Model type: {type(model)}")

# Check what classes this model can detect
print(f"\nThis model can detect {len(model.names)} different objects")
print(f"Some examples: {list(model.names.values())[:10]}")
