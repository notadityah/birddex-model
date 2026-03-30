from ultralytics import YOLO

# Load the trained model
model = YOLO("best.pt")

# Export to ONNX format
# - simplify: applies ONNX-simplifier to reduce graph complexity
# - opset: ONNX opset version (17 is widely supported)
# - dynamic: enables dynamic batch size input
model.export(
    format="onnx",
    simplify=True,
    opset=17,
    dynamic=False,   # set True if you want variable batch sizes
    imgsz=640,       # input image size (must match training size)
)

print("Export complete! Model saved as best.onnx")