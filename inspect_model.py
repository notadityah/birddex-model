# from ultralytics import YOLO

# # Load the model
# try:
#     model = YOLO("model.pt")
    
#     print("-" * 30)
#     print("Model Architecture Summary:")
#     print("-" * 30)
#     model.info()
    
#     print("\n" + "-" * 30)
#     print("Classes (What it detects):")
#     print("-" * 30)
#     print(model.names)
    
# except Exception as e:
#     print(f"Error loading model: {e}")


import onnx

model = onnx.load("model.onnx")

# Check model is valid
onnx.checker.check_model(model)

# Print a human-readable summary
print(onnx.helper.printable_graph(model.graph))

# Inspect inputs/outputs
for inp in model.graph.input:
    print("Input:", inp.name, inp.type.tensor_type.shape)

for out in model.graph.output:
    print("Output:", out.name, out.type.tensor_type.shape)