import torch
from ultralytics import YOLO

def train():
    # -------------------------------------------------------------------------
    # OPTION 1: Fine-Tuning (Recommended for Adding New Classes)
    # Start with a pretrained base model (e.g., yolo11s.pt) and train on your FULL dataset (old + new birds).
    # This prevents "catastrophic forgetting" where the model forgets old classes.
    # -------------------------------------------------------------------------
    print("Loading base model...")
    model = YOLO("yolo11s.pt")  # Load a pretrained model (recommended)

    # -------------------------------------------------------------------------
    # OPTION 2: Continue Training Existing Model (Only if classes remain exactly the same)
    # If you just want to improve detection of EXISTING birds with more data.
    # -------------------------------------------------------------------------
    # model = YOLO("model.pt") 

    print("Starting training...")
    try:
        # data="data.yaml" -> Point this to your dataset configuration file
        # epochs=50 -> Number of times to iterate over the dataset
        # imgsz=640 -> Image size multiple of 32
        results = model.train(data="dataset/data.yaml", epochs=50, imgsz=640, device=0)
        
        print("Training complete!")
        print(f"Best model saved at: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure your 'data.yaml' path is correct and your dataset is properly formatted.")

if __name__ == '__main__':
    train()
