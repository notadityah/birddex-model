# How to Train the Model to Detect More Birds

To teach your model to detect new bird species (e.g., adding "Parrot" to the existing list), you need to follow these steps. You cannot simply "add" a class to the existing file; you usually need to retrain a model on a dataset that includes both the old and new birds.

## Step 1: Collect and Label Data

1.  **Gather Images**: Collect images of the new birds you want to detect (and the old ones if you want to keep detecting them).
2.  **Label Images**: Use a tool to draw bounding boxes around the birds.
    - **Recommended Tools**: [Roboflow](https://roboflow.com/) (easiest, web-based), [CVAT](https://github.com/opencv/cvat), or [LabelImg](https://github.com/heartexlabs/labelImg).
3.  **Export Dataset**: Export your dataset in **YOLOv8** (or YOLOv11) format.
    - This will give you a folder structure like:
      ```
      dataset/
      ├── images/
      │   ├── train/
      │   └── val/
      ├── labels/
      │   ├── train/
      │   └── val/
      └── data.yaml
      ```

## Step 2: Configure the Dataset (`data.yaml`)

Your `data.yaml` file tells the model where to find the images and what the classes are. It should look like this:

```yaml
train: ./dataset/images/train
val: ./dataset/images/val

nc: 8 # Number of classes (Old 7 + New 1)
names:
  [
    "Crow",
    "Kingfisher",
    "Myna",
    "Owl",
    "Peacock",
    "Pigeon",
    "Sparrow",
    "Parrot",
  ]
```

## Step 3: Run Training

Use the `train_model.py` script provided in this folder.

### Option A: Retrain from Scratch (Recommended)

This uses a base pretrained model (like `yolo11s.pt`) and trains on your FULL dataset (old birds + new birds). This produces the best results.

```python
model = YOLO("yolo11s.pt")  # Load base model
model.train(data="data.yaml", epochs=50) # Train on new data
```

### Option B: Fine-tune Existing Model

You can try to continue training `model.pt`, but this is tricky when adding new classes because the model's structure changes (7 outputs -> 8 outputs). It's usually better to use Option A.

## Step 4: Use Your New Model

After training, a new model file (e.g., `runs/detect/train/weights/best.pt`) will be generated.

1.  Rename it to something like `new_model.pt`.
2.  Update your code to use `model = YOLO("new_model.pt")`.
