import os
import yaml
from ultralytics import YOLO
from pathlib import Path

def train_rgbd_model(
    data_yaml_path,
    model_size='n',
    epochs=100,
    imgsz=640,
    batch=16,
    patience=20,
    device=0
):
    """
    Train a YOLOv8 segmentation model on RGBD (4-channel) data.
    
    Args:
        data_yaml_path: Path to the dataset YAML file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        patience: Early stopping patience
        device: Training device (GPU id or 'cpu')
    
    Returns:
        Trained model
    """
    print(f"Starting training with YOLOv8{model_size}-seg on RGBD data...")
    
    # Verify the dataset YAML has channels=4
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    if 'channels' not in data_config or data_config['channels'] != 4:
        print(f"Warning: Dataset YAML does not specify channels=4. Adding it...")
        data_config['channels'] = 4
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    # Load pre-trained YOLOv8 segmentation model
    # The first layer will be automatically adapted for 4 channels
    model = YOLO(f'yolov8{model_size}-seg.pt')
    
    # Print model summary to verify the first layer has 4 input channels
    model.info()
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        optimizer='Adam',
        lr0=0.001,
        lrf=0.1,
        save=True,
        plots=True
    )
    
    print(f"Training completed. Model saved at: {model.trainer.best}")
    
    # Validate the model
    print("Validating model...")
    val_results = model.val()
    
    # Save the final model
    save_dir = Path(model.trainer.save_dir)
    final_model_path = save_dir / 'rgbd_model.pt'
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    return model

if __name__ == "__main__":
    # Check if RGBD dataset exists, otherwise create it
    if not Path("./data/rgbd.yaml").exists():
        print("RGBD dataset not found. Creating it first...")
        from rgbd_dataset import create_rgbd_dataset
        dataset_path = create_rgbd_dataset()
        data_yaml_path = str(dataset_path / "rgbd.yaml")
    else:
        data_yaml_path = "./data/rgbd.yaml"
    
    # Train model
    model = train_rgbd_model(
        data_yaml_path=data_yaml_path,
        model_size='n',  # Use nano version as requested
        epochs=100,
        imgsz=640,
        batch=16,
        patience=5
    ) 