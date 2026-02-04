from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=0.001,
        workers=2,
        project="models/yolo",
        name="eye_detector"
    )

if __name__ == "__main__":
    main()



