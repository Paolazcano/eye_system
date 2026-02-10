from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,          # Epocas: Número de veces que el modelo ve todo el dataset (100 es el de ley)
        imgsz=640,           # Tamaño al que se redimensionan las imágenes (Se puede intentar con 768)
        batch=8,             # Número de imágenes procesadas al mismo tiempo (Depende del GPU)
        optimizer="AdamW",   # Optimizador usado para ajustar los pesos
        lr0=0.001,           # Learning rate inicial (Valor estándar para AdamW)
        workers=2,           # Número de "trabajadores"para cargar datos
        project="models/yolo",
        name="eye_detector"
    )

if __name__ == "__main__":
    main()



