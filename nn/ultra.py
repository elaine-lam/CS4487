from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="AIGC-Detection-Dataset\AIGC-Detection-Dataset", epochs=50, imgsz=224)

    model.save("fine.pt")  # save the model to file
