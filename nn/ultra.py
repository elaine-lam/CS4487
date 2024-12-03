from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="AIGC-Detection-Dataset\AIGC-Detection-Dataset", epochs=40, imgsz=224, project='', dropout=0.3)  # train the model for 50 epochs

    # model.save("fine.pt")  # save the model to file
    model._save_to_state_dict("fine_state_dict.pt")  # save the model to state_dict
