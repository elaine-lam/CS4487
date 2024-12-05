from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model    
    model = YOLO("yolo11x-cls.pt", task='classify')
    
    data_path = 'AIGC-Detection-Dataset'
    # Train the model
    #embed to add our own feature extraction
    results = model.train(
            data=data_path, 
            epochs=40, 
            imgsz=224, 
            project='', 
            name='C:\CS4487\yololog', 
            overlap_mask=False, 
            save_period=1, 
            mask_ratio=0, 
            dropout=0.1,
            close_mosaic=40, 
            device=0, 
            rect=False,
            hsv_h=0, 
            hsv_s=0, 
            hsv_v=0, 
            translate=0,
            scale=0,
            fliplr=0, 
            erasing=0,
            mosaic=False,
            auto_augment=''
        )