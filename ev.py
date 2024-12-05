from ultralytics import YOLO
import torch
import os
import fnmatch
import numpy as np
from sklearn import metrics
from PIL import Image
import csv

def data_loder(path):
    img_paths = []
    labels = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, "*.jpg"):
                labels.append(0 if '0_real' in root else 1)
                img_paths.append(os.path.join(root, name))
    return (img_paths, labels)

def evaluate(model, test_loader, batch_size):
    # Validation phase
    model.eval()
    
    y_true = np.array(test_loader[1])
    
    img_paths = test_loader[0]
    y_pred = []

    # Process images in batches
    for i in range(0, len(img_paths), batch_size):
        j = min(i + batch_size, len(img_paths))
        batch_paths = img_paths[i:j]
        results = model.predict(batch_paths, verbose=False)  # Predict batch
        y_pred.extend([result.probs.top1 for result in results])  # Collect predictions

    y_pred = np.array(y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return accuracy

def load_model(model_path):
    model = YOLO(model_path, task='classify')
    model.to(DEVICE)
    print(f'Model loaded from {model_path}')
    return model

def get_model_paths(folder_path):
    paths = []
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if fnmatch.fnmatch(name, "*.pt"):
                paths.append(os.path.join(root, name))
    return paths

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folders = ['yololog2\weights', 'yololog3\weights']
    val_path = r'AIGC-Detection-Dataset\val'
    test_path = r'AIGC-Detection-Dataset\test'
    path, val_acc, test_acc, avg_acc = [], [], [], []
    batch_size = 256
    for folder in folders:
        model_paths = get_model_paths(folder)
        for model_path in model_paths:
            model = load_model(model_path)
            val_loader = data_loder(val_path)
            val_accuracy = evaluate(model, val_loader, batch_size)
            test_loader = data_loder(test_path)
            test_accuracy = evaluate(model, test_loader, batch_size)
            path.append(model_path)
            val_acc.append(val_accuracy)
            test_acc.append(test_accuracy)
            avg_acc.append((val_accuracy + test_accuracy) / 2)
            print(f'Model: {model_path}, val accuracy: {val_accuracy}, Test accuracy: {test_accuracy}, Average accuracy: {(val_accuracy + test_accuracy) / 2}')
        
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Validation Accuracy', 'Test Accuracy', 'Average Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(path)):
            writer.writerow({'Model': path[i], 'Validation Accuracy': val_acc[i], 'Test Accuracy': test_acc[i], 'Average Accuracy': avg_acc[i]})
    
    