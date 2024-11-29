import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision import datasets, transforms

import timm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import time
from sklearn import metrics

import zipfile
import fnmatch
from PIL import Image, ImageChops, ImageEnhance

#For texture extraction
from skimage import feature
import os
import csv
import logging
import datetime


# Setup logging
log_dir = 'log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"app_{current_datetime_str}.log")
logging.basicConfig(filename=log_file_path, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = gray_image.astype(np.uint8)  # Convert to integer type
    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(gray_image, n_points, radius, method='uniform')

    lbp = lbp / lbp.max()  # Normalize the LBP values to the range [0, 1]
    
    return lbp

def extract_color_features(image, quality=95, enhance_factor=10):
    # temp_path = "temp_recompressed.jpg"
    # image.save(temp_path, format="JPEG", quality=quality)
    # with Image.open(temp_path) as recompressed:
    #     ela_image = ImageChops.difference(image, recompressed)
    # os.remove(temp_path)
    
    image_bytes = image.tobytes()
    recompressed = Image.frombytes("RGB", image.size, image_bytes)
    ela_image = ImageChops.difference(image, recompressed)

    enhancer = ImageEnhance.Brightness(ela_image)
    enhanced_ela = enhancer.enhance(enhance_factor)

    resized_ela = enhanced_ela.resize((224, 224)).convert("L")
    feature_array = np.array(resized_ela).astype(np.float32) / 255.0
    return feature_array


def extract_shape_features(image):
    kernel_size = 5
    transform_iteration = 5

    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    image = cv2.resize(image, (224, 224))  # Resize to (224, 224)

    image_dict = {}
    image_dict["original_image"] = image
    image_dict["eroded_image"] = cv2.erode(image_dict["original_image"], kernel, iterations=transform_iteration)
    image_dict["dilated_image"] = cv2.dilate(image_dict["original_image"], kernel, iterations=transform_iteration)
    image_dict["opened_image"] = cv2.dilate(image_dict['eroded_image'], kernel, iterations=transform_iteration)
    image_dict["closed_image"] = cv2.erode(image_dict['dilated_image'], kernel, iterations=transform_iteration)

    opened_image_resized = cv2.cvtColor(image_dict["opened_image"], cv2.COLOR_RGB2GRAY)

    return opened_image_resized  # Shape: (224, 224)

def load_image_from_zip(zip_path, img_path):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(img_path) as file:
            img = Image.open(file)
            return img.convert("RGB")  # Ensure the image is in RGB format
        
class ZipImageFolderDataset(datasets.ImageFolder):
    def __init__(self, zip_path, root, transform=None):
        self.zip_path = zip_path
        self.root = root
        self.transform = transform
        self.classes = ['0_real', '1_fake']
        self.img_paths = self._get_image_paths()

    def _get_image_paths(self):
        img_paths = []
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for file_info in zf.infolist():
                name = file_info.filename
                if fnmatch.fnmatch(name, f"{self.root}/*.jpg"):
                    label = 0 if '0_real' in name.split('/')[1] else 1
                    img_paths.append((name, label))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        img = load_image_from_zip(self.zip_path, img_path)
        if self.transform:
            img_tensor = self.transform(img)
        
        # Ensure the image is now a tensor
        if not isinstance(img_tensor, torch.Tensor):
            msg = f"Expected image to be a tensor, but got {type(img_tensor)}."
            logging.critical(msg)
            raise TypeError(msg)
        
        # Convert tensor to numpy array for feature extraction
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        
        # Extract features
        texture_features = extract_texture_features(img_np)
        color_features = extract_color_features(img)
        shape_features = extract_shape_features(img_np)

        img.close()
        
        features = np.stack([texture_features, color_features, shape_features], axis=0)
        features = torch.tensor(features).float().permute(1, 2, 0)  # Change the shape to [height, width, channels]
        features = features.permute(2, 0, 1)  # Change the shape to [channels, height, width]
        
        return features, label
    
class ImageFolderDataset(datasets.ImageFolder):
    def __init__(self, img_path, root, transform=None):
        self.img_path = img_path
        self.root = root
        self.transform = transform
        self.classes = ['0_real', '1_fake']
        self.img_paths = self._get_image_paths()

    def _get_image_paths(self):
        img_paths = []
        root_path = os.path.join(self.img_path, self.root)
        for root, dirs, files in os.walk(root_path):
            for name in files:
                if fnmatch.fnmatch(name, "*.jpg"):
                    label = 0 if '0_real' in root else 1
                    img_paths.append((os.path.join(root, name), label))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        img = Image.open(img_path)
        img = img.convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
        
        # Ensure the image is now a tensor
        if not isinstance(img_tensor, torch.Tensor):
            msg = f"Expected image to be a tensor, but got {type(img_tensor)}."
            logging.critical(msg)
            raise TypeError(msg)
        
        # Convert tensor to numpy array for feature extraction
        img_np = img_tensor.numpy().transpose(1, 2, 0)
        
        # Extract features
        texture_features = extract_texture_features(img_np)
        color_features = extract_color_features(img)
        shape_features = extract_shape_features(img_np)

        img.close()
        
        features = np.stack([texture_features, color_features, shape_features], axis=0)
        features = torch.tensor(features).float().permute(1, 2, 0)  # Change the shape to [height, width, channels]
        features = features.permute(2, 0, 1)  # Change the shape to [channels, height, width]
        
        return features, label
    
def load_data(zip_path, batch_size, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = r"AIGC-Detection-Dataset\train"
    val_dir = r"AIGC-Detection-Dataset\val"
    # test_dir = "AIGC-Detection-Dataset/val"

    train_dataset = ImageFolderDataset(zip_path, train_dir, transform=transform)
    val_dataset = ImageFolderDataset(zip_path, val_dir, transform=transform)
    # test_dataset = ZipImageFolderDataset(zip_path, test_dir, transform=transform)
    logging.info(f"Data prepared:\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    logging.info("Data loaded")
    return train_loader, val_loader

def evaluate(model, test_loader):
    # Validation phase
    model.eval()
    
    y_true = []
    y_pred = []
    
    batch = 0
    logging.info(f"Total batches: {len(test_loader)}")
    for img, label in test_loader:
        # Please make sure that the "pred" is binary result
        output = model(img.to(DEVICE))
        pred = np.argmax(output.detach().to('cpu'), axis=1).numpy()
        
        y_true.extend(label.numpy())
        y_pred.extend(pred)
        
        logging.info(f"Batch: {batch} completed")
        batch += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    logging.info(f'Validation Accuracy: {accuracy}')
    return accuracy

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    model.train()
    best_val_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        running_loss = 0
        logging.info(f"Epoch {epoch+1} started...")
        logging.info(f"length of train_loader: {len(train_loader)}")
        start_time = time.time()
        
        batch = 1
        # for features, labels in train_loader:
        #     # images = images.to(DEVICE)
        #     labels = labels.to(DEVICE)
        #     features = features.to(DEVICE)

        #     optimizer.zero_grad()
        #     outputs = model(features)
            
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss.item() * features.size(0)
        #     logging.info(f'Batch {batch} completed, loss = {loss:.4f}')
        #     batch += 1
        
        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Time: {time.time()-start_time:.2f}s')
        train_losses.append(epoch_loss)
        
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_model(model, f'seresnent_e{epoch+1}.pth')
        else:
            epochs_without_improvement += 1
            
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        if epochs_without_improvement >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break
    return train_losses, val_losses, val_accuracies
        

def save_model(model, path='testing.pth'):
    torch.save(model.state_dict(), path)
    logging.info("Model saved successfully!")

def save_results(train_losses, val_losses, val_accuracies, path='results.csv'):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Val Accuracy'])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], val_losses[i], val_accuracies[i]])

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    logging.info(f"length of val_loader: {len(val_loader)}")
    batch = 1
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch += 1
    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    return val_loss, accuracy

def create_custom_model(pretrained):
    model = timm.create_model('seresnext101_32x4d', pretrained)
    
    # Modify the first convolutional layer
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(3, original_conv1.out_channels, kernel_size=original_conv1.kernel_size,
                        stride=original_conv1.stride, padding=original_conv1.padding, bias=False)
    with torch.no_grad():
        new_conv1.weight[:, :3, :, :] = original_conv1.weight[:, :3, :, :]
    model.conv1 = new_conv1

    # Adjust the final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model

def get_model(filename='seresnext_finetuned.pth', force_new=False):
    file_path = os.path.join(os.getcwd(), filename)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use pre-existing weights
    if os.path.exists(file_path) and not force_new:
        model = create_custom_model(pretrained=False)
        model.load_state_dict(torch.load(file_path, map_location=DEVICE, weights_only=True))
        logging.info("Model loaded successfully!")
        return model
    
    else:   # Create a new model
        model = create_custom_model(pretrained=True)
        logging.info("Creating a new model")
        return model

if __name__ == '__main__':
    # Start Training
    model_weight_filename = 'seresnext_finetuned.pth'
    
    model = get_model(model_weight_filename, force_new=False)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(DEVICE)
    
    # Load the data
    zip_path = 'AIGC-Detection-Dataset'
    batch_size = 32
    image_size = 224
    train_loader, val_loader = load_data(zip_path, batch_size, image_size)
    
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, optimizer, criterion, 50)
    
    # Save the results to a CSV file
    save_results(train_losses, val_losses, val_accuracies)
    
    save_model(model, model_weight_filename)
    # evaluate(model, train_loader)
    # evaluate(model, val_loader)
