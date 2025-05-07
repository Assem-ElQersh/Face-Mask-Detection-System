"""
Face Mask Detection Project
--------------------------
A complete implementation of a face mask detection system using PyTorch.
The system detects whether a person is:
1. Wearing a mask properly
2. Not wearing a mask
3. Wearing a mask incorrectly

This implementation uses transfer learning with MobileNetV2 as the base model
and adds a custom classifier on top.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

# Define constants
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224
CLASSES = ['WithMask', 'WithoutMask']  # Updated class names to match directory names
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# Define dataset class
class FaceMaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load images for each class
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):  # Check if directory exists
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Define data transformations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Define the model 
class FaceMaskDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(FaceMaskDetector, self).__init__()
        
        # Load pre-trained MobileNetV2
        self.base_model = models.mobilenet_v2(pretrained=True)
        
        # Freeze the parameters of the feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Replace the classifier
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.base_model(x)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation accuracy: {val_acc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

# Function to plot training results
def plot_training_results(history):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

# Function to download and prepare dataset
def download_and_prepare_dataset(custom_data_path=None):
    """
    This function uses either a custom data path or downloads the dataset using kagglehub.
    """
    if custom_data_path:
        print(f"Using custom dataset path: {custom_data_path}")
        return custom_data_path
        
    try:
        import kagglehub
        print("Downloading dataset using kagglehub...")
        path = kagglehub.dataset_download("ashishjangra27/face-mask-12k-images-dataset")
        print(f"Dataset downloaded to: {path}")
        
        # Check the directory structure
        print("Analyzing directory structure...")
        for root, dirs, files in os.walk(path):
            if len(files) > 0 and any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files):
                print(f"Found images in: {root}")
                parent_dir = os.path.basename(os.path.dirname(root))
                if parent_dir in ['train', 'test', 'validation', 'Train', 'Test', 'Validation'] and os.path.basename(root) in CLASSES:
                    print(f"Found valid directory: {parent_dir}/{os.path.basename(root)}")
        
        return path
    except ImportError:
        print("kagglehub not installed. Please install it with: pip install kagglehub")
        print("Or download the dataset manually from:")
        print("https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset")
        return 'data'  # Default path if kagglehub fails

# Face detection using OpenCV's Haar Cascade classifier
def detect_faces(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

# Function to preprocess an image for the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict mask status
def predict_mask(model, image, face_coords):
    x, y, w, h = face_coords
    face_image = image[y:y+h, x:x+w]
    
    # Preprocess the face image
    processed_face = preprocess_image(face_image)
    
    # Move to device
    processed_face = processed_face.to(DEVICE)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(processed_face)
        _, predicted = torch.max(outputs, 1)
        probability = torch.softmax(outputs, dim=1)[0][predicted[0]].item()
    
    return CLASSES[predicted[0]], probability

# Function to process an image for mask detection
def process_image(model, image_path):
    # Clean the path - remove quotes and normalize path separators
    image_path = image_path.strip('"\'')
    image_path = os.path.normpath(image_path)
    
    # Load the image
    print(f"Attempting to load image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
        
    # Detect faces
    faces = detect_faces(image)
    
    # Process each face
    for (x, y, w, h) in faces:
        # Predict mask status
        mask_status, probability = predict_mask(model, image, (x, y, w, h))
        
        # Define color based on mask status
        if mask_status == 'with_mask':
            color = (0, 255, 0)  # Green
        else:  # without_mask
            color = (0, 0, 255)  # Red
        
        # Draw rectangle and add label
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, f'{mask_status} ({probability:.2f})', 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

# Function to process a video for mask detection
def process_video(model, video_path=0):  # Default to webcam
    # Open the video stream
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        return
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            break
            
        # Detect faces
        faces = detect_faces(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Predict mask status
            mask_status, probability = predict_mask(model, frame, (x, y, w, h))
            
            # Define color based on mask status
            if mask_status == 'with_mask':
                color = (0, 255, 0)  # Green
            else:  # without_mask
                color = (0, 0, 255)  # Red
            
            # Draw rectangle and add label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{mask_status} ({probability:.2f})', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the frame
        cv2.imshow('Face Mask Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Main function for training
def main():
    # Use custom dataset path for the structure you provided
    data_path = download_and_prepare_dataset(custom_data_path="Face Mask Dataset")
    
    # Get data transformations
    train_transform, val_transform = get_transforms()
    
    # Find the correct paths for train and validation directories
    train_dir = os.path.join(data_path, 'Train')  # Note the capital T
    val_dir = os.path.join(data_path, 'Validation')  # Note the capital V
    
    # Create datasets
    train_dataset = FaceMaskDataset(data_dir=train_dir, transform=train_transform)
    val_dataset = FaceMaskDataset(data_dir=val_dir, transform=val_transform)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = FaceMaskDetector(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    
    # Plot training results
    plot_training_results(history)
    
    # Final message
    print("Training complete! Model saved as 'best_model.pth'")
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = FaceMaskDetector(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    
    # Plot training results
    plot_training_results(history)
    
    # Final message
    print("Training complete! Model saved as 'best_model.pth'")

# Function to load a trained model
def load_model(model_path='best_model.pth'):
    model = FaceMaskDetector(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    return model

# Demo function that lets you try different modes
def demo():
    # Load the trained model
    model = load_model()
    
    print("Face Mask Detection Demo")
    print("------------------------")
    print("1. Process an image")
    print("2. Process a video file")
    print("3. Use webcam")
    print("4. Exit")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == '1':
        image_path = input("Enter the path to the image: ")
        output_image = process_image(model, image_path)
        
        if output_image is not None:
            cv2.imshow('Result', output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save the output
            output_path = 'output_' + os.path.basename(image_path)
            cv2.imwrite(output_path, output_image)
            print(f"Output saved as {output_path}")
    
    elif choice == '2':
        video_path = input("Enter the path to the video: ")
        process_video(model, video_path)
    
    elif choice == '3':
        process_video(model)  # Default to webcam (0)
    
    elif choice == '4':
        print("Exiting...")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Face Mask Detection')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--demo', action='store_true', help='Run the demo')
    
    args = parser.parse_args()
    
    if args.train:
        main()
    elif args.demo:
        demo()
    else:
        print("Please specify an action: --train or --demo")