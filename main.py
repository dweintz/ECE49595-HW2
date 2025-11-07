import os
import random
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time


# dataset class for loading and manipulating data
class H2Dataset(Dataset):
    def __init__(self, dataset_folder, list_file, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform

        # get a list of images from file (train.txt or test.txt)
        list_path = os.path.join(dataset_folder, list_file)
        with open(list_path, "r") as f:
            self.image_list = [line.strip() for line in f.readlines()]

        # get class names (read folder from image name)
        class_names = []
        for img_name in self.image_list:
            folder_name = img_name.split('_')[0]
            if folder_name not in class_names:
                class_names.append(folder_name)
        self.classes = sorted(class_names)

        # create a class map from name to numeric index
        self.class_to_idx = {}
        for idx, name in enumerate(self.classes):
            self.class_to_idx[name] = idx

        # load images
        print("Preloading images...")
        self.images = []
        self.labels = []
        for filename in self.image_list:
            # get path to image
            class_name = filename.split('_')[0]
            img_path = os.path.join(dataset_folder, class_name, filename)
            print(f"    Loading image: {img_path}")

            # open image, apply any transforms
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            self.images.append(image)
            self.labels.append(self.class_to_idx[class_name])
        print("\nPreloading complete.\n")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
        

# class to define network architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # define CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # define CNN fully connected layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    # define forward pass through network
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# function to train the model
def train_model(model, train_loader, test_loader, device, epochs, lr):
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accuracies = []

    # compute each epoch
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # iterate over batches
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()   

        # compute average loss
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # compute accuracy on test set
        acc = evaluate(model, test_loader, device)
        test_accuracies.append(acc)

        print(f"    Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Test Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "cnn_model.pth")
    print("\nModel saved as cnn_model.pth")


# function to evaluate the model
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


# function to run predictions on images and save to folder
def predict_random_samples(model, dataset_folder, list_file, transform,
                           class_names, device, output_folder, num_samples=20):
    # create output folder
    os.makedirs(output_folder, exist_ok=True)
    # model.eval()

    # load test filenames
    list_path = os.path.join(dataset_folder, list_file)
    with open(list_path, "r") as f:
        all_images = [line.strip() for line in f.readlines()]

    # randomly pick subset
    sample_images = random.sample(all_images, min(num_samples, len(all_images)))

    print(f"\nStarting inference on {num_samples} samples.\n")

    for filename in sample_images:
        # open image
        class_name = filename.split('_')[0]
        img_path = os.path.join(dataset_folder, class_name, filename)
        image = Image.open(img_path).convert("RGB")

        # prepare tensor
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # inference
        print(f"    Running inference on {img_path}")
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            pred_class = class_names[pred.item()]

        # annotate
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        # text settings
        text = f"Predicted: {pred_class}"
        font_size = 60 

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # padding and background rectangle (highlight)
        padding = 6
        bg_x0, bg_y0 = 10 - padding, 10 - padding
        bg_x1, bg_y1 = 10 + text_width + padding, 10 + text_height + padding
        draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=(255, 255, 255, 180))

        # draw text on top
        draw.text((10, 10), text, fill=(255, 0, 0), font=font)

        # save annotated image
        save_path = os.path.join(output_folder, filename)
        annotated.save(save_path)
        print(f"    Saved annotated image: {save_path}")

    print("\nRandom sample prediction complete.")
    print(f"See annotated images in: {output_folder}.\n")


if __name__ == "__main__":
    start_time = time.time()

    # configure hyperparameters
    dataset_folder = "h2-data"
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001

    # set to True if you want to run predictions on sample images and save
    SAMPLE_PREDICTIONS = True

    # configure transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # datasets
    train_dataset = H2Dataset(dataset_folder, "train.txt", transform)
    test_dataset = H2Dataset(dataset_folder, "test.txt", transform)

    # loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=len(train_dataset.classes)).to(device)

    print(f"Training on {device}.\n")

    train_model(model, train_loader, test_loader, 
               device, num_epochs, learning_rate)
    
    # optionally run inference on random samples and display class on image
    if SAMPLE_PREDICTIONS:
        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load saved model
        model = SimpleCNN(num_classes=4).to(device)
        model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
        model.eval()

        # run 20 sample predictions and save to folder
        if not model:
            print("Model not loaded. Skipping predictions.")
        else:
            predict_random_samples(
                model=model,
                dataset_folder=dataset_folder,
                list_file="test.txt",
                transform=transform,
                class_names=["Egyptian Cat", "African Elephant", "Mountain Bike", "Banana"],
                device=device,
                output_folder="sample_predictions",
                num_samples=20
            )
    
    # compute latency
    end_time = time.time()
    latency = end_time - start_time
    print(f"Total execution time: {latency:.2f} seconds ({latency / 60:.2f} min).\n")