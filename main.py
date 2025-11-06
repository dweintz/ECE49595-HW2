import os
from PIL import Image
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
            print(f"  Loading image: {img_path}")

            # open image, apply any transforms
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            self.images.append(image)
            self.labels.append(self.class_to_idx[class_name])
        print("Preloading complete.")

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

        print(f"  Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Test Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "cnn_model.pth")
    print("Model saved as cnn_model.pth")


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


if __name__ == "__main__":
    start_time = time.time()

    # configure hyperparameters
    dataset_folder = "h2-data"
    batch_size = 16
    num_epochs = 20
    learning_rate = 0.001

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

    print(f"Training on {device}")

    train_model(model, train_loader, test_loader, 
               device, num_epochs, learning_rate)
    
    # compute latency
    end_time = time.time()
    latency = end_time - start_time
    print(f"Total execution time: {latency:.2f} seconds ({latency / 60:.2f} min).")
