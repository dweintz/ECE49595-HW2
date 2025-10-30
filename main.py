import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transformers
import matplotlib.pyplot as plt


class H2Dataset(Dataset):
    def __init__(self, dataset_folder, list_file, transform=None)
        self.dataset_folder = dataset_folder
        self.transform = transform

        # load list of images (train.txt or test.txt)
        list_path = os.path.join(dataset_folder, list_file)
        with open(list_file, "r") as f:
            self.image_list = [line.strip() for line in f.readlines()]
        
        # get class names (read folder from image name)
        class_names = []
        for img_name in self.image_list:
            folder_name = img_name.split('.')[0]
            if folder_name not in class_names:
                class_names.append(folder_name)

        # create a class map from name to numeric index
        self.class_to_idx = {}
        for idx, name in enumerate(self.classes):
            self.class_to_idx[name] = idx
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # obtain path to the image
        rel_path = self.image_list[idx]
        class_name, filename = rel_path.split('.', 1)
        img_path = os.path.join(self.dataset_folder, class_name, filename)

        # open the image
        image = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[class_name]

        # apply any transforms
        if self.transform:
            image - self.transform(image)
        
        return image, label
        

class SimpleCNN(nn.Module):
    def __init__(self)
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
            nn.Linear(128 * 12 * 12, 256),  # assuming input 96x96
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
    pass    


# function to evaluate the model
def evaluate(model, data_loader, device):
    pass


if __name__ == "__main__":
    dataset_folder = "h2-data"
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
