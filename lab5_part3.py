import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Check if we can use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#loading and normalizing the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "fruit_images")

transform = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ]
)

dataset = ImageFolder(root=data_path, transform=transform)

#Stacking into tensors
labels = []
images = []
for i in range(len(dataset)):
    img, label = dataset[i]
    labels.append(label)
    images.append(img)

labels = torch.tensor(labels)
images = torch.stack(images)

#converting to a 1D Vector
images_flat = images.view(images.size(0), -1)

#Making the stratisized split
x_train, x_test, y_train, y_test = train_test_split(
    images_flat.numpy(),
    labels.numpy(),
    test_size = 0.2,
    stratify = labels.numpy(),
    random_state = 42
)


#Converting x & y test and train back into tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#creating the test and train data sets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

#Creating Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

