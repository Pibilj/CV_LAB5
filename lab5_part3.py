import numpy as np
import os
import matplotlib.pyplot as plt
import torch

from torchvision.datasets import ImageFolder
from torchvision import transforms

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

#Writting the images and labels to stacks
labels = []
images = []
for i in range(len(dataset)):
    img, label = dataset[i]
    labels.append(label)
    images.append(img)

labels = torch.tensor(labels)
images = torch.stack(images)
