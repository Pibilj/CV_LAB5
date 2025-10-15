import numpy as np
import os
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision import transforms

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure

#testing

data_path = ".data/cifar-10-batches-py"

download_flag = not os.path.exists(data_path)

transform = transforms.ToTensor()

dataset = CIFAR10(
    root="./data",
    train=False,
    download=download_flag,
    transform=transform
)

N = 10
imgs_gray = []
labels = []

for i in range(N):
    x, y = dataset[i]
    img_rgb = x.permute(1, 2, 0).numpy()
    img_gray = rgb2gray(img_rbg)
    imgs_gray.append(img_gray)
    labels.append(dataset.classes[y])

imgs_gray = np.stack(imgs_gray, axis=0)
print("Loaded greyscale images shape:", imgs_gray.shape)