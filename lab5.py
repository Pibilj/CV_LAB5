import numpy as np
import os
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision import transforms

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from skimage.transform import resize


#fetching and grayscaling the images from the dataset

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

#This method is for showing 10 of the images and their label
def show_images_grid(imgs, labels=None, rows=2, cols=5, title="CIFAR-10 Samples", display_size = (32,32)):
    fig, axes = plt.subplots(rows,cols, figsize=(12,5))
    fig.subtitle(title, fontsize=14)
    for i in range(rows * cols):
        ax = axes[i// cols, i % cols]
        ax.axis('off')
        if i < len(imgs):
            img_resized = resize(imgs[i], display_size, anti_aliasing=True)
            if img_resized.ndim == 2:
                ax.imshow(img_resized, interpolation="nearest")
            if labels is not None:
                ax.set_title(labels[i], fonstsize=9)
    plt.tight_layout()
    plt.show()

show_images_grid(imgs_gray[:10], labels[:10], rows = 2, cols=5, display_size=(64,64))

#computing the hog

features_list = []
hog_imgs = []

for i in range(len(imgs_gray)):
    feat, hog_img = hog(
        imgs_gray[i],
        orientations=9,
        pixels_per_cell=(8,8),
        cells_per_block=(2,2),
        block_norm="L2-Hys",
        visualization=True,
    )
    print(feat.shape)
    features_list.append(feat)
    hog_imgs.append(exposure.rescale_intensity(hog_img, in_range=(0,hog_img.max())))

print("number of images processed:" , len(features_list))
print("HOG feature length (first image):", len(features_list[0]))

#displaying the HOGed images
def show_hog_grid(hog_images, rows=2, cols=5, title="HOG Renderings"):
    fig, axes = plt.subplots(rows, cols, figuresize=(12, 5))
    fig.subtitle(title, fontsize=14)
    for i in range (rows*cols):
        ax = axes[i // cols, i % cols]
        ax.axis('off')
        if i < len(hog_images):
            ax.imshow(hog_images[1], cmap="gray")
    plt.tight_layout()
    plt.show()

show_hog_grid(hog_imgs[:10], rows = 2, cols = 5)